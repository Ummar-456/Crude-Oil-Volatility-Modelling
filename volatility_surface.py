# src/models/volatility_surface.py

import numpy as np
import pandas as pd
import scipy.optimize as optimize
from scipy.interpolate import griddata, SmoothBivariateSpline
from typing import Dict, List, Optional, Tuple, Union, Callable
import logging
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class VolatilitySurface:
    """
    A class to construct, calibrate, and visualize volatility surfaces for commodity options.
    
    This class implements various methods for volatility surface modeling, including:
    - Interpolation techniques (linear, cubic, thin-plate spline)
    - Parametric models (SVI, SABR)
    - Local volatility models
    - Stochastic-local volatility hybrid models
    """
    
    def __init__(self):
        """Initialize the VolatilitySurface object."""
        self.calibrated_params = {}
        self.surface_data = None
        self.model_type = None
        self.interpolation_method = None
    
    def preprocess_options_data(
        self, 
        calls_data: pd.DataFrame, 
        puts_data: pd.DataFrame,
        underlying_price: float,
        risk_free_rate: float = 0.05,
        filter_illiquid: bool = True,
        min_volume: int = 10,
        use_put_call_parity: bool = True
    ) -> pd.DataFrame:
        """
        Preprocess options data to prepare for volatility surface construction.
        
        Args:
            calls_data: DataFrame containing call options data
            puts_data: DataFrame containing put options data
            underlying_price: Current price of the underlying commodity futures
            risk_free_rate: Risk-free interest rate (annualized)
            filter_illiquid: Whether to filter out illiquid options
            min_volume: Minimum volume for an option to be considered liquid
            use_put_call_parity: Whether to use put-call parity to validate implied volatilities
            
        Returns:
            DataFrame with processed options data including implied volatilities
        """
        # Ensure required columns exist
        required_cols = ['strike', 'lastPrice', 'volume', 'openInterest', 'expiration']
        for df in [calls_data, puts_data]:
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Combine calls and puts
        calls = calls_data.copy()
        puts = puts_data.copy()
        
        # Add option type identifier
        calls['option_type'] = 'call'
        puts['option_type'] = 'put'
        
        # Combine into a single DataFrame
        options = pd.concat([calls, puts], ignore_index=True)
        
        # Filter illiquid options if requested
        if filter_illiquid:
            before_count = len(options)
            options = options[options['volume'] >= min_volume]
            after_count = len(options)
            logger.info(f"Filtered out {before_count - after_count} illiquid options")
        
        # Calculate time to expiration in years
        if isinstance(options['expiration'].iloc[0], str):
            options['expiration'] = pd.to_datetime(options['expiration'])
        
        current_date = datetime.now()
        options['time_to_expiry'] = options['expiration'].apply(
            lambda x: max((x - current_date).days / 365.0, 0.001)  # Ensure minimum value to avoid division by zero
        )
        
        # Calculate moneyness (K/S)
        options['moneyness'] = options['strike'] / underlying_price
        
        # Calculate log-moneyness (log(K/S))
        options['log_moneyness'] = np.log(options['moneyness'])
        
        # Calculate normalized log-moneyness (log(K/S)/sqrt(T))
        options['normalized_log_moneyness'] = options['log_moneyness'] / np.sqrt(options['time_to_expiry'])
        
        # Calculate implied volatility using Black-Scholes model
        options['implied_vol'] = options.apply(
            lambda row: self._calculate_implied_volatility(
                option_price=row['lastPrice'],
                underlying_price=underlying_price,
                strike_price=row['strike'],
                time_to_expiry=row['time_to_expiry'],
                risk_free_rate=risk_free_rate,
                option_type=row['option_type']
            ),
            axis=1
        )
        
        # Filter out invalid implied volatilities
        options = options[options['implied_vol'].between(0.001, 2.0)]
        
        # If using put-call parity, validate and adjust implied volatilities
        if use_put_call_parity:
            options = self._validate_with_put_call_parity(
                options, underlying_price, risk_free_rate
            )
        
        return options
    
    def _calculate_implied_volatility(
        self,
        option_price: float,
        underlying_price: float,
        strike_price: float,
        time_to_expiry: float,
        risk_free_rate: float,
        option_type: str
    ) -> float:
        """
        Calculate implied volatility using the Black-Scholes model.
        
        Args:
            option_price: Market price of the option
            underlying_price: Price of the underlying asset
            strike_price: Strike price of the option
            time_to_expiry: Time to expiration in years
            risk_free_rate: Risk-free interest rate (annualized)
            option_type: 'call' or 'put'
            
        Returns:
            Implied volatility value
        """
        # Define the objective function (difference between model price and market price)
        def objective(volatility):
            try:
                model_price = self._black_scholes_price(
                    underlying_price, strike_price, time_to_expiry, 
                    risk_free_rate, volatility, option_type
                )
                return abs(model_price - option_price)
            except:
                return float('inf')
        
        # Initial guess and bounds
        initial_guess = 0.2  # 20% volatility as initial guess
        bounds = (0.001, 2.0)  # Reasonable bounds for volatility
        
        try:
            # Optimize to find implied volatility
            result = optimize.minimize_scalar(
                objective, 
                bounds=bounds, 
                method='bounded',
                options={'xatol': 1e-6}
            )
            
            if result.success:
                return result.x
            else:
                logger.warning(f"Implied volatility calculation failed: {result.message}")
                return np.nan
                
        except Exception as e:
            logger.warning(f"Error in implied volatility calculation: {e}")
            return np.nan
    
    def _black_scholes_price(
        self,
        S: float,  # Underlying price
        K: float,  # Strike price
        T: float,  # Time to expiry in years
        r: float,  # Risk-free rate
        sigma: float,  # Volatility
        option_type: str  # 'call' or 'put'
    ) -> float:
        """
        Calculate option price using the Black-Scholes model.
        
        Args:
            S: Underlying price
            K: Strike price
            T: Time to expiry in years
            r: Risk-free rate
            sigma: Volatility
            option_type: 'call' or 'put'
            
        Returns:
            Option price according to Black-Scholes model
        """
        if sigma <= 0 or T <= 0:
            return 0.0
        
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        if option_type.lower() == 'call':
            price = S * self._norm_cdf(d1) - K * np.exp(-r * T) * self._norm_cdf(d2)
        else:  # put
            price = K * np.exp(-r * T) * self._norm_cdf(-d2) - S * self._norm_cdf(-d1)
        
        return price
    
    def _norm_cdf(self, x: float) -> float:
        """
        Calculate the cumulative distribution function of the standard normal distribution.
        
        Args:
            x: Input value
            
        Returns:
            CDF value
        """
        return (1.0 + np.math.erf(x / np.sqrt(2.0))) / 2.0
    
    def _validate_with_put_call_parity(
        self,
        options: pd.DataFrame,
        underlying_price: float,
        risk_free_rate: float,
        tolerance: float = 0.1
    ) -> pd.DataFrame:
        """
        Validate and adjust implied volatilities using put-call parity.
        
        Args:
            options: DataFrame containing options data
            underlying_price: Current price of the underlying asset
            risk_free_rate: Risk-free interest rate (annualized)
            tolerance: Maximum allowed deviation in put-call parity
            
        Returns:
            DataFrame with validated implied volatilities
        """
        # Group options by expiration and strike
        grouped = options.groupby(['expiration', 'strike'])
        validated_options = []
        
        for (expiry, strike), group in grouped:
            # If we have both call and put for this strike and expiry
            if len(group) == 2 and set(group['option_type']) == {'call', 'put'}:
                call = group[group['option_type'] == 'call'].iloc[0]
                put = group[group['option_type'] == 'put'].iloc[0]
                
                # Check put-call parity: C - P = S - K*exp(-rT)
                time_to_expiry = call['time_to_expiry']  # Same for both
                parity_rhs = underlying_price - strike * np.exp(-risk_free_rate * time_to_expiry)
                parity_lhs = call['lastPrice'] - put['lastPrice']
                
                # If parity holds within tolerance
                if abs(parity_lhs - parity_rhs) <= tolerance:
                    # Average the implied volatilities
                    avg_vol = (call['implied_vol'] + put['implied_vol']) / 2
                    
                    # Update both options with the average implied volatility
                    call_copy = call.copy()
                    put_copy = put.copy()
                    
                    call_copy['implied_vol'] = avg_vol
                    put_copy['implied_vol'] = avg_vol
                    
                    validated_options.append(call_copy)
                    validated_options.append(put_copy)
                else:
                    # If parity doesn't hold, keep the option with higher volume
                    if call['volume'] >= put['volume']:
                        validated_options.append(call)
                    else:
                        validated_options.append(put)
            else:
                # If we don't have both call and put, keep all options in the group
                for _, row in group.iterrows():
                    validated_options.append(row)
        
        # Combine into a new DataFrame
        validated_df = pd.DataFrame(validated_options)
        
        logger.info(f"Validated {len(validated_df)} options using put-call parity")
        
        return validated_df
    
    def fit_interpolated_surface(
        self,
        options_data: pd.DataFrame,
        method: str = 'cubic',
        grid_points: int = 50
    ) -> Dict:
        """
        Fit an interpolated volatility surface.
        
        Args:
            options_data: Preprocessed options data with implied volatilities
            method: Interpolation method ('linear', 'cubic', 'thin_plate')
            grid_points: Number of grid points in each dimension
            
        Returns:
            Dictionary containing the fitted surface data
        """
        # Extract relevant data
        x = options_data['log_moneyness'].values
        y = options_data['time_to_expiry'].values
        z = options_data['implied_vol'].values
        
        # Create a grid for interpolation
        xi = np.linspace(min(x) - 0.1, max(x) + 0.1, grid_points)
        yi = np.linspace(min(y) - 0.01, max(y) + 0.1, grid_points)
        XI, YI = np.meshgrid(xi, yi)
        
        # Perform interpolation
        if method == 'thin_plate':
            # Use Smooth Bivariate Spline for thin plate spline
            # Limit the number of points for computational efficiency
            max_points = 200
            if len(x) > max_points:
                indices = np.random.choice(len(x), max_points, replace=False)
                x_sample = x[indices]
                y_sample = y[indices]
                z_sample = z[indices]
            else:
                x_sample = x
                y_sample = y
                z_sample = z
                
            spline = SmoothBivariateSpline(
                x_sample, y_sample, z_sample, 
                kx=3, ky=3, s=0.1
            )
            ZI = spline(xi, yi, grid=True)
        else:
            # Use griddata for linear and cubic interpolation
            ZI = griddata((x, y), z, (XI, YI), method=method, fill_value=np.nan)
            
            # Fill NaN values with nearest neighbor interpolation
            mask = np.isnan(ZI)
            if np.any(mask):
                ZI[mask] = griddata(
                    (x, y), z, (XI[mask], YI[mask]), 
                    method='nearest'
                )
        
        # Store the results
        self.surface_data = {
            'x_grid': XI,
            'y_grid': YI,
            'z_grid': ZI,
            'x_raw': x,
            'y_raw': y,
            'z_raw': z,
            'x_label': 'Log Moneyness (log(K/S))',
            'y_label': 'Time to Expiry (years)',
            'z_label': 'Implied Volatility'
        }
        
        self.model_type = 'interpolation'
        self.interpolation_method = method
        
        logger.info(f"Fitted interpolated volatility surface using {method} method")
        
        return self.surface_data
    
    def fit_svi_surface(
        self,
        options_data: pd.DataFrame,
        initial_params: Dict = None
    ) -> Dict:
        """
        Fit a Stochastic Volatility Inspired (SVI) parametric surface.
        
        Args:
            options_data: Preprocessed options data with implied volatilities
            initial_params: Initial parameter values for optimization
            
        Returns:
            Dictionary containing the fitted surface parameters and data
        """
        # Group options by expiration
        grouped = options_data.groupby('expiration')
        
        # Store parameters for each expiration
        expiry_params = {}
        all_fitted_vols = []
        
        # Default initial parameters if not provided
        if initial_params is None:
            initial_params = {
                'a': 0.04,  # Overall level
                'b': 0.4,   # Slope of the wings
                'rho': -0.5, # Correlation (controls skew)
                'm': 0.0,   # Shift along moneyness
                'sigma': 0.1 # Width of the smile
            }
        
        # Fit SVI parameters for each expiration
        for expiry, group in grouped:
            # Extract data for this expiration
            k = group['log_moneyness'].values
            v = group['implied_vol'].values
            t = group['time_to_expiry'].iloc[0]  # Time to expiry is the same for all options in this group
            
            # Initial parameters for this expiry
            params_0 = [
                initial_params['a'],
                initial_params['b'],
                initial_params['rho'],
                initial_params['m'],
                initial_params['sigma']
            ]
            
            # Define the objective function (sum of squared errors)
            def objective(params):
                a, b, rho, m, sigma = params
                # Ensure parameters are within bounds
                if abs(rho) >= 1 or sigma <= 0 or b < 0:
                    return 1e10
                
                # Calculate SVI implied variance
                svi_var = self._svi_variance(k, a, b, rho, m, sigma)
                svi_vol = np.sqrt(svi_var)
                
                # Return sum of squared errors
                return np.sum((svi_vol - v) ** 2)
            
            # Bounds for parameters
            bounds = [
                (0.001, 0.5),    # a: overall level
                (0.001, 2.0),    # b: slope of wings
                (-0.999, 0.999), # rho: correlation
                (-1.0, 1.0),     # m: shift
                (0.001, 1.0)     # sigma: width
            ]
            
            try:
                # Optimize to find SVI parameters
                result = optimize.minimize(
                    objective, 
                    params_0, 
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 1000}
                )
                
                if result.success:
                    # Extract optimized parameters
                    a, b, rho, m, sigma = result.x
                    
                    # Store parameters for this expiry
                    expiry_params[expiry] = {
                        'a': a,
                        'b': b,
                        'rho': rho,
                        'm': m,
                        'sigma': sigma,
                        'time_to_expiry': t
                    }
                    
                    # Calculate fitted volatilities
                    k_values = group['log_moneyness'].values
                    fitted_var = self._svi_variance(k_values, a, b, rho, m, sigma)
                    fitted_vol = np.sqrt(fitted_var)
                    
                    # Store fitted values
                    for i, idx in enumerate(group.index):
                        all_fitted_vols.append({
                            'index': idx,
                            'expiry': expiry,
                            'log_moneyness': k_values[i],
                            'time_to_expiry': t,
                            'market_vol': group['implied_vol'].iloc[i],
                            'fitted_vol': fitted_vol[i]
                        })
                    
                    logger.info(f"Fitted SVI parameters for expiry {expiry}: a={a:.4f}, b={b:.4f}, rho={rho:.4f}, m={m:.4f}, sigma={sigma:.4f}")
                else:
                    logger.warning(f"SVI optimization failed for expiry {expiry}: {result.message}")
            except Exception as e:
                logger.error(f"Error fitting SVI parameters for expiry {expiry}: {e}")

        # Create a DataFrame with all fitted values
        fitted_df = pd.DataFrame(all_fitted_vols)
        
        # Calculate grid for visualization
        if expiry_params:
            # Create a grid of log moneyness and time to expiry values
            k_min = options_data['log_moneyness'].min() - 0.1
            k_max = options_data['log_moneyness'].max() + 0.1
            t_min = options_data['time_to_expiry'].min()
            t_max = options_data['time_to_expiry'].max() + 0.1
            
            k_grid = np.linspace(k_min, k_max, 50)
            t_grid = np.linspace(t_min, t_max, 50)
            
            K, T = np.meshgrid(k_grid, t_grid)
            Z = np.zeros_like(K)
            
            # Interpolate SVI parameters across time dimension
            times = np.array([params['time_to_expiry'] for params in expiry_params.values()])
            a_values = np.array([params['a'] for params in expiry_params.values()])
            b_values = np.array([params['b'] for params in expiry_params.values()])
            rho_values = np.array([params['rho'] for params in expiry_params.values()])
            m_values = np.array([params['m'] for params in expiry_params.values()])
            sigma_values = np.array([params['sigma'] for params in expiry_params.values()])
            
            # For each point in the grid, interpolate parameters and calculate volatility
            for i in range(len(t_grid)):
                t = t_grid[i]
                
                # Interpolate parameters for this time to expiry
                if len(times) > 1:
                    a = np.interp(t, times, a_values)
                    b = np.interp(t, times, b_values)
                    rho = np.interp(t, times, rho_values)
                    m = np.interp(t, times, m_values)
                    sigma = np.interp(t, times, sigma_values)
                else:
                    # If only one expiry is available, use its parameters
                    a = a_values[0]
                    b = b_values[0]
                    rho = rho_values[0]
                    m = m_values[0]
                    sigma = sigma_values[0]
                
                # Calculate SVI variance for this row of the grid
                Z[i, :] = np.sqrt(self._svi_variance(k_grid, a, b, rho, m, sigma))
            
            # Store the results
            self.surface_data = {
                'x_grid': K,
                'y_grid': T,
                'z_grid': Z,
                'x_raw': options_data['log_moneyness'].values,
                'y_raw': options_data['time_to_expiry'].values,
                'z_raw': options_data['implied_vol'].values,
                'x_label': 'Log Moneyness (log(K/S))',
                'y_label': 'Time to Expiry (years)',
                'z_label': 'Implied Volatility',
                'fitted_values': fitted_df,
                'parameters': expiry_params
            }
            
            self.model_type = 'svi'
            self.calibrated_params = expiry_params
            
            logger.info(f"Fitted SVI volatility surface with {len(expiry_params)} expiration dates")
            
            return self.surface_data
        else:
            logger.error("Failed to fit SVI surface: no valid parameters found")
            return None
    
    def _svi_variance(
        self, 
        k: np.ndarray, 
        a: float, 
        b: float, 
        rho: float, 
        m: float, 
        sigma: float
    ) -> np.ndarray:
        """
        Calculate implied variance using the SVI parameterization.
        
        Args:
            k: Log moneyness values (log(K/S))
            a: Overall level parameter
            b: Slope of the wings parameter
            rho: Correlation parameter (controls skew)
            m: Shift along moneyness parameter
            sigma: Width of the smile parameter
            
        Returns:
            Array of implied variance values
        """
        # SVI formula for implied variance
        return a + b * (rho * (k - m) + np.sqrt((k - m)**2 + sigma**2))
    
    def fit_sabr_surface(
        self,
        options_data: pd.DataFrame,
        initial_params: Dict = None,
        forward_curve: Dict = None
    ) -> Dict:
        """
        Fit a SABR (Stochastic Alpha Beta Rho) parametric surface.
        
        Args:
            options_data: Preprocessed options data with implied volatilities
            initial_params: Initial parameter values for optimization
            forward_curve: Dictionary mapping expiry to forward price
            
        Returns:
            Dictionary containing the fitted surface parameters and data
        """
        # Group options by expiration
        grouped = options_data.groupby('expiration')
        
        # Store parameters for each expiration
        expiry_params = {}
        all_fitted_vols = []
        
        # Default initial parameters if not provided
        if initial_params is None:
            initial_params = {
                'alpha': 0.2,  # Initial volatility
                'beta': 0.5,   # CEV parameter (0 for normal, 1 for lognormal)
                'rho': -0.3,   # Correlation between price and vol
                'nu': 0.4      # Volatility of volatility
            }
        
        # Fit SABR parameters for each expiration
        for expiry, group in grouped:
            # Extract data for this expiration
            strikes = group['strike'].values
            market_vols = group['implied_vol'].values
            t = group['time_to_expiry'].iloc[0]  # Time to expiry is the same for all options in this group
            
            # Get forward price for this expiry
            if forward_curve is not None and expiry in forward_curve:
                forward = forward_curve[expiry]
            else:
                # Use the underlying price as an approximation
                forward = group['underlying_price'].iloc[0] if 'underlying_price' in group.columns else 100.0
            
            # Initial parameters for this expiry
            params_0 = [
                initial_params['alpha'],
                initial_params['beta'],
                initial_params['rho'],
                initial_params['nu']
            ]
            
            # Define the objective function (sum of squared errors)
            def objective(params):
                alpha, beta, rho, nu = params
                # Ensure parameters are within bounds
                if abs(rho) >= 1 or alpha <= 0 or nu <= 0 or beta < 0 or beta > 1:
                    return 1e10
                
                # Calculate SABR implied volatilities
                sabr_vols = np.array([
                    self._sabr_volatility(forward, K, t, alpha, beta, rho, nu)
                    for K in strikes
                ])
                
                # Return sum of squared errors
                return np.sum((sabr_vols - market_vols) ** 2)
            
            # Bounds for parameters
            bounds = [
                (0.001, 1.0),    # alpha: initial volatility
                (0.0, 1.0),      # beta: CEV parameter
                (-0.999, 0.999), # rho: correlation
                (0.001, 2.0)     # nu: volatility of volatility
            ]
            
            try:
                # Optimize to find SABR parameters
                result = optimize.minimize(
                    objective, 
                    params_0, 
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 1000}
                )
                
                if result.success:
                    # Extract optimized parameters
                    alpha, beta, rho, nu = result.x
                    
                    # Store parameters for this expiry
                    expiry_params[expiry] = {
                        'alpha': alpha,
                        'beta': beta,
                        'rho': rho,
                        'nu': nu,
                        'forward': forward,
                        'time_to_expiry': t
                    }
                    
                    # Calculate fitted volatilities
                    fitted_vol = np.array([
                        self._sabr_volatility(forward, K, t, alpha, beta, rho, nu)
                        for K in strikes
                    ])
                    
                    # Store fitted values
                    for i, idx in enumerate(group.index):
                        all_fitted_vols.append({
                            'index': idx,
                            'expiry': expiry,
                            'strike': strikes[i],
                            'log_moneyness': np.log(strikes[i] / forward),
                            'time_to_expiry': t,
                            'market_vol': market_vols[i],
                            'fitted_vol': fitted_vol[i]
                        })
                    
                    logger.info(f"Fitted SABR parameters for expiry {expiry}: alpha={alpha:.4f}, beta={beta:.4f}, rho={rho:.4f}, nu={nu:.4f}")
                else:
                    logger.warning(f"SABR optimization failed for expiry {expiry}: {result.message}")
            except Exception as e:
                logger.error(f"Error fitting SABR parameters for expiry {expiry}: {e}")
        
        # Create a DataFrame with all fitted values
        fitted_df = pd.DataFrame(all_fitted_vols)
        
        # Calculate grid for visualization
        if expiry_params:
            # Create a grid of log moneyness and time to expiry values
            k_min = options_data['log_moneyness'].min() - 0.1
            k_max = options_data['log_moneyness'].max() + 0.1
            t_min = options_data['time_to_expiry'].min()
            t_max = options_data['time_to_expiry'].max() + 0.1
            
            k_grid = np.linspace(k_min, k_max, 50)
            t_grid = np.linspace(t_min, t_max, 50)
            
            K, T = np.meshgrid(k_grid, t_grid)
            Z = np.zeros_like(K)
            
            # Interpolate SABR parameters across time dimension
            times = np.array([params['time_to_expiry'] for params in expiry_params.values()])
            forwards = np.array([params['forward'] for params in expiry_params.values()])
            alpha_values = np.array([params['alpha'] for params in expiry_params.values()])
            beta_values = np.array([params['beta'] for params in expiry_params.values()])
            rho_values = np.array([params['rho'] for params in expiry_params.values()])
            nu_values = np.array([params['nu'] for params in expiry_params.values()])
            
            # For each point in the grid, interpolate parameters and calculate volatility
            for i in range(len(t_grid)):
                t = t_grid[i]
                
                # Interpolate parameters for this time to expiry
                if len(times) > 1:
                    forward = np.interp(t, times, forwards)
                    alpha = np.interp(t, times, alpha_values)
                    beta = np.interp(t, times, beta_values)
                    rho = np.interp(t, times, rho_values)
                    nu = np.interp(t, times, nu_values)
                else:
                    # If only one expiry is available, use its parameters
                    forward = forwards[0]
                    alpha = alpha_values[0]
                    beta = beta_values[0]
                    rho = rho_values[0]
                    nu = nu_values[0]
                
                # Calculate strikes from log moneyness
                strikes = forward * np.exp(k_grid)
                
                # Calculate SABR volatility for this row of the grid
                Z[i, :] = np.array([
                    self._sabr_volatility(forward, K, t, alpha, beta, rho, nu)
                    for K in strikes
                ])
            
            # Store the results
            self.surface_data = {
                'x_grid': K,
                'y_grid': T,
                'z_grid': Z,
                'x_raw': options_data['log_moneyness'].values,
                'y_raw': options_data['time_to_expiry'].values,
                'z_raw': options_data['implied_vol'].values,
                'x_label': 'Log Moneyness (log(K/S))',
                'y_label': 'Time to Expiry (years)',
                'z_label': 'Implied Volatility',
                'fitted_values': fitted_df,
                'parameters': expiry_params
            }
            
            self.model_type = 'sabr'
            self.calibrated_params = expiry_params
            
            logger.info(f"Fitted SABR volatility surface with {len(expiry_params)} expiration dates")
            
            return self.surface_data
        else:
            logger.error("Failed to fit SABR surface: no valid parameters found")
            return None
    
    def _sabr_volatility(
        self, 
        forward: float, 
        strike: float, 
        time_to_expiry: float, 
        alpha: float, 
        beta: float, 
        rho: float, 
        nu: float
    ) -> float:
        """
        Calculate implied volatility using the SABR model.
        
        Args:
            forward: Forward price
            strike: Strike price
            time_to_expiry: Time to expiration in years
            alpha: Initial volatility parameter
            beta: CEV parameter (0 for normal, 1 for lognormal)
            rho: Correlation parameter
            nu: Volatility of volatility parameter
            
        Returns:
            SABR implied volatility
        """
        # Handle special cases
        if abs(forward - strike) < 1e-10:
            # ATM formula
            z = (nu / alpha) * (forward ** (1 - beta)) * np.sqrt(time_to_expiry)
            term1 = alpha / (forward ** (1 - beta))
            term2 = 1 + (((2 - 3 * rho**2) / 24) * (nu**2) * time_to_expiry)
            return term1 * term2
        
        # Handle extreme strikes
        if strike <= 0 or forward <= 0:
            return 0.999  # Return a high value
        
        # Calculate log(F/K)
        log_fk = np.log(forward / strike)
        
        # Calculate z
        fk_beta = (forward * strike) ** (1 - beta)
        z = (nu / alpha) * fk_beta * log_fk
        
        # Calculate chi
        chi = np.log((np.sqrt(1 - 2*rho*z + z**2) + z - rho) / (1 - rho))
        
        # Calculate the volatility
        term1 = alpha / fk_beta
        term2 = (1 + ((1 - beta)**2 / 24) * (log_fk**2) + ((1 - beta)**4 / 1920) * (log_fk**4))
        term3 = z / chi
        term4 = 1 + (((2 - 3*rho**2) / 24) * (nu**2) * time_to_expiry)
        
        return term1 * term2 * term3 * term4
    
    def fit_stochastic_local_volatility(
        self,
        options_data: pd.DataFrame,
        underlying_data: pd.DataFrame,
        initial_params: Dict = None,
        num_monte_carlo: int = 1000,
        num_time_steps: int = 100
    ) -> Dict:
        """
        Fit a stochastic-local volatility (SLV) model.
        
        Args:
            options_data: Preprocessed options data with implied volatilities
            underlying_data: Historical price data for the underlying
            initial_params: Initial parameter values for optimization
            num_monte_carlo: Number of Monte Carlo simulations
            num_time_steps: Number of time steps in simulation
            
        Returns:
            Dictionary containing the fitted model parameters and data
        """
        logger.info("Stochastic-local volatility model fitting is computationally intensive")
        logger.info("This is a simplified implementation for demonstration purposes")
        
        # Default initial parameters if not provided
        if initial_params is None:
            initial_params = {
                'kappa': 2.0,    # Mean reversion speed
                'theta': 0.04,   # Long-term variance
                'sigma': 0.4,    # Volatility of volatility
                'rho': -0.7,     # Correlation
                'v0': 0.04       # Initial variance
            }
        
        # Extract historical volatility from underlying data
        if 'realized_vol' in underlying_data.columns:
            hist_vol = underlying_data['realized_vol'].mean()
            hist_var = hist_vol ** 2
        else:
            # Calculate historical volatility if not available
            returns = np.log(underlying_data['Close'] / underlying_data['Close'].shift(1)).dropna()
            hist_vol = returns.std() * np.sqrt(252)  # Annualized
            hist_var = hist_vol ** 2
        
        # Update initial theta based on historical variance
        initial_params['theta'] = hist_var
        initial_params['v0'] = hist_var
        
        # Initial parameters for optimization
        params_0 = [
            initial_params['kappa'],
            initial_params['theta'],
            initial_params['sigma'],
            initial_params['rho']
        ]
        
        # Define the objective function (calibration to market prices)
        def objective(params):
            kappa, theta, sigma, rho = params
            v0 = initial_params['v0']  # Use fixed initial variance
            
            # Ensure parameters are within bounds
            if kappa <= 0 or theta <= 0 or sigma <= 0 or abs(rho) >= 1:
                return 1e10
            
            # Calculate model prices for all options
            model_vols = []
            for _, option in options_data.iterrows():
                strike = option['strike']
                time_to_expiry = option['time_to_expiry']
                
                # Calculate model implied volatility
                try:
                    model_vol = self._heston_volatility(
                        strike, time_to_expiry, kappa, theta, sigma, rho, v0
                    )
                    model_vols.append(model_vol)
                except:
                    # If calculation fails, return a high error
                    return 1e10
            
            # Calculate error between model and market implied volatilities
            # src/models/volatility_surface.py (continued)

            # Calculate error between model and market implied volatilities
            market_vols = options_data['implied_vol'].values
            error = np.sum((np.array(model_vols) - market_vols) ** 2)
            
            return error
        
        # Bounds for parameters
        bounds = [
            (0.1, 10.0),     # kappa: mean reversion speed
            (0.001, 0.25),   # theta: long-term variance
            (0.01, 2.0),     # sigma: volatility of volatility
            (-0.999, 0.999)  # rho: correlation
        ]
        
        try:
            # Optimize to find Heston parameters
            result = optimize.minimize(
                objective, 
                params_0, 
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 100}  # Limit iterations for demonstration
            )
            
            if result.success:
                # Extract optimized parameters
                kappa, theta, sigma, rho = result.x
                v0 = initial_params['v0']
                
                # Store parameters
                self.calibrated_params = {
                    'kappa': kappa,
                    'theta': theta,
                    'sigma': sigma,
                    'rho': rho,
                    'v0': v0
                }
                
                logger.info(f"Fitted SLV parameters: kappa={kappa:.4f}, theta={theta:.4f}, sigma={sigma:.4f}, rho={rho:.4f}, v0={v0:.4f}")
                
                # Calculate local volatility surface
                # This is a simplified implementation - in production, you would:
                # 1. Calculate Dupire local volatility surface
                # 2. Compute the leverage function
                # 3. Combine with the stochastic volatility model
                
                # Create a grid for visualization
                k_min = options_data['log_moneyness'].min() - 0.1
                k_max = options_data['log_moneyness'].max() + 0.1
                t_min = options_data['time_to_expiry'].min()
                t_max = options_data['time_to_expiry'].max() + 0.1
                
                k_grid = np.linspace(k_min, k_max, 50)
                t_grid = np.linspace(t_min, t_max, 50)
                
                K, T = np.meshgrid(k_grid, t_grid)
                Z = np.zeros_like(K)
                
                # Calculate Heston volatilities on the grid
                for i in range(K.shape[0]):
                    for j in range(K.shape[1]):
                        log_moneyness = K[i, j]
                        time_to_expiry = T[i, j]
                        
                        # Convert log moneyness to strike
                        spot = underlying_data['Close'].iloc[-1]
                        strike = spot * np.exp(log_moneyness)
                        
                        # Calculate Heston volatility
                        try:
                            Z[i, j] = self._heston_volatility(
                                strike, time_to_expiry, kappa, theta, sigma, rho, v0
                            )
                        except:
                            # Use a reasonable value if calculation fails
                            Z[i, j] = np.sqrt(theta)
                
                # Store the results
                self.surface_data = {
                    'x_grid': K,
                    'y_grid': T,
                    'z_grid': Z,
                    'x_raw': options_data['log_moneyness'].values,
                    'y_raw': options_data['time_to_expiry'].values,
                    'z_raw': options_data['implied_vol'].values,
                    'x_label': 'Log Moneyness (log(K/S))',
                    'y_label': 'Time to Expiry (years)',
                    'z_label': 'Implied Volatility',
                    'parameters': self.calibrated_params
                }
                
                self.model_type = 'stochastic_local_volatility'
                
                logger.info("Fitted stochastic-local volatility surface")
                
                return self.surface_data
            else:
                logger.error(f"SLV optimization failed: {result.message}")
                return None
        except Exception as e:
            logger.error(f"Error fitting SLV model: {e}")
            return None
    
    def _heston_volatility(
        self, 
        strike: float, 
        time_to_expiry: float, 
        kappa: float, 
        theta: float, 
        sigma: float, 
        rho: float, 
        v0: float
    ) -> float:
        """
        Calculate implied volatility using the Heston model.
        
        This is a simplified implementation using a moment-matching approximation.
        For production use, you would implement the full Heston characteristic function.
        
        Args:
            strike: Strike price
            time_to_expiry: Time to expiration in years
            kappa: Mean reversion speed
            theta: Long-term variance
            sigma: Volatility of volatility
            rho: Correlation
            v0: Initial variance
            
        Returns:
            Heston implied volatility approximation
        """
        # Simplified Heston approximation based on moment matching
        # This is not accurate for all parameter combinations but serves as an example
        
        # Calculate integrated variance
        if abs(kappa) < 1e-6:
            integrated_var = v0 * time_to_expiry
        else:
            integrated_var = theta * time_to_expiry + (v0 - theta) * (1 - np.exp(-kappa * time_to_expiry)) / kappa
        
        # Calculate skew adjustment
        skew_factor = rho * sigma * (v0 - theta) * (1 - np.exp(-kappa * time_to_expiry)) / (kappa**2)
        skew_factor += rho * sigma * theta * time_to_expiry / kappa
        
        # Calculate log moneyness
        spot = 100.0  # Arbitrary reference value
        log_moneyness = np.log(strike / spot)
        
        # Apply skew adjustment
        adjusted_var = integrated_var + skew_factor * log_moneyness
        
        # Ensure positive variance
        if adjusted_var <= 0:
            adjusted_var = 0.0001
        
        # Convert to volatility
        return np.sqrt(adjusted_var / time_to_expiry)
    
    def visualize_surface(
        self,
        title: str = None,
        save_path: str = None,
        show_market_points: bool = True,
        azimuth: float = -60,
        elevation: float = 30
    ) -> plt.Figure:
        """
        Visualize the volatility surface.
        
        Args:
            title: Plot title
            save_path: Path to save the figure
            show_market_points: Whether to show market data points
            azimuth: Horizontal viewing angle
            elevation: Vertical viewing angle
            
        Returns:
            Matplotlib figure object
        """
        if self.surface_data is None:
            logger.error("No surface data available. Fit a model first.")
            return None
        
        # Extract data
        X = self.surface_data['x_grid']
        Y = self.surface_data['y_grid']
        Z = self.surface_data['z_grid']
        
        x_label = self.surface_data['x_label']
        y_label = self.surface_data['y_label']
        z_label = self.surface_data['z_label']
        
        # Create figure
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot the surface
        surf = ax.plot_surface(
            X, Y, Z, 
            cmap=cm.viridis,
            linewidth=0, 
            antialiased=True,
            alpha=0.8
        )
        
        # Add market data points if requested
        if show_market_points and 'x_raw' in self.surface_data:
            x_raw = self.surface_data['x_raw']
            y_raw = self.surface_data['y_raw']
            z_raw = self.surface_data['z_raw']
            
            ax.scatter(
                x_raw, y_raw, z_raw, 
                color='red', 
                s=20, 
                label='Market Data'
            )
        
        # Set labels
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_zlabel(z_label)
        
        # Set title
        if title:
            ax.set_title(title)
        else:
            model_name = self.model_type.replace('_', ' ').title()
            ax.set_title(f"{model_name} Volatility Surface")
        
        # Add colorbar
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
        
        # Set view angle
        ax.view_init(elevation, azimuth)
        
        # Save figure if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved figure to {save_path}")
        
        return fig
    
    def visualize_slices(
        self,
        slice_type: str = 'time',
        values: List[float] = None,
        title: str = None,
        save_path: str = None
    ) -> plt.Figure:
        """
        Visualize slices of the volatility surface.
        
        Args:
            slice_type: 'time' for time slices or 'moneyness' for moneyness slices
            values: List of values to slice at
            title: Plot title
            save_path: Path to save the figure
            
        Returns:
            Matplotlib figure object
        """
        if self.surface_data is None:
            logger.error("No surface data available. Fit a model first.")
            return None
        
        # Extract data
        X = self.surface_data['x_grid']
        Y = self.surface_data['y_grid']
        Z = self.surface_data['z_grid']
        
        x_label = self.surface_data['x_label']
        y_label = self.surface_data['y_label']
        z_label = self.surface_data['z_label']
        
        # Default values if not provided
        if values is None:
            if slice_type == 'time':
                # Default time slices
                y_unique = np.unique(Y[:, 0])
                num_slices = min(5, len(y_unique))
                indices = np.linspace(0, len(y_unique) - 1, num_slices, dtype=int)
                values = [y_unique[i] for i in indices]
            else:
                # Default moneyness slices
                x_unique = np.unique(X[0, :])
                num_slices = min(5, len(x_unique))
                indices = np.linspace(0, len(x_unique) - 1, num_slices, dtype=int)
                values = [x_unique[i] for i in indices]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot slices
        if slice_type == 'time':
            # Plot volatility smile at different times
            for time_value in values:
                # Find closest row in the grid
                row_idx = np.abs(Y[:, 0] - time_value).argmin()
                
                # Extract data for this slice
                x_slice = X[row_idx, :]
                z_slice = Z[row_idx, :]
                
                # Sort by x value
                sort_idx = np.argsort(x_slice)
                x_slice = x_slice[sort_idx]
                z_slice = z_slice[sort_idx]
                
                # Plot this slice
                ax.plot(x_slice, z_slice, '-', linewidth=2, label=f'T = {time_value:.2f}')
            
            ax.set_xlabel(x_label)
            ax.set_ylabel(z_label)
            ax.set_title(title or 'Volatility Smile at Different Times')
            
        else:  # moneyness slices
            # Plot volatility term structure at different moneyness levels
            for moneyness_value in values:
                # Find closest column in the grid
                col_idx = np.abs(X[0, :] - moneyness_value).argmin()
                
                # Extract data for this slice
                y_slice = Y[:, col_idx]
                z_slice = Z[:, col_idx]
                
                # Sort by y value
                sort_idx = np.argsort(y_slice)
                y_slice = y_slice[sort_idx]
                z_slice = z_slice[sort_idx]
                
                # Plot this slice
                label = 'ATM' if abs(moneyness_value) < 0.01 else f'K/S = {np.exp(moneyness_value):.2f}'
                ax.plot(y_slice, z_slice, '-', linewidth=2, label=label)
            
            ax.set_xlabel(y_label)
            ax.set_ylabel(z_label)
            ax.set_title(title or 'Volatility Term Structure at Different Moneyness Levels')
        
        # Add legend and grid
        ax.legend()
        ax.grid(True)
        
        # Save figure if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved figure to {save_path}")
        
        return fig

