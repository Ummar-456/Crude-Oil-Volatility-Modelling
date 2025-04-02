# src/models/local_volatility.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.optimize import minimize
from scipy.interpolate import RectBivariateSpline, griddata
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
import logging
from typing import Dict, List, Optional, Tuple, Union, Callable
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LocalVolatilityModel:
    """
    Implementation of a local volatility model for commodity options.
    
    This class implements the Dupire local volatility model, which infers
    the local volatility surface from market implied volatilities. It includes
    calibration methods, pricing functions, and visualization tools.
    """
    
    def __init__(
        self, 
        a0: float = 0.2,
        a_min: float = 0.01,
        a_max: float = 1.0,
        regularization_param: float = 0.1
    ):
        """
        Initialize the LocalVolatilityModel.
        
        Args:
            a0: Initial volatility value for calibration
            a_min: Lower bound for volatility
            a_max: Upper bound for volatility
            regularization_param: Tikhonov regularization parameter
        """
        self.a0 = a0
        self.a_min = a_min
        self.a_max = a_max
        self.regularization_param = regularization_param
        
        # Calibrated parameters
        self.local_vol_surface = None
        self.strike_grid = None
        self.time_grid = None
        self.calibrated = False
        
        # Finite difference parameters
        self.dt = 0.01  # Time step for finite difference
        self.ds = 0.01  # Price step for finite difference
        
        # Interpolation function
        self.interpolator = None
    
    def calibrate(
        self,
        option_data: pd.DataFrame,
        underlying_price: float,
        risk_free_rate: float = 0.05,
        dividend_yield: float = 0.0,
        method: str = 'tikhonov',
        grid_points: Tuple[int, int] = (50, 50)
    ) -> Dict:
        """
        Calibrate the local volatility model to market option prices.
        
        Args:
            option_data: DataFrame containing option data (strikes, expiries, implied vols)
            underlying_price: Current price of the underlying asset
            risk_free_rate: Risk-free interest rate (annualized)
            dividend_yield: Dividend yield for the underlying (annualized)
            method: Calibration method ('tikhonov' or 'direct')
            grid_points: Number of grid points in (strike, time) dimensions
            
        Returns:
            Dictionary containing calibration results
        """
        logger.info("Calibrating local volatility model...")
        
        # Extract data
        strikes = option_data['strike'].values
        expiries = option_data['time_to_expiry'].values
        implied_vols = option_data['implied_vol'].values
        
        # Create grid for local volatility surface
        
        min_strike = np.maximum(0.5 * underlying_price, strikes.min() * 0.9)
        max_strike = max(strikes) * 1.1
        min_time = max(0.01, min(expiries) * 0.9)
        max_time = max(expiries) * 1.1
        
        self.strike_grid = np.linspace(min_strike, max_strike, grid_points[0])
        self.time_grid = np.linspace(min_time, max_time, grid_points[1])
        
        # Create meshgrid for surface
        K, T = np.meshgrid(self.strike_grid, self.time_grid)
        
        if method == 'direct':
            # Direct calibration using Dupire formula
            local_vol = self._calibrate_direct(
                strikes, expiries, implied_vols, underlying_price,
                risk_free_rate, dividend_yield, K, T
            )
        else:
            # Tikhonov regularization method
            local_vol = self._calibrate_tikhonov(
                strikes, expiries, implied_vols, underlying_price,
                risk_free_rate, dividend_yield, K, T
            )
        
        # Store the calibrated surface
        self.local_vol_surface = local_vol
        self.calibrated = True
        
        # Create interpolation function
        self.interpolator = RectBivariateSpline(
            self.time_grid, self.strike_grid, local_vol
        )
        
        logger.info("Local volatility model calibration completed")
        
        # Return calibration results
        return {
            'strike_grid': self.strike_grid,
            'time_grid': self.time_grid,
            'local_vol_surface': local_vol
        }
    
    def _calibrate_direct(
        self,
        strikes: np.ndarray,
        expiries: np.ndarray,
        implied_vols: np.ndarray,
        underlying_price: float,
        risk_free_rate: float,
        dividend_yield: float,
        K: np.ndarray,
        T: np.ndarray
    ) -> np.ndarray:
        """
        Calibrate using direct application of Dupire formula.
        
        Args:
            strikes: Array of strike prices
            expiries: Array of times to expiration
            implied_vols: Array of implied volatilities
            underlying_price: Current price of the underlying asset
            risk_free_rate: Risk-free interest rate (annualized)
            dividend_yield: Dividend yield for the underlying (annualized)
            K: Meshgrid of strike prices
            T: Meshgrid of times to expiration
            
        Returns:
            2D array of local volatilities
        """
        # First, interpolate implied volatilities onto the grid
        points = np.column_stack((expiries, strikes))
        implied_vol_grid = griddata(
            points, implied_vols, (T, K), method='cubic', fill_value=self.a0
        )
        
        # Apply smoothing to the implied volatility surface
        implied_vol_grid = self._smooth_surface(implied_vol_grid)
        
        # Initialize local volatility surface
        local_vol = np.zeros_like(implied_vol_grid)
        
        # Apply Dupire formula to calculate local volatilities
        for i in range(1, len(self.time_grid) - 1):
            for j in range(1, len(self.strike_grid) - 1):
                t = self.time_grid[i]
                k = self.strike_grid[j]
                
                # Get implied volatility and its derivatives
                sigma = implied_vol_grid[i, j]
                
                # Calculate derivatives using finite differences
                d_sigma_dt = (implied_vol_grid[i+1, j] - implied_vol_grid[i-1, j]) / (self.time_grid[i+1] - self.time_grid[i-1])
                d_sigma_dk = (implied_vol_grid[i, j+1] - implied_vol_grid[i, j-1]) / (self.strike_grid[j+1] - self.strike_grid[j-1])
                d2_sigma_dk2 = (implied_vol_grid[i, j+1] - 2*implied_vol_grid[i, j] + implied_vol_grid[i, j-1]) / ((self.strike_grid[j+1] - self.strike_grid[j-1])/2)**2
                
                # Apply Dupire formula
                try:
                    local_vol[i, j] = self._dupire_formula(
                        sigma, t, k, underlying_price, risk_free_rate, dividend_yield,
                        d_sigma_dt, d_sigma_dk, d2_sigma_dk2
                    )
                except:
                    # In case of numerical issues, use implied volatility
                    local_vol[i, j] = sigma
                
                # Ensure volatility is within bounds
                local_vol[i, j] = max(min(local_vol[i, j], self.a_max), self.a_min)
        
        # Fill boundary values
        local_vol[0, :] = local_vol[1, :]
        local_vol[-1, :] = local_vol[-2, :]
        local_vol[:, 0] = local_vol[:, 1]
        local_vol[:, -1] = local_vol[:, -2]
        
        return local_vol
    
    def _calibrate_tikhonov(
        self,
        strikes: np.ndarray,
        expiries: np.ndarray,
        implied_vols: np.ndarray,
        underlying_price: float,
        risk_free_rate: float,
        dividend_yield: float,
        K: np.ndarray,
        T: np.ndarray
    ) -> np.ndarray:
        """
        Calibrate using Tikhonov regularization.
        
        Args:
            strikes: Array of strike prices
            expiries: Array of times to expiration
            implied_vols: Array of implied volatilities
            underlying_price: Current price of the underlying asset
            risk_free_rate: Risk-free interest rate (annualized)
            dividend_yield: Dividend yield for the underlying (annualized)
            K: Meshgrid of strike prices
            T: Meshgrid of times to expiration
            
        Returns:
            2D array of local volatilities
        """
        # Create initial guess for local volatility (constant surface)
        initial_guess = np.full(K.shape, self.a0).flatten()
        
        # Create market data points
        market_points = []
        for i in range(len(strikes)):
            market_points.append({
                'strike': strikes[i],
                'expiry': expiries[i],
                'implied_vol': implied_vols[i]
            })
        
        # Define objective function for optimization
        def objective(a):
            # Reshape flat array to 2D surface
            a_surface = a.reshape(K.shape)
            
            # Calculate model prices
            model_vols = []
            for point in market_points:
                k = point['strike']
                t = point['expiry']
                
                # Find closest grid points
                k_idx = np.abs(self.strike_grid - k).argmin()
                t_idx = np.abs(self.time_grid - t).argmin()
                
                # Use local volatility to calculate implied volatility
                # This is a simplification - in practice, you'd solve the forward equation
                model_vol = a_surface[t_idx, k_idx]
                model_vols.append(model_vol)
            
            # Calculate data fitting term
            data_term = np.sum((np.array(model_vols) - implied_vols)**2)
            
            # Calculate regularization term (smoothness penalty)
            reg_term = self._tikhonov_regularization(a_surface)
            
            # Total objective
            total = data_term + self.regularization_param * reg_term
            
            return total
        
        # Define bounds for optimization
        bounds = [(self.a_min, self.a_max) for _ in range(len(initial_guess))]
        
        # Perform optimization
        logger.info("Optimizing local volatility surface...")
        result = minimize(
            objective,
            initial_guess,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 100}
        )
        
        if result.success:
            logger.info(f"Optimization successful: {result.message}")
        else:
            logger.warning(f"Optimization may not have converged: {result.message}")
        
        # Reshape result to 2D surface
        local_vol = result.x.reshape(K.shape)
        
        # Apply smoothing
        local_vol = self._smooth_surface(local_vol)
        
        return local_vol
    
    def _dupire_formula(
        self,
        sigma: float,
        t: float,
        k: float,
        s: float,
        r: float,
        q: float,
        d_sigma_dt: float,
        d_sigma_dk: float,
        d2_sigma_dk2: float
    ) -> float:
        """
        Calculate local volatility using Dupire formula.
        
        Args:
            sigma: Implied volatility
            t: Time to expiration
            k: Strike price
            s: Underlying price
            r: Risk-free rate
            q: Dividend yield
            d_sigma_dt: Derivative of implied vol with respect to time
            d_sigma_dk: Derivative of implied vol with respect to strike
            d2_sigma_dk2: Second derivative of implied vol with respect to strike
            
        Returns:
            Local volatility value
        """
        # Calculate d1 and d2 terms
        d1 = (np.log(s/k) + (r - q + 0.5 * sigma**2) * t) / (sigma * np.sqrt(t))
        
        # Calculate terms in Dupire formula
        term1 = 2 * (d_sigma_dt + (r - q) * k * d_sigma_dk)
        term2 = 2 * r * sigma
        term3 = k**2 * sigma * (d2_sigma_dk2 + d_sigma_dk**2)
        term4 = 2 * k * d_sigma_dk * (r - q - 0.5 * sigma**2)
        
        # Calculate denominator
        denom = (1 + k * d1 * np.sqrt(t) * d_sigma_dk)**2 + k**2 * t * sigma * (d2_sigma_dk2 + d1 * np.sqrt(t) * d_sigma_dk**2)
        
        # Calculate local volatility
        if denom <= 0:
            return sigma  # Fallback to implied volatility
        
        local_vol = sigma * np.sqrt((term1 + term2) / denom)
        
        return local_vol
    
    def _tikhonov_regularization(self, a: np.ndarray) -> float:
        """
        Calculate Tikhonov regularization term (smoothness penalty).
        
        Args:
            a: 2D array of local volatilities
            
        Returns:
            Regularization term value
        """
        # Calculate first derivatives
        dx = np.diff(a, axis=1)
        dy = np.diff(a, axis=0)
        
        # Calculate second derivatives
        dxx = np.diff(a, n=2, axis=1)
        dyy = np.diff(a, n=2, axis=0)
        
        # Sum of squared derivatives (smoothness penalty)
        reg_term = (
            np.sum(dx**2) + np.sum(dy**2) + 
            np.sum(dxx**2) + np.sum(dyy**2)
        )
        
        return reg_term
    
    def _smooth_surface(self, surface: np.ndarray, window_size: int = 3) -> np.ndarray:
        """
        Apply smoothing to a volatility surface.
        
        Args:
            surface: 2D array to smooth
            window_size: Size of the smoothing window
            
        Returns:
            Smoothed 2D array
        """
        # Create a copy to avoid modifying the original
        smoothed = surface.copy()
        
        # Apply simple moving average smoothing
        half_window = window_size // 2
        for i in range(half_window, surface.shape[0] - half_window):
            for j in range(half_window, surface.shape[1] - half_window):
                window = surface[i-half_window:i+half_window+1, j-half_window:j+half_window+1]
                smoothed[i, j] = np.mean(window)
        
        return smoothed
    
    def get_local_volatility(self, strike: float, time: float) -> float:
        """
        Get local volatility for a specific strike and time.
        
        Args:
            strike: Strike price
            time: Time to expiration
            
        Returns:
            Local volatility value
        """
        if not self.calibrated:
            raise ValueError("Model must be calibrated before getting local volatility")
        
        # Use interpolation to get local volatility
        return float(self.interpolator(time, strike))
    
    def price_european_option(
        self,
        option_type: str,
        strike: float,
        time_to_expiry: float,
        underlying_price: float,
        risk_free_rate: float = 0.05,
        dividend_yield: float = 0.0,
        num_steps: Tuple[int, int] = (100, 100)
    ) -> float:
        """
        Price a European option using finite difference method.
        
        Args:
            option_type: 'call' or 'put'
            strike: Strike price
            time_to_expiry: Time to expiration in years
            underlying_price: Current price of the underlying asset
            risk_free_rate: Risk-free interest rate (annualized)
            dividend_yield: Dividend yield for the underlying (annualized)
            num_steps: Number of steps in (time, price) dimensions
            
        Returns:
            Option price
        """
        if not self.calibrated:
            raise ValueError("Model must be calibrated before pricing options")
        
        # Validate option type
        if option_type.lower() not in ['call', 'put']:
            raise ValueError("Option type must be 'call' or 'put'")
        
        # Set up grid
        s_max = 2 * underlying_price
        s_min = 0.01 * underlying_price
        
        # Create price and time grids
        s_values = np.linspace(s_min, s_max, num_steps[1])
        t_values = np.linspace(0, time_to_expiry, num_steps[0])
        
        ds = s_values[1] - s_values[0]
        dt = t_values[1] - t_values[0]
        
        # Initialize option value grid
        option_values = np.zeros((num_steps[0], num_steps[1]))
        
        # Set terminal condition (payoff at expiry)
        if option_type.lower() == 'call':
            option_values[-1, :] = np.maximum(s_values - strike, 0)
        else:  # put
            option_values[-1, :] = np.maximum(strike - s_values, 0)
        
        # Work backwards in time
        for t_idx in range(num_steps[0] - 2, -1, -1):
            # Current time
            t = t_values[t_idx]
            
            # Create tridiagonal system for implicit scheme
            diagonals = []
            
            # Main diagonal
            main_diag = np.ones(num_steps[1])
            
            # Upper and lower diagonals
            upper_diag = np.zeros(num_steps[1] - 1)
            lower_diag = np.zeros(num_steps[1] - 1)
            
            # Right-hand side
            rhs = np.copy(option_values[t_idx + 1, :])
            
                        # Fill diagonals
            for j in range(1, num_steps[1] - 1):
                s = s_values[j]
                sigma = self.get_local_volatility(s, t)
                
                alpha = 0.5 * dt * (sigma**2 * s**2 / ds**2 - (risk_free_rate - dividend_yield) * s / ds)
                beta = 0.5 * dt * (sigma**2 * s**2 / ds**2 + (risk_free_rate - dividend_yield) * s / ds)
                
                main_diag[j] = 1 + dt * (sigma**2 * s**2 / ds**2 + risk_free_rate)
                upper_diag[j-1] = -beta
                lower_diag[j-1] = -alpha
                
                rhs[j] += alpha * option_values[t_idx + 1, j-1] + \
                          (1 - dt * (sigma**2 * s**2 / ds**2 + risk_free_rate)) * option_values[t_idx + 1, j] + \
                          beta * option_values[t_idx + 1, j+1]
            
            # Boundary conditions
            if option_type.lower() == 'call':
                rhs[0] = s_min
                rhs[-1] = s_max - strike * np.exp(-risk_free_rate * (time_to_expiry - t))
            else:  # put
                rhs[0] = strike * np.exp(-risk_free_rate * (time_to_expiry - t)) - s_min
                rhs[-1] = 0
            
            # Construct and solve tridiagonal system
            diagonals = [lower_diag, main_diag, upper_diag]
            tridiag_matrix = diags(diagonals, [-1, 0, 1]).toarray()
            option_values[t_idx, :] = np.linalg.solve(tridiag_matrix, rhs)
        
        # Interpolate to get option price at current stock price
        return np.interp(underlying_price, s_values, option_values[0, :])
    
    def price_american_option(
        self,
        option_type: str,
        strike: float,
        time_to_expiry: float,
        underlying_price: float,
        risk_free_rate: float = 0.05,
        dividend_yield: float = 0.0,
        num_steps: Tuple[int, int] = (100, 100)
    ) -> float:
        """
        Price an American option using finite difference method.
        
        Args:
            option_type: 'call' or 'put'
            strike: Strike price
            time_to_expiry: Time to expiration in years
            underlying_price: Current price of the underlying asset
            risk_free_rate: Risk-free interest rate (annualized)
            dividend_yield: Dividend yield for the underlying (annualized)
            num_steps: Number of steps in (time, price) dimensions
            
        Returns:
            Option price
        """
        if not self.calibrated:
            raise ValueError("Model must be calibrated before pricing options")
        
        # Validate option type
        if option_type.lower() not in ['call', 'put']:
            raise ValueError("Option type must be 'call' or 'put'")
        
        # Set up grid
        s_max = 2 * underlying_price
        s_min = 0.01 * underlying_price
        
        # Create price and time grids
        s_values = np.linspace(s_min, s_max, num_steps[1])
        t_values = np.linspace(0, time_to_expiry, num_steps[0])
        
        ds = s_values[1] - s_values[0]
        dt = t_values[1] - t_values[0]
        
        # Initialize option value grid
        option_values = np.zeros((num_steps[0], num_steps[1]))
        
        # Set terminal condition (payoff at expiry)
        if option_type.lower() == 'call':
            option_values[-1, :] = np.maximum(s_values - strike, 0)
        else:  # put
            option_values[-1, :] = np.maximum(strike - s_values, 0)
        
        # Work backwards in time
        for t_idx in range(num_steps[0] - 2, -1, -1):
            # Current time
            t = t_values[t_idx]
            
            # Create tridiagonal system for implicit scheme
            diagonals = []
            
            # Main diagonal
            main_diag = np.ones(num_steps[1])
            
            # Upper and lower diagonals
            upper_diag = np.zeros(num_steps[1] - 1)
            lower_diag = np.zeros(num_steps[1] - 1)
            
            # Right-hand side
            rhs = np.copy(option_values[t_idx + 1, :])
            
            # Fill diagonals
            for j in range(1, num_steps[1] - 1):
                s = s_values[j]
                sigma = self.get_local_volatility(s, t)
                
                alpha = 0.5 * dt * (sigma**2 * s**2 / ds**2 - (risk_free_rate - dividend_yield) * s / ds)
                beta = 0.5 * dt * (sigma**2 * s**2 / ds**2 + (risk_free_rate - dividend_yield) * s / ds)
                
                main_diag[j] = 1 + dt * (sigma**2 * s**2 / ds**2 + risk_free_rate)
                upper_diag[j-1] = -beta
                lower_diag[j-1] = -alpha
                
                rhs[j] += alpha * option_values[t_idx + 1, j-1] + \
                          (1 - dt * (sigma**2 * s**2 / ds**2 + risk_free_rate)) * option_values[t_idx + 1, j] + \
                          beta * option_values[t_idx + 1, j+1]
            
            # Boundary conditions
            if option_type.lower() == 'call':
                rhs[0] = s_min
                rhs[-1] = s_max - strike * np.exp(-risk_free_rate * (time_to_expiry - t))
            else:  # put
                rhs[0] = strike * np.exp(-risk_free_rate * (time_to_expiry - t)) - s_min
                rhs[-1] = 0
            
            # Construct and solve tridiagonal system
            diagonals = [lower_diag, main_diag, upper_diag]
            tridiag_matrix = diags(diagonals, [-1, 0, 1]).toarray()
            option_values[t_idx, :] = np.linalg.solve(tridiag_matrix, rhs)
            
            # Apply early exercise condition
            if option_type.lower() == 'call':
                option_values[t_idx, :] = np.maximum(option_values[t_idx, :], s_values - strike)
            else:  # put
                option_values[t_idx, :] = np.maximum(option_values[t_idx, :], strike - s_values)
        
        # Interpolate to get option price at current stock price
        return np.interp(underlying_price, s_values, option_values[0, :])
    
    def calculate_greeks(
        self,
        option_type: str,
        strike: float,
        time_to_expiry: float,
        underlying_price: float,
        risk_free_rate: float = 0.05,
        dividend_yield: float = 0.0,
        is_american: bool = False
    ) -> Dict[str, float]:
        """
        Calculate option Greeks using finite differences.
        
        Args:
            option_type: 'call' or 'put'
            strike: Strike price
            time_to_expiry: Time to expiration in years
            underlying_price: Current price of the underlying asset
            risk_free_rate: Risk-free interest rate (annualized)
            dividend_yield: Dividend yield for the underlying (annualized)
            is_american: Whether the option is American (True) or European (False)
            
        Returns:
            Dictionary containing calculated Greeks
        """
        if not self.calibrated:
            raise ValueError("Model must be calibrated before calculating Greeks")
        
        # Small changes for finite differences
        dS = 0.01 * underlying_price
        dT = 0.01
        dV = 0.01
        
        # Price the option
        if is_american:
            price = self.price_american_option(
                option_type, strike, time_to_expiry, underlying_price,
                risk_free_rate, dividend_yield
            )
            price_up = self.price_american_option(
                option_type, strike, time_to_expiry, underlying_price + dS,
                risk_free_rate, dividend_yield
            )
            price_down = self.price_american_option(
                option_type, strike, time_to_expiry, underlying_price - dS,
                risk_free_rate, dividend_yield
            )
            price_t_down = self.price_american_option(
                option_type, strike, max(0, time_to_expiry - dT), underlying_price,
                risk_free_rate, dividend_yield
            )
        else:
            price = self.price_european_option(
                option_type, strike, time_to_expiry, underlying_price,
                risk_free_rate, dividend_yield
            )
            price_up = self.price_european_option(
                option_type, strike, time_to_expiry, underlying_price + dS,
                risk_free_rate, dividend_yield
            )
            price_down = self.price_european_option(
                option_type, strike, time_to_expiry, underlying_price - dS,
                risk_free_rate, dividend_yield
            )
            price_t_down = self.price_european_option(
                option_type, strike, max(0, time_to_expiry - dT), underlying_price,
                risk_free_rate, dividend_yield
            )
        
        # Calculate Greeks
        delta = (price_up - price_down) / (2 * dS)
        gamma = (price_up - 2 * price + price_down) / (dS ** 2)
        theta = -(price_t_down - price) / dT
        
        # Calculate vega using local volatility
        local_vol = self.get_local_volatility(underlying_price, time_to_expiry)
        price_v_up = self.price_european_option(
            option_type, strike, time_to_expiry, underlying_price,
            risk_free_rate, dividend_yield
        )
        vega = (price_v_up - price) / dV
        
        # Calculate rho (sensitivity to interest rate)
        price_r_up = self.price_european_option(
            option_type, strike, time_to_expiry, underlying_price,
            risk_free_rate + 0.01, dividend_yield
        )
        rho = (price_r_up - price) / 0.01
        
        return {
            'delta': delta,
            'gamma': gamma,
            'theta': theta,
            'vega': vega,
            'rho': rho
        }
    
    def plot_local_volatility_surface(
        self,
        title: str = "Local Volatility Surface",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot the calibrated local volatility surface.
        
        Args:
            title: Plot title
            save_path: If provided, save the figure to this path
            
        Returns:
            Matplotlib figure object
        """
        if not self.calibrated:
            raise ValueError("Model must be calibrated before plotting")
        
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        X, Y = np.meshgrid(self.strike_grid, self.time_grid)
        
        surf = ax.plot_surface(
            X, Y, self.local_vol_surface, 
            cmap=cm.coolwarm,
            linewidth=0, 
            antialiased=False
        )
        
        ax.set_xlabel('Strike')
        ax.set_ylabel('Time to Expiry')
        ax.set_zlabel('Local Volatility')
        ax.set_title(title)
        
        fig.colorbar(surf, shrink=0.5, aspect=5)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved local volatility surface plot to {save_path}")
        
        return fig

