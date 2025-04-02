# src/models/stochastic_local_volatility_hybrid.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.optimize import minimize
from scipy.interpolate import RectBivariateSpline, griddata
import logging
from typing import Dict, List, Optional, Tuple, Union, Callable
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class StochasticLocalVolatilityHybrid:
    """
    A hybrid stochastic-local volatility model for commodity derivatives.
    
    This model combines the benefits of both local volatility (exact calibration to market smiles)
    and stochastic volatility (realistic dynamics) approaches. It uses a parsimonious 
    parametrization to handle the limited number of options quoted in commodity markets.
    """
    
    def __init__(
        self, 
        kappa: float = 2.0,
        theta: float = 0.04,
        sigma: float = 0.4,
        rho: float = -0.7,
        v0: float = 0.04,
        regularization_param: float = 0.1
    ):
        """
        Initialize the stochastic-local volatility hybrid model.
        
        Args:
            kappa: Mean reversion speed of volatility
            theta: Long-term mean of volatility
            sigma: Volatility of volatility
            rho: Correlation between asset and volatility
            v0: Initial variance
            regularization_param: Parameter for Tikhonov regularization
        """
        # Stochastic volatility parameters
        self.kappa = kappa
        self.theta = theta
        self.sigma = sigma
        self.rho = rho
        self.v0 = v0
        
        # Regularization parameter
        self.regularization_param = regularization_param
        
        # Calibrated parameters
        self.leverage_function = None
        self.strike_grid = None
        self.time_grid = None
        self.calibrated = False
        
        # Interpolation function
        self.interpolator = None
    
    def calibrate(
        self,
        option_data: pd.DataFrame,
        underlying_price: float,
        risk_free_rate: float = 0.05,
        dividend_yield: float = 0.0,
        grid_points: Tuple[int, int] = (50, 50)
    ) -> Dict:
        """
        Calibrate the hybrid model to market option prices.
        
        Args:
            option_data: DataFrame containing option data (strikes, expiries, implied vols)
            underlying_price: Current price of the underlying asset
            risk_free_rate: Risk-free interest rate (annualized)
            dividend_yield: Dividend yield for the underlying (annualized)
            grid_points: Number of grid points in (strike, time) dimensions
            
        Returns:
            Dictionary containing calibration results
        """
        logger.info("Calibrating stochastic-local volatility hybrid model...")
        
        # Extract data
        strikes = option_data['strike'].values
        expiries = option_data['time_to_expiry'].values
        implied_vols = option_data['implied_vol'].values
        
        # Create grid for leverage function
        min_strike = max(0.5 * underlying_price, min(strikes) * 0.9)
        max_strike = max(strikes) * 1.1
        min_time = max(0.01, min(expiries) * 0.9)
        max_time = max(expiries) * 1.1
        
        self.strike_grid = np.linspace(min_strike, max_strike, grid_points[0])
        self.time_grid = np.linspace(min_time, max_time, grid_points[1])
        
        # Create meshgrid for surface
        K, T = np.meshgrid(self.strike_grid, self.time_grid)
        
        # First, calibrate the stochastic volatility parameters
        self._calibrate_stochastic_vol_params(option_data, underlying_price, risk_free_rate, dividend_yield)
        
        # Then, calibrate the leverage function
        leverage_function = self._calibrate_leverage_function(
            strikes, expiries, implied_vols, underlying_price,
            risk_free_rate, dividend_yield, K, T
        )
        
        # Store the calibrated leverage function
        self.leverage_function = leverage_function
        self.calibrated = True
        
        # Create interpolation function
        self.interpolator = RectBivariateSpline(
            self.time_grid, self.strike_grid, leverage_function
        )
        
        logger.info("Hybrid model calibration completed")
        
        # Return calibration results
        return {
            'strike_grid': self.strike_grid,
            'time_grid': self.time_grid,
            'leverage_function': leverage_function,
            'stochastic_vol_params': {
                'kappa': self.kappa,
                'theta': self.theta,
                'sigma': self.sigma,
                'rho': self.rho,
                'v0': self.v0
            }
        }
    
    def _calibrate_stochastic_vol_params(
        self,
        option_data: pd.DataFrame,
        underlying_price: float,
        risk_free_rate: float,
        dividend_yield: float
    ) -> None:
        """
        Calibrate the stochastic volatility parameters.
        
        Args:
            option_data: DataFrame containing option data
            underlying_price: Current price of the underlying asset
            risk_free_rate: Risk-free interest rate (annualized)
            dividend_yield: Dividend yield for the underlying (annualized)
        """
        logger.info("Calibrating stochastic volatility parameters...")
        
        # Filter for at-the-money options for simpler calibration
        atm_options = option_data[
            (option_data['strike'] >= 0.95 * underlying_price) &
            (option_data['strike'] <= 1.05 * underlying_price)
        ]
        
        if len(atm_options) < 5:
            logger.warning("Not enough ATM options for calibration, using all options")
            atm_options = option_data
        
        # Initial parameters
        initial_params = [self.kappa, self.theta, self.sigma, self.rho]
        
        # Define objective function for optimization
        def objective(params):
            kappa, theta, sigma, rho = params
            
            # Ensure parameters are within bounds
            if kappa <= 0 or theta <= 0 or sigma <= 0 or abs(rho) >= 1:
                return 1e10
            
            # Calculate model implied volatilities
            model_vols = []
            for _, option in atm_options.iterrows():
                strike = option['strike']
                time_to_expiry = option['time_to_expiry']
                
                # Calculate Heston implied volatility
                model_vol = self._heston_implied_vol(
                    strike, time_to_expiry, underlying_price,
                    kappa, theta, sigma, rho, self.v0,
                    risk_free_rate, dividend_yield
                )
                model_vols.append(model_vol)
            
            # Calculate error between model and market implied volatilities
            market_vols = atm_options['implied_vol'].values
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
            result = minimize(
                objective, 
                initial_params, 
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 100}
            )
            
            if result.success:
                # Update parameters
                self.kappa, self.theta, self.sigma, self.rho = result.x
                logger.info(f"Calibrated stochastic volatility parameters: kappa={self.kappa:.4f}, theta={self.theta:.4f}, sigma={self.sigma:.4f}, rho={self.rho:.4f}")
            else:
                logger.warning(f"Stochastic volatility calibration may not have converged: {result.message}")
        except Exception as e:
            logger.error(f"Error in stochastic volatility calibration: {e}")
    
    def _calibrate_leverage_function(
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
        Calibrate the leverage function using Tikhonov regularization.
        
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
            2D array of leverage function values
        """
        logger.info("Calibrating leverage function...")
        
        # First, calculate local volatility surface using Dupire formula
        local_vol_surface = self._calculate_local_volatility(
            strikes, expiries, implied_vols, underlying_price,
            risk_free_rate, dividend_yield, K, T
        )
        
        # Calculate Heston volatility surface
        heston_vol_surface = np.zeros_like(K)
        for i in range(len(self.time_grid)):
            for j in range(len(self.strike_grid)):
                t = self.time_grid[i]
                k = self.strike_grid[j]
                
                heston_vol_surface[i, j] = self._heston_implied_vol(
                    k, t, underlying_price,
                    self.kappa, self.theta, self.sigma, self.rho, self.v0,
                    risk_free_rate, dividend_yield
                )
        
        # Calculate leverage function as ratio of local vol to Heston vol
        leverage_function = np.zeros_like(K)
        for i in range(len(self.time_grid)):
            for j in range(len(self.strike_grid)):
                if heston_vol_surface[i, j] > 0:
                    leverage_function[i, j] = local_vol_surface[i, j] / heston_vol_surface[i, j]
                else:
                    leverage_function[i, j] = 1.0
        
        # Apply Tikhonov regularization to smooth the leverage function
        leverage_function = self._apply_tikhonov_regularization(leverage_function)
        
        return leverage_function
    
    def _calculate_local_volatility(
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
        Calculate local volatility surface using Dupire formula.
        
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
        # Interpolate implied volatilities onto the grid
        points = np.column_stack((expiries, strikes))
        implied_vol_grid = griddata(
            points, implied_vols, (T, K), method='cubic', fill_value=np.mean(implied_vols)
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
                local_vol[i, j] = max(min(local_vol[i, j], 1.0), 0.01)
        
        # Fill boundary values
        local_vol[0, :] = local_vol[1, :]
        local_vol[-1, :] = local_vol[-2, :]
        local_vol[:, 0] = local_vol[:, 1]
        local_vol[:, -1] = local_vol[:, -2]
        
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
        # Calculate d1 term
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
    
    def _heston_implied_vol(
        self,
        strike: float,
        time_to_expiry: float,
        spot: float,
        kappa: float,
        theta: float,
        sigma: float,
        rho: float,
        v0: float,
        r: float,
        q: float
    ) -> float:
        """
        Calculate implied volatility using the Heston model.
        
        This is a simplified implementation using moment matching approximation.
        
        Args:
            strike: Strike price
            time_to_expiry: Time to expiration in years
            spot: Current price of the underlying asset
            kappa: Mean reversion speed
            theta: Long-term variance
            sigma: Volatility of volatility
            rho: Correlation between asset and volatility
            v0: Initial variance
            r: Risk-free rate
            q: Dividend yield
            
        Returns:
            Heston implied volatility
        """
        # Calculate log moneyness
        log_moneyness = np.log(strike / spot)
        
        # Calculate integrated variance
        if kappa < 1e-6:
            integrated_var = v0 * time_to_expiry
        else:
            integrated_var = theta * time_to_expiry + (v0 - theta) * (1 - np.exp(-kappa * time_to_expiry)) / kappa
        
        # Calculate skew adjustment
        skew_factor = rho * sigma * (v0 - theta) * (1 - np.exp(-kappa * time_to_expiry)) / (kappa**2)
        skew_factor += rho * sigma * theta * time_to_expiry / kappa
        
        # Apply skew adjustment
        adjusted_var = integrated_var + skew_factor * log_moneyness
        
        # Ensure positive variance
        if adjusted_var <= 0:
            adjusted_var = 0.0001
        
        # Convert to volatility
        return np.sqrt(adjusted_var / time_to_expiry)
    
    def _smooth_surface(self, surface: np.ndarray, window_size: int = 3) -> np.ndarray:
            """
        Apply smoothing to a surface.
        
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
    
    def _apply_tikhonov_regularization(self, surface: np.ndarray) -> np.ndarray:
        """
        Apply Tikhonov regularization to smooth a surface.
        
        Args:
            surface: 2D array to regularize
            
        Returns:
            Regularized 2D array
        """
        # Create a copy to avoid modifying the original
        regularized = surface.copy()
        
        # Apply iterative regularization
        alpha = self.regularization_param
        max_iter = 100
        
        for _ in range(max_iter):
            # Calculate Laplacian (approximation of second derivatives)
            laplacian = np.zeros_like(regularized)
            
            # Interior points
            for i in range(1, regularized.shape[0] - 1):
                for j in range(1, regularized.shape[1] - 1):
                    laplacian[i, j] = (
                        regularized[i+1, j] + regularized[i-1, j] +
                        regularized[i, j+1] + regularized[i, j-1] - 4 * regularized[i, j]
                    )
            
            # Update using regularization
            regularized = regularized + alpha * laplacian
        
        # Ensure values are positive
        regularized = np.maximum(regularized, 0.01)
        
        return regularized
    
    def get_leverage(self, strike: float, time: float) -> float:
        """
        Get leverage function value for a specific strike and time.
        
        Args:
            strike: Strike price
            time: Time to expiration
            
        Returns:
            Leverage function value
        """
        if not self.calibrated:
            raise ValueError("Model must be calibrated before getting leverage function")
        
        # Use interpolation to get leverage function value
        return float(self.interpolator(time, strike))
    
    def get_local_volatility(
        self,
        strike: float,
        time: float,
        variance: float
    ) -> float:
        """
        Get local volatility for a specific strike, time, and variance level.
        
        Args:
            strike: Strike price
            time: Time to expiration
            variance: Current variance level
            
        Returns:
            Local volatility value
        """
        if not self.calibrated:
            raise ValueError("Model must be calibrated before getting local volatility")
        
        # Calculate Heston volatility
        heston_vol = np.sqrt(variance)
        
        # Get leverage function value
        leverage = self.get_leverage(strike, time)
        
        # Calculate local volatility
        local_vol = leverage * heston_vol
        
        return local_vol
    
    def simulate_paths(
        self,
        num_paths: int,
        time_horizon: float,
        num_steps: int,
        underlying_price: float,
        risk_free_rate: float = 0.05,
        dividend_yield: float = 0.0,
        random_seed: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate asset price paths using the stochastic-local volatility model.
        
        Args:
            num_paths: Number of paths to simulate
            time_horizon: Time horizon in years
            num_steps: Number of time steps
            underlying_price: Current price of the underlying asset
            risk_free_rate: Risk-free interest rate (annualized)
            dividend_yield: Dividend yield for the underlying (annualized)
            random_seed: Random seed for reproducibility
            
        Returns:
            Tuple containing:
            - 2D array of simulated asset price paths [path, time]
            - 2D array of simulated variance paths [path, time]
        """
        if not self.calibrated:
            raise ValueError("Model must be calibrated before simulating paths")
        
        # Set random seed if provided
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # Time step
        dt = time_horizon / num_steps
        sqrt_dt = np.sqrt(dt)
        
        # Initialize arrays for paths
        asset_paths = np.zeros((num_paths, num_steps + 1))
        variance_paths = np.zeros((num_paths, num_steps + 1))
        
        # Set initial values
        asset_paths[:, 0] = underlying_price
        variance_paths[:, 0] = self.v0
        
        # Generate correlated random numbers
        z1 = np.random.normal(0, 1, (num_paths, num_steps))
        z2 = self.rho * z1 + np.sqrt(1 - self.rho**2) * np.random.normal(0, 1, (num_paths, num_steps))
        
        # Simulate paths
        for i in range(num_steps):
            # Current values
            s = asset_paths[:, i]
            v = variance_paths[:, i]
            t = i * dt
            
            # Ensure positive variance
            v = np.maximum(v, 1e-6)
            
            # Calculate local volatility adjustment
            leverage = np.array([
                self.get_leverage(s_i, t) for s_i in s
            ])
            
            # Calculate drift and diffusion terms
            drift_s = (risk_free_rate - dividend_yield) * s * dt
            drift_v = self.kappa * (self.theta - v) * dt
            
            diffusion_s = leverage * np.sqrt(v) * s * sqrt_dt * z1[:, i]
            diffusion_v = self.sigma * np.sqrt(v) * sqrt_dt * z2[:, i]
            
            # Update paths
            asset_paths[:, i+1] = s + drift_s + diffusion_s
            variance_paths[:, i+1] = v + drift_v + diffusion_v
            
            # Ensure positive values
            asset_paths[:, i+1] = np.maximum(asset_paths[:, i+1], 1e-6)
            variance_paths[:, i+1] = np.maximum(variance_paths[:, i+1], 1e-6)
        
        return asset_paths, variance_paths
    
    def price_european_option(
        self,
        option_type: str,
        strike: float,
        time_to_expiry: float,
        underlying_price: float,
        risk_free_rate: float = 0.05,
        dividend_yield: float = 0.0,
        num_paths: int = 10000,
        num_steps: int = 100,
        random_seed: Optional[int] = None
    ) -> float:
        """
        Price a European option using Monte Carlo simulation.
        
        Args:
            option_type: 'call' or 'put'
            strike: Strike price
            time_to_expiry: Time to expiration in years
            underlying_price: Current price of the underlying asset
            risk_free_rate: Risk-free interest rate (annualized)
            dividend_yield: Dividend yield for the underlying (annualized)
            num_paths: Number of paths to simulate
            num_steps: Number of time steps
            random_seed: Random seed for reproducibility
            
        Returns:
            Option price
        """
        if not self.calibrated:
            raise ValueError("Model must be calibrated before pricing options")
        
        # Validate option type
        if option_type.lower() not in ['call', 'put']:
            raise ValueError("Option type must be 'call' or 'put'")
        
        # Simulate paths
        asset_paths, _ = self.simulate_paths(
            num_paths, time_to_expiry, num_steps, underlying_price,
            risk_free_rate, dividend_yield, random_seed
        )
        
        # Calculate payoffs
        if option_type.lower() == 'call':
            payoffs = np.maximum(asset_paths[:, -1] - strike, 0)
        else:  # put
            payoffs = np.maximum(strike - asset_paths[:, -1], 0)
        
        # Calculate option price (discounted expected payoff)
        option_price = np.exp(-risk_free_rate * time_to_expiry) * np.mean(payoffs)
        
        return option_price
    
    def plot_leverage_function(
        self,
        title: str = "Leverage Function",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot the calibrated leverage function.
        
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
            X, Y, self.leverage_function, 
            cmap=cm.viridis,
            linewidth=0, 
            antialiased=False
        )
        
        ax.set_xlabel('Strike')
        ax.set_ylabel('Time to Expiry')
        ax.set_zlabel('Leverage Function')
        ax.set_title(title)
        
        fig.colorbar(surf, shrink=0.5, aspect=5)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved leverage function plot to {save_path}")
        
        return fig
    
    def plot_simulated_paths(
        self,
        num_paths: int = 10,
        time_horizon: float = 1.0,
        num_steps: int = 100,
        underlying_price: float = 100.0,
        risk_free_rate: float = 0.05,
        dividend_yield: float = 0.0,
        random_seed: int = 42,
        title: str = "Simulated Asset Price Paths",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot simulated asset price paths.
        
        Args:
            num_paths: Number of paths to simulate
            time_horizon: Time horizon in years
            num_steps: Number of time steps
            underlying_price: Current price of the underlying asset
            risk_free_rate: Risk-free interest rate (annualized)
            dividend_yield: Dividend yield for the underlying (annualized)
            random_seed: Random seed for reproducibility
            title: Plot title
            save_path: If provided, save the figure to this path
            
        Returns:
            Matplotlib figure object
        """
        if not self.calibrated:
            raise ValueError("Model must be calibrated before simulating paths")
        
        # Simulate paths
        asset_paths, variance_paths = self.simulate_paths(
            num_paths, time_horizon, num_steps, underlying_price,
            risk_free_rate, dividend_yield, random_seed
        )
        
        # Create time grid
        time_grid = np.linspace(0, time_horizon, num_steps + 1)
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot asset price paths
        for i in range(num_paths):
            ax1.plot(time_grid, asset_paths[i, :])
        
        ax1.set_xlabel('Time (years)')
        ax1.set_ylabel('Asset Price')
        ax1.set_title('Simulated Asset Price Paths')
        ax1.grid(True)
        
        # Plot variance paths
        for i in range(num_paths):
            ax2.plot(time_grid, np.sqrt(variance_paths[i, :]) * 100)  # Convert to percentage
        
        ax2.set_xlabel('Time (years)')
        ax2.set_ylabel('Volatility (%)')
        ax2.set_title('Simulated Volatility Paths')
        ax2.grid(True)
        
        # Set overall title
        fig.suptitle(title, fontsize=16)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved simulated paths plot to {save_path}")
        
        return fig
