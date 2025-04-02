# src/models/sabr_model.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import logging
from typing import Dict, List, Optional, Tuple, Union, Callable
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SABRModel:
    """
    Implementation of the SABR (Stochastic Alpha Beta Rho) model for commodity options.
    
    The SABR model is a stochastic volatility model that captures the volatility smile
    and is widely used in interest rate and commodity markets. The model is defined by:
    
    dF_t = α_t * F_t^β * dW_t
    dα_t = ν * α_t * dZ_t
    d<W_t, Z_t> = ρ * dt
    
    where:
    - F_t is the forward price
    - α_t is the instantaneous volatility
    - β is the CEV (Constant Elasticity of Variance) parameter
    - ν is the volatility of volatility
    - ρ is the correlation between the forward price and volatility
    """
    
    def __init__(
        self,
        alpha: float = 0.2,
        beta: float = 0.5,
        rho: float = -0.3,
        nu: float = 0.4
    ):
        """
        Initialize the SABR model with parameters.
        
        Args:
            alpha: Initial volatility parameter
            beta: CEV parameter (0 for normal, 1 for lognormal)
            rho: Correlation between price and volatility
            nu: Volatility of volatility
        """
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.nu = nu
        
        # Calibration results
        self.calibrated = False
        self.calibrated_params = {}
        self.term_structure = {}
    
    def calibrate(
        self,
        option_data: pd.DataFrame,
        forward_curve: Dict[str, float] = None,
        method: str = 'L-BFGS-B',
        by_expiry: bool = True
    ) -> Dict:
        """
        Calibrate the SABR model to market option prices.
        
        Args:
            option_data: DataFrame containing option data (strikes, expiries, implied vols)
            forward_curve: Dictionary mapping expiry to forward price
            method: Optimization method to use
            by_expiry: Whether to calibrate separately for each expiry
            
        Returns:
            Dictionary containing calibration results
        """
        logger.info("Calibrating SABR model...")
        
        # Extract data
        if 'expiration' in option_data.columns:
            option_data = option_data.rename(columns={'expiration': 'expiry'})
        
        if by_expiry:
            # Calibrate separately for each expiry
            calibration_results = {}
            
            # Group by expiry
            grouped = option_data.groupby('expiry')
            
            for expiry, group in grouped:
                logger.info(f"Calibrating for expiry: {expiry}")
                
                # Get forward price for this expiry
                if forward_curve is not None and expiry in forward_curve:
                    forward = forward_curve[expiry]
                elif 'forward' in group.columns:
                    forward = group['forward'].iloc[0]
                else:
                    # Use the underlying price as an approximation
                    forward = group['underlying_price'].iloc[0] if 'underlying_price' in group.columns else 100.0
                
                # Extract strikes and implied volatilities
                strikes = group['strike'].values
                implied_vols = group['implied_vol'].values
                time_to_expiry = group['time_to_expiry'].iloc[0]
                
                # Calibrate for this expiry
                result = self._calibrate_single_expiry(
                    strikes, implied_vols, time_to_expiry, forward, method
                )
                
                # Store results
                calibration_results[expiry] = {
                    'alpha': result['alpha'],
                    'beta': result['beta'],
                    'rho': result['rho'],
                    'nu': result['nu'],
                    'forward': forward,
                    'time_to_expiry': time_to_expiry,
                    'rmse': result['rmse']
                }
                
                logger.info(f"Calibration for expiry {expiry} completed with RMSE: {result['rmse']:.6f}")
            
            # Store calibration results
            self.calibrated_params = calibration_results
            self.calibrated = True
            
            return calibration_results
        else:
            # Calibrate all expiries together
            # This is more challenging and may not work well in practice
            logger.warning("Calibrating all expiries together is not recommended for SABR model")
            
            # Extract data
            strikes = option_data['strike'].values
            implied_vols = option_data['implied_vol'].values
            expiries = option_data['time_to_expiry'].values
            
            # Get forward prices
            if forward_curve is not None:
                forwards = np.array([forward_curve.get(expiry, 100.0) for expiry in option_data['expiry']])
            elif 'forward' in option_data.columns:
                forwards = option_data['forward'].values
            else:
                # Use the underlying price as an approximation
                forwards = option_data['underlying_price'].values if 'underlying_price' in option_data.columns else np.full_like(strikes, 100.0)
            
            # Define objective function
            def objective(params):
                alpha, beta, rho, nu = params
                
                # Ensure parameters are within bounds
                if alpha <= 0 or beta < 0 or beta > 1 or abs(rho) >= 1 or nu <= 0:
                    return 1e10
                
                # Calculate model implied volatilities
                model_vols = np.array([
                    self.implied_volatility(
                        strike, expiry, forward, alpha, beta, rho, nu
                    )
                    for strike, expiry, forward in zip(strikes, expiries, forwards)
                ])
                
                # Calculate error
                error = np.sum((model_vols - implied_vols) ** 2)
                
                return error
            
            # Initial parameters
            initial_params = [self.alpha, self.beta, self.rho, self.nu]
            
            # Bounds for parameters
            bounds = [
                (0.001, 1.0),    # alpha: initial volatility
                (0.0, 1.0),      # beta: CEV parameter
                (-0.999, 0.999), # rho: correlation
                (0.001, 2.0)     # nu: volatility of volatility
            ]
            
            # Perform optimization
            result = minimize(
                objective,
                initial_params,
                method=method,
                bounds=bounds
            )
            
            if result.success:
                # Update parameters
                self.alpha, self.beta, self.rho, self.nu = result.x
                
                # Calculate RMSE
                model_vols = np.array([
                    self.implied_volatility(
                        strike, expiry, forward, self.alpha, self.beta, self.rho, self.nu
                    )
                    for strike, expiry, forward in zip(strikes, expiries, forwards)
                ])
                rmse = np.sqrt(np.mean((model_vols - implied_vols) ** 2))
                
                logger.info(f"Calibration completed with RMSE: {rmse:.6f}")
                
                # Store calibration results
                self.calibrated_params = {
                    'alpha': self.alpha,
                    'beta': self.beta,
                    'rho': self.rho,
                    'nu': self.nu,
                    'rmse': rmse
                }
                self.calibrated = True
                
                return self.calibrated_params
            else:
                logger.error(f"Calibration failed: {result.message}")
                return None
    
    def _calibrate_single_expiry(
        self,
        strikes: np.ndarray,
        implied_vols: np.ndarray,
        time_to_expiry: float,
        forward: float,
        method: str
    ) -> Dict:
        """
        Calibrate SABR parameters for a single expiry.
        
        Args:
            strikes: Array of strike prices
            implied_vols: Array of implied volatilities
            time_to_expiry: Time to expiration in years
            forward: Forward price
            method: Optimization method to use
            
        Returns:
            Dictionary containing calibrated parameters
        """
        # Define objective function
        def objective(params):
            alpha, beta, rho, nu = params
            
            # Ensure parameters are within bounds
            if alpha <= 0 or beta < 0 or beta > 1 or abs(rho) >= 1 or nu <= 0:
                return 1e10
            
            # Calculate model implied volatilities
            model_vols = np.array([
                self.implied_volatility(
                    strike, time_to_expiry, forward, alpha, beta, rho, nu
                )
                for strike in strikes
            ])
            
            # Calculate error
            error = np.sum((model_vols - implied_vols) ** 2)
            
            return error
        
        # Initial parameters
        initial_params = [self.alpha, self.beta, self.rho, self.nu]
        
        # Bounds for parameters
        bounds = [
            (0.001, 1.0),    # alpha: initial volatility
            (0.0, 1.0),      # beta: CEV parameter
            (-0.999, 0.999), # rho: correlation
            (0.001, 2.0)     # nu: volatility of volatility
        ]
        
        # Perform optimization
        result = minimize(
            objective,
            initial_params,
            method=method,
            bounds=bounds
        )
        
        if result.success:
            # Extract optimized parameters
            alpha, beta, rho, nu = result.x
            
            # Calculate RMSE
            model_vols = np.array([
                self.implied_volatility(
                    strike, time_to_expiry, forward, alpha, beta, rho, nu
                )
                for strike in strikes
            ])
            rmse = np.sqrt(np.mean((model_vols - implied_vols) ** 2))
            
            return {
                'alpha': alpha,
                'beta': beta,
                'rho': rho,
                'nu': nu,
                'rmse': rmse
            }
        else:
            logger.error(f"Calibration failed for expiry {time_to_expiry}: {result.message}")
            return {
                'alpha': self.alpha,
                'beta': self.beta,
                'rho': self.rho,
                'nu': self.nu,
                'rmse': float('inf')
            }
    
    def implied_volatility(
        self,
        strike: float,
        time_to_expiry: float,
        forward: float,
        alpha: float = None,
        beta: float = None,
        rho: float = None,
        nu: float = None
    ) -> float:
        """
        Calculate implied volatility using the SABR formula.
        
        Args:
            strike: Strike price
            time_to_expiry: Time to expiration in years
            forward: Forward price
            alpha: Volatility parameter (use calibrated value if None)
            beta: CEV parameter (use calibrated value if None)
            rho: Correlation parameter (use calibrated value if None)
            nu: Volatility of volatility parameter (use calibrated value if None)
            
        Returns:
            SABR implied volatility
        """
        # Use calibrated parameters if not provided
        if alpha is None:
            alpha = self.alpha
        if beta is None:
            beta = self.beta
        if rho is None:
            rho = self.rho
        if nu is None:
            nu = self.nu
        
        # Handle ATM case separately
        if abs(forward - strike) < 1e-10:
            return self._sabr_atm_vol(forward, time_to_expiry, alpha, beta, rho, nu)
        
        # Handle extreme strikes
        if strike <= 0 or forward <= 0:
            return 0.999  # Return a high value
        
        # Calculate z
        x = np.log(forward / strike)
        fk_beta = (forward * strike) ** ((1 - beta) / 2)
        z = (nu / alpha) * fk_beta * x
        
        # Calculate chi
        if abs(z) < 1e-10:
            # For small z, use Taylor expansion
            chi = 1 + (1/2) * (rho * z + z**2 / 6 * (3 - rho**2))
        else:
            chi = np.log((np.sqrt(1 - 2*rho*z + z**2) + z - rho) / (1 - rho))
        
        # Calculate the volatility
        term1 = alpha / fk_beta
        term2 = (1 + ((1 - beta)**2 / 24) * (x**2) + ((1 - beta)**4 / 1920) * (x**4))
        term3 = z / chi
        term4 = 1 + (((2 - 3*rho**2) / 24) * (nu**2) * time_to_expiry)
        
        return term1 * term2 * term3 * term4
    
    def _sabr_atm_vol(
        self,
        forward: float,
        time_to_expiry: float,
        alpha: float,
        beta: float,
        rho: float,
        nu: float
    ) -> float:
        """
        Calculate ATM implied volatility using the SABR formula.
        
        Args:
            forward: Forward price
            time_to_expiry: Time to expiration in years
            alpha: Volatility parameter
            beta: CEV parameter
            rho: Correlation parameter
            nu: Volatility of volatility parameter
            
        Returns:
            ATM SABR implied volatility
        """
        # ATM formula
        term1 = alpha / (forward ** (1 - beta))
        term2 = 1 + (((2 - 3 * rho**2) / 24) * (nu**2) * time_to_expiry)
        
        return term1 * term2
    
    def get_volatility_surface(
        self,
        strikes: np.ndarray,
        expiries: np.ndarray,
        forward_curve: Dict[str, float] = None
    ) -> np.ndarray:
        """
        Calculate the implied volatility surface using the SABR model.
        
        Args:
            strikes: Array of strike prices
            expiries: Array of expiry times
            forward_curve: Dictionary mapping expiry to forward price
            
        Returns:
            2D array of implied volatilities [expiry, strike]
        """
        if not self.calibrated:
            raise ValueError("Model must be calibrated before calculating volatility surface")
        
        # Initialize surface
        surface = np.zeros((len(expiries), len(strikes)))
        
        # Calculate volatilities
        for i, expiry in enumerate(expiries):
            # Find closest calibrated expiry
            if isinstance(list(self.calibrated_params.keys())[0], str):
                # If keys are strings (e.g., dates), convert expiry to string
                expiry_str = str(expiry)
                closest_expiry = min(self.calibrated_params.keys(), key=lambda x: abs(float(x) - float(expiry_str)))
            else:
                # If keys are numeric
                closest_expiry = min(self.calibrated_params.keys(), key=lambda x: abs(x - expiry))
            
            # Get parameters for this expiry
            params = self.calibrated_params[closest_expiry]
            
            # Get forward price
            if forward_curve is not None and expiry in forward_curve:
                forward = forward_curve[expiry]
            else:
                forward = params.get('forward', 100.0)
            
            # Calculate volatilities for this expiry
            for j, strike in enumerate(strikes):
                surface[i, j] = self.implied_volatility(
                    strike, expiry, forward,
                    params['alpha'], params['beta'], params['rho'], params['nu']
                )
        
        return surface
    
    def plot_volatility_smile(
        self,
        strikes: np.ndarray,
        expiry: float,
        forward: float,
        market_vols: np.ndarray = None,
        title: str = "SABR Volatility Smile",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot the volatility smile for a specific expiry.
        
        Args:
            strikes: Array of strike prices
            expiry: Time to expiration
            forward: Forward price
            market_vols: Array of market implied volatilities (optional)
            title: Plot title
            save_path: If provided, save the figure to this path
            
        Returns:
            Matplotlib figure object
        """
        if not self.calibrated:
            raise ValueError("Model must be calibrated before plotting")
        
        # Find closest calibrated expiry
        if isinstance(list(self.calibrated_params.keys())[0], str):
            # If keys are strings (e.g., dates), convert expiry to string
            expiry_str = str(expiry)
            closest_expiry = min(self.calibrated_params.keys(), key=lambda x: abs(float(x) - float(expiry_str)))
        else:
            # If keys are numeric
            closest_expiry = min(self.calibrated_params.keys(), key=lambda x: abs(x - expiry))
        
        # Get parameters for this expiry
        params = self.calibrated_params[closest_expiry]
        
        # Calculate model volatilities
        model_vols = np.array([
            self.implied_volatility(
                strike, expiry, forward,
                params['alpha'], params['beta'], params['rho'], params['nu']
            )
            for strike in strikes
        ])
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot model volatilities
        ax.plot(strikes, model_vols, 'b-', linewidth=2, label='SABR Model')
        
        # Plot market volatilities if provided
        if market_vols is not None:
            ax.plot(strikes, market_vols, 'ro', markersize=6, label='Market')
        
        # Add vertical line at forward price
        ax.axvline(x=forward, color='g', linestyle='--', label='Forward')
        
        # Set labels and title
        ax.set_xlabel('Strike')
        ax.set_ylabel('Implied Volatility')
        ax.set_title(title)
        
        # Add legend and grid
        ax.legend()
        ax.grid(True)
        
        # Save figure if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved figure to {save_path}")
    
        return fig

def plot_volatility_surface(
    self,
    strikes: np.ndarray,
    expiries: np.ndarray,
    forward_curve: Dict[str, float] = None,
    title: str = "SABR Volatility Surface",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot the volatility surface.
    
    Args:
        strikes: Array of strike prices
        expiries: Array of expiry times
        forward_curve: Dictionary mapping expiry to forward price
        title: Plot title
        save_path: If provided, save the figure to this path
        
    Returns:
        Matplotlib figure object
    """
    if not self.calibrated:
        raise ValueError("Model must be calibrated before plotting")
    
    # Calculate volatility surface
    surface = self.get_volatility_surface(strikes, expiries, forward_curve)
    
    # Create meshgrid
    X, Y = np.meshgrid(strikes, expiries)
    
    # Create figure
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot surface
    surf = ax.plot_surface(
        X, Y, surface, 
        cmap='viridis',
        linewidth=0, 
        antialiased=True,
        alpha=0.8
    )
    
    # Set labels and title
    ax.set_xlabel('Strike')
    ax.set_ylabel('Time to Expiry')
    ax.set_zlabel('Implied Volatility')
    ax.set_title(title)
    
    # Add colorbar
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    
    # Save figure if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved figure to {save_path}")
    
    return fig

def plot_parameter_evolution(
    self,
    title: str = "SABR Parameter Evolution",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot the evolution of SABR parameters across different expiries.
    
    Args:
        title: Plot title
        save_path: If provided, save the figure to this path
        
    Returns:
        Matplotlib figure object
    """
    if not self.calibrated:
        raise ValueError("Model must be calibrated before plotting")
    
    # Extract expiries and parameters
    expiries = []
    alphas = []
    betas = []
    rhos = []
    nus = []
    
    for expiry, params in self.calibrated_params.items():
        # Convert expiry to float if it's a string
        if isinstance(expiry, str):
            try:
                expiry_float = float(expiry)
            except:
                # If it's a date string, use time_to_expiry from params
                expiry_float = params['time_to_expiry']
        else:
            expiry_float = expiry
        
        expiries.append(expiry_float)
        alphas.append(params['alpha'])
        betas.append(params['beta'])
        rhos.append(params['rho'])
        nus.append(params['nu'])
    
    # Sort by expiry
    sorted_indices = np.argsort(expiries)
    expiries = [expiries[i] for i in sorted_indices]
    alphas = [alphas[i] for i in sorted_indices]
    betas = [betas[i] for i in sorted_indices]
    rhos = [rhos[i] for i in sorted_indices]
    nus = [nus[i] for i in sorted_indices]
    
    # Create figure with subplots
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot alpha
    axs.plot(expiries, alphas, 'o-', linewidth=2)
    axs.set_xlabel('Time to Expiry')
    axs.set_ylabel('Alpha')
    axs.set_title('Alpha Parameter')
    axs.grid(True)
    
    # Plot beta
    axs.plot(expiries, betas, 'o-', linewidth=2)
    axs.set_xlabel('Time to Expiry')
    axs.set_ylabel('Beta')
    ('Beta Parameter')
    axs.grid(True)
    
    # Plot rho
    axs.plot(expiries, rhos, 'o-', linewidth=2)
    axs.set_xlabel('Time to Expiry')
    axs.set_ylabel('Rho')
    axs.set_title('Rho Parameter')
    axs.grid(True)
    
    # Plot nu
    axs.plot(expiries, nus, 'o-', linewidth=2)
    axs.set_xlabel('Time to Expiry')
    axs.set_ylabel('Nu')
    axs.set_title('Nu Parameter')
    axs.grid(True)
    
    # Set overall title
    fig.suptitle(title, fontsize=16)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Save figure if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved figure to {save_path}")
    
    return fig

def simulate_paths(
    self,
    num_paths: int,
    time_horizon: float,
    num_steps: int,
    forward: float,
    alpha: float = None,
    beta: float = None,
    rho: float = None,
    nu: float = None,
    random_seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate asset price paths using the SABR model.
    
    Args:
        num_paths: Number of paths to simulate
        time_horizon: Time horizon in years
        num_steps: Number of time steps
        forward: Initial forward price
        alpha: Volatility parameter (use calibrated value if None)
        beta: CEV parameter (use calibrated value if None)
        rho: Correlation parameter (use calibrated value if None)
        nu: Volatility of volatility parameter (use calibrated value if None)
        random_seed: Random seed for reproducibility
        
    Returns:
        Tuple containing:
        - 2D array of simulated forward price paths [path, time]
        - 2D array of simulated volatility paths [path, time]
    """
    # Use calibrated parameters if not provided
    if alpha is None:
        alpha = self.alpha
    if beta is None:
        beta = self.beta
    if rho is None:
        rho = self.rho
    if nu is None:
        nu = self.nu
    
    # Set random seed if provided
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Time step
    dt = time_horizon / num_steps
    sqrt_dt = np.sqrt(dt)
    
    # Initialize arrays for paths
    forward_paths = np.zeros((num_paths, num_steps + 1))
    alpha_paths = np.zeros((num_paths, num_steps + 1))
    
    # Set initial values
    forward_paths[:, 0] = forward
    alpha_paths[:, 0] = alpha
    
    # Generate correlated random numbers
    z1 = np.random.normal(0, 1, (num_paths, num_steps))
    z2 = rho * z1 + np.sqrt(1 - rho**2) * np.random.normal(0, 1, (num_paths, num_steps))
    
    # Simulate paths
    for i in range(num_steps):
        # Current values
        f = forward_paths[:, i]
        a = alpha_paths[:, i]
        
        # Ensure positive values
        f = np.maximum(f, 1e-6)
        a = np.maximum(a, 1e-6)
        
        # Calculate diffusion terms
        diffusion_f = a * (f ** beta) * sqrt_dt * z1[:, i]
        diffusion_a = nu * a * sqrt_dt * z2[:, i]
        
        # Update paths
        forward_paths[:, i+1] = f + diffusion_f
        alpha_paths[:, i+1] = a * np.exp(diffusion_a - 0.5 * nu**2 * dt)  # Log-normal process
        
        # Ensure positive values
        forward_paths[:, i+1] = np.maximum(forward_paths[:, i+1], 1e-6)
        alpha_paths[:, i+1] = np.maximum(alpha_paths[:, i+1], 1e-6)
    
    return forward_paths, alpha_paths

def plot_simulated_paths(
    self,
    num_paths: int = 10,
    time_horizon: float = 1.0,
    num_steps: int = 100,
    forward: float = 100.0,
    alpha: float = None,
    beta: float = None,
    rho: float = None,
    nu: float = None,
    random_seed: int = 42,
    title: str = "SABR Simulated Paths",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot simulated paths from the SABR model.
    
    Args:
        num_paths: Number of paths to simulate
        time_horizon: Time horizon in years
        num_steps: Number of time steps
        forward: Initial forward price
        alpha: Volatility parameter (use calibrated value if None)
        beta: CEV parameter (use calibrated value if None)
        rho: Correlation parameter (use calibrated value if None)
        nu: Volatility of volatility parameter (use calibrated value if None)
        random_seed: Random seed for reproducibility
        title: Plot title
        save_path: If provided, save the figure to this path
        
    Returns:
        Matplotlib figure object
    """
    # Simulate paths
    forward_paths, alpha_paths = self.simulate_paths(
        num_paths, time_horizon, num_steps, forward,
        alpha, beta, rho, nu, random_seed
    )
    
    # Create time grid
    time_grid = np.linspace(0, time_horizon, num_steps + 1)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot forward price paths
    for i in range(num_paths):
        ax1.plot(time_grid, forward_paths[i, :])
    
    ax1.set_xlabel('Time (years)')
    ax1.set_ylabel('Forward Price')
    ax1.set_title('Simulated Forward Price Paths')
    ax1.grid(True)
    
    # Plot volatility paths
    for i in range(num_paths):
        ax2.plot(time_grid, alpha_paths[i, :])
    
    ax2.set_xlabel('Time (years)')
    ax2.set_ylabel('Volatility (alpha)')
    ax2.set_title('Simulated Volatility Paths')
    ax2.grid(True)
    
    # Set overall title
    fig.suptitle(title, fontsize=16)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Save figure if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved figure to {save_path}")
    
    return fig

