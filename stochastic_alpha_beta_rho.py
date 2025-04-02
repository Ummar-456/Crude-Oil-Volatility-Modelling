# src/models/stochastic_alpha_beta_rho.py

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Union
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SABRModel:
    """
    Stochastic Alpha Beta Rho (SABR) model for commodity options.
    
    The SABR model is defined by the following stochastic differential equations:
    dF_t = α_t * F_t^β * dW_t
    dα_t = ν * α_t * dZ_t
    d<W,Z>_t = ρ dt
    
    where:
    F_t is the forward price
    α_t is the instantaneous volatility
    β is the CEV (Constant Elasticity of Variance) parameter
    ν (nu) is the volatility of volatility
    ρ (rho) is the correlation between the forward price and volatility
    """

    def __init__(self, alpha: float = 0.2, beta: float = 0.5, rho: float = -0.3, nu: float = 0.4):
        """
        Initialize the SABR model with parameters.

        Args:
            alpha (float): Initial volatility
            beta (float): CEV parameter (0 <= beta <= 1)
            rho (float): Correlation between price and volatility (-1 <= rho <= 1)
            nu (float): Volatility of volatility
        """
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.nu = nu
        self.is_calibrated = False

    def calibrate(self, option_data: pd.DataFrame, forward_price: float) -> Dict[str, float]:
        """
        Calibrate the SABR model to market data.

        Args:
            option_data (pd.DataFrame): DataFrame containing 'strike', 'maturity', and 'implied_vol'
            forward_price (float): Current forward price

        Returns:
            Dict[str, float]: Calibrated parameters
        """
        logger.info("Starting SABR model calibration...")

        def objective(params):
            alpha, beta, rho, nu = params
            if not (0 <= beta <= 1) or not (-1 <= rho <= 1) or alpha <= 0 or nu <= 0:
                return np.inf
            
            model_vols = [self.implied_volatility(row['strike'], forward_price, row['maturity'], 
                                                  alpha, beta, rho, nu) 
                          for _, row in option_data.iterrows()]
            return np.sum((option_data['implied_vol'] - model_vols)**2)

        initial_guess = [self.alpha, self.beta, self.rho, self.nu]
        bounds = [(1e-5, None), (0, 1), (-1, 1), (1e-5, None)]

        result = minimize(objective, initial_guess, method='L-BFGS-B', bounds=bounds)

        if result.success:
            self.alpha, self.beta, self.rho, self.nu = result.x
            self.is_calibrated = True
            logger.info(f"Calibration successful. Parameters: alpha={self.alpha:.4f}, beta={self.beta:.4f}, rho={self.rho:.4f}, nu={self.nu:.4f}")
            return {'alpha': self.alpha, 'beta': self.beta, 'rho': self.rho, 'nu': self.nu}
        else:
            logger.error("Calibration failed.")
            raise ValueError("SABR model calibration failed.")

    def implied_volatility(self, strike: float, forward: float, maturity: float, 
                           alpha: Optional[float] = None, beta: Optional[float] = None, 
                           rho: Optional[float] = None, nu: Optional[float] = None) -> float:
        """
        Calculate the SABR implied volatility.

        Args:
            strike (float): Option strike price
            forward (float): Forward price
            maturity (float): Time to maturity in years
            alpha, beta, rho, nu (Optional[float]): SABR parameters (use calibrated if None)

        Returns:
            float: SABR implied volatility
        """
        alpha = alpha or self.alpha
        beta = beta or self.beta
        rho = rho or self.rho
        nu = nu or self.nu

        if not self.is_calibrated and any(p is None for p in (alpha, beta, rho, nu)):
            raise ValueError("Model not calibrated and parameters not provided.")

        # Handle ATM case separately
        if abs(forward - strike) < 1e-8:
            return self._sabr_atm_vol(forward, maturity, alpha, beta, rho, nu)

        x = np.log(forward / strike)
        z = (nu / alpha) * (forward * strike)**((1 - beta) / 2) * x
        chi = np.log((np.sqrt(1 - 2*rho*z + z**2) + z - rho) / (1 - rho))

        impl_var = (alpha**2 / ((forward * strike)**((1-beta))) * 
                    (1 + ((1-beta)**2/24) * x**2 + ((1-beta)**4/1920) * x**4) * 
                    z / chi * 
                    (1 + (((1-beta)**2/24) * alpha**2 / ((forward*strike)**(1-beta)) + 
                     (1/4) * rho * beta * nu * alpha / ((forward*strike)**((1-beta)/2)) + 
                     ((2-3*rho**2)/24) * nu**2) * maturity))

        return np.sqrt(impl_var / maturity)

    def _sabr_atm_vol(self, forward: float, maturity: float, alpha: float, beta: float, rho: float, nu: float) -> float:
        """Calculate SABR ATM implied volatility."""
        return (alpha / (forward**(1-beta))) * (1 + ((1-beta)**2/24 + (1-beta)**4/1920) * np.log(forward)**2 + 
                                                (rho*beta*nu*alpha)/(4*(forward**(1-beta))) + 
                                                ((2-3*rho**2)/24)*nu**2) * maturity

    def simulate_paths(self, num_paths: int, time_steps: int, forward: float, maturity: float, 
                       random_seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate price and volatility paths using the SABR model.

        Args:
            num_paths (int): Number of paths to simulate
            time_steps (int): Number of time steps
            forward (float): Initial forward price
            maturity (float): Time to maturity in years
            random_seed (Optional[int]): Random seed for reproducibility

        Returns:
            Tuple[np.ndarray, np.ndarray]: Simulated forward price and volatility paths
        """
        if not self.is_calibrated:
            raise ValueError("Model must be calibrated before simulation.")

        if random_seed is not None:
            np.random.seed(random_seed)

        dt = maturity / time_steps
        sqrt_dt = np.sqrt(dt)

        # Initialize arrays for paths
        forward_paths = np.zeros((num_paths, time_steps + 1))
        vol_paths = np.zeros((num_paths, time_steps + 1))

        # Set initial values
        forward_paths[:, 0] = forward
        vol_paths[:, 0] = self.alpha

        # Generate correlated random numbers
        rng = np.random.default_rng(random_seed)
        z1 = rng.standard_normal((num_paths, time_steps))
        z2 = self.rho * z1 + np.sqrt(1 - self.rho**2) * rng.standard_normal((num_paths, time_steps))

        for i in range(time_steps):
            # Update volatility
            vol_paths[:, i+1] = vol_paths[:, i] * np.exp(
                -0.5 * self.nu**2 * dt + self.nu * sqrt_dt * z2[:, i]
            )
            
            # Update forward price
            forward_paths[:, i+1] = forward_paths[:, i] * np.exp(
                -0.5 * vol_paths[:, i]**2 * forward_paths[:, i]**(2*self.beta-2) * dt + 
                vol_paths[:, i] * forward_paths[:, i]**(self.beta-1) * sqrt_dt * z1[:, i]
            )

        return forward_paths, vol_paths

    def price_european_option(self, option_type: str, strike: float, forward: float, maturity: float, 
                              num_paths: int = 100000, random_seed: Optional[int] = None) -> float:
        """
        Price a European option using Monte Carlo simulation.

        Args:
            option_type (str): 'call' or 'put'
            strike (float): Option strike price
            forward (float): Forward price
            maturity (float): Time to maturity in years
            num_paths (int): Number of Monte Carlo paths
            random_seed (Optional[int]): Random seed for reproducibility

        Returns:
            float: Option price
        """
        if option_type not in ['call', 'put']:
            raise ValueError("Option type must be 'call' or 'put'")

        forward_paths, _ = self.simulate_paths(num_paths, 100, forward, maturity, random_seed)
        final_prices = forward_paths[:, -1]

        if option_type == 'call':
            payoffs = np.maximum(final_prices - strike, 0)
        else:  # put
            payoffs = np.maximum(strike - final_prices, 0)

        option_price = np.mean(payoffs)
        return option_price

    def plot_volatility_smile(self, forward: float, maturity: float, strikes: np.ndarray, 
                              market_vols: Optional[np.ndarray] = None) -> None:
        """
        Plot the SABR volatility smile.

        Args:
            forward (float): Forward price
            maturity (float): Time to maturity in years
            strikes (np.ndarray): Array of strike prices
            market_vols (Optional[np.ndarray]): Market implied volatilities for comparison
        """
        if not self.is_calibrated:
            raise ValueError("Model must be calibrated before plotting.")

        sabr_vols = [self.implied_volatility(k, forward, maturity) for k in strikes]

        plt.figure(figsize=(10, 6))
        plt.plot(strikes, sabr_vols, label='SABR')
        if market_vols is not None:
            plt.scatter(strikes, market_vols, color='red', label='Market')
        plt.xlabel('Strike')
        plt.ylabel('Implied Volatility')
        plt.title(f'SABR Volatility Smile (F={forward:.2f}, T={maturity:.2f})')
        plt.legend()
        plt.grid(True)
        plt.show()
