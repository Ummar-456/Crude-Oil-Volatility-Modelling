# src/utils/arbitrage_checker.py

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ArbitrageChecker:
    """
    A class to check for arbitrage opportunities in volatility surfaces.
    
    This class implements various tests for static arbitrage constraints:
    - Butterfly arbitrage (convexity in strike dimension)
    - Calendar spread arbitrage (monotonicity in time dimension)
    - Absence of negative implied volatilities
    - Absence of negative option prices
    """
    
    def __init__(self):
        """Initialize the ArbitrageChecker."""
        pass
    
    def check_butterfly_arbitrage(
        self,
        strikes: np.ndarray,
        implied_vols: np.ndarray,
        time_to_expiry: float,
        underlying_price: float,
        risk_free_rate: float = 0.05,
        tolerance: float = 1e-6
    ) -> Tuple[bool, np.ndarray]:
        """
        Check for butterfly arbitrage (non-convexity in strike dimension).
        
        Args:
            strikes: Array of strike prices
            implied_vols: Array of implied volatilities corresponding to strikes
            time_to_expiry: Time to expiration in years
            underlying_price: Current price of the underlying asset
            risk_free_rate: Risk-free interest rate (annualized)
            tolerance: Tolerance for numerical errors
            
        Returns:
            Tuple containing:
            - Boolean indicating if arbitrage is present
            - Array of butterfly spreads (negative values indicate arbitrage)
        """
        if len(strikes) < 3:
            logger.warning("At least 3 strikes needed to check butterfly arbitrage")
            return False, np.array([])
        
        # Sort by strike price
        sorted_indices = np.argsort(strikes)
        sorted_strikes = strikes[sorted_indices]
        sorted_vols = implied_vols[sorted_indices]
        
        # Calculate call prices
        call_prices = np.array([
            self._black_scholes_price(
                underlying_price, k, time_to_expiry, risk_free_rate, vol, 'call'
            )
            for k, vol in zip(sorted_strikes, sorted_vols)
        ])
        
        # Calculate butterfly spreads
        butterfly_spreads = np.zeros(len(sorted_strikes) - 2)
        
        for i in range(len(sorted_strikes) - 2):
            # Butterfly = Call(K1) - 2*Call(K2) + Call(K3) where K1 < K2 < K3
            k1, k2, k3 = sorted_strikes[i], sorted_strikes[i+1], sorted_strikes[i+2]
            c1, c2, c3 = call_prices[i], call_prices[i+1], call_prices[i+2]
            
            # Weight the middle option based on strike distances
            # This accounts for unequal strike spacing
            w = (k3 - k2) / (k3 - k1)
            butterfly = c1 * (1 - w) + c3 * w - c2
            
            butterfly_spreads[i] = butterfly
        
        # Check if any butterfly spread is negative (allowing for numerical errors)
        has_arbitrage = np.any(butterfly_spreads < -tolerance)
        
        if has_arbitrage:
            logger.warning(f"Butterfly arbitrage detected: min spread = {butterfly_spreads.min()}")
        
        return has_arbitrage, butterfly_spreads
    
    def check_calendar_spread_arbitrage(
        self,
        implied_vols: np.ndarray,
        times_to_expiry: np.ndarray,
        strikes: np.ndarray,
        underlying_price: float,
        risk_free_rate: float = 0.05,
        tolerance: float = 1e-6
    ) -> Tuple[bool, np.ndarray]:
        """
        Check for calendar spread arbitrage (non-monotonicity in time dimension).
        
        Args:
            implied_vols: 2D array of implied volatilities [expiry, strike]
            times_to_expiry: Array of times to expiration
            strikes: Array of strike prices
            underlying_price: Current price of the underlying asset
            risk_free_rate: Risk-free interest rate (annualized)
            tolerance: Tolerance for numerical errors
            
        Returns:
            Tuple containing:
            - Boolean indicating if arbitrage is present
            - 2D array of calendar spreads (negative values indicate arbitrage)
        """
        if len(times_to_expiry) < 2:
            logger.warning("At least 2 expiries needed to check calendar spread arbitrage")
            return False, np.array([])
        
        # Sort by time to expiry
        sorted_indices = np.argsort(times_to_expiry)
        sorted_times = times_to_expiry[sorted_indices]
        sorted_vols = implied_vols[sorted_indices]
        
        # Calculate total variance (volatility squared * time)
        total_variance = np.zeros_like(sorted_vols)
        for i, t in enumerate(sorted_times):
            total_variance[i] = sorted_vols[i] ** 2 * t
        
        # Check if total variance is non-decreasing with time
        # For each strike, check if total variance increases with time
        calendar_spreads = np.zeros((len(sorted_times) - 1, len(strikes)))
        
        for j in range(len(strikes)):
            for i in range(len(sorted_times) - 1):
                # Calendar spread = TotalVar(T2) - TotalVar(T1) where T2 > T1
                calendar_spreads[i, j] = total_variance[i+1, j] - total_variance[i, j]
        
        # Check if any calendar spread is negative (allowing for numerical errors)
        has_arbitrage = np.any(calendar_spreads < -tolerance)
        
        if has_arbitrage:
            logger.warning(f"Calendar spread arbitrage detected: min spread = {calendar_spreads.min()}")
        
        return has_arbitrage, calendar_spreads
    
    def check_negative_implied_volatility(
        self,
        implied_vols: np.ndarray,
        tolerance: float = 1e-6
    ) -> Tuple[bool, np.ndarray]:
        """
        Check for negative implied volatilities.
        
        Args:
            implied_vols: Array of implied volatilities
            tolerance: Tolerance for numerical errors
            
        Returns:
            Tuple containing:
            - Boolean indicating if negative volatilities are present
            - Array of booleans indicating which volatilities are negative
        """
        negative_vols = implied_vols < tolerance
        has_negative_vols = np.any(negative_vols)
        
        if has_negative_vols:
            logger.warning(f"Negative implied volatilities detected: min vol = {implied_vols.min()}")
        
        return has_negative_vols, negative_vols
    
    def check_negative_option_prices(
        self,
        strikes: np.ndarray,
        implied_vols: np.ndarray,
        times_to_expiry: np.ndarray,
        underlying_price: float,
        risk_free_rate: float = 0.05,
        tolerance: float = 1e-6
    ) -> Tuple[bool, np.ndarray]:
        """
        Check for negative option prices.
        
        Args:
            strikes: Array of strike prices
            implied_vols: 2D array of implied volatilities [expiry, strike]
            times_to_expiry: Array of times to expiration
            underlying_price: Current price of the underlying asset
            risk_free_rate: Risk-free interest rate (annualized)
            tolerance: Tolerance for numerical errors
            
        Returns:
            Tuple containing:
            - Boolean indicating if negative prices are present
            - 2D array of booleans indicating which prices are negative
        """
        # Calculate call and put prices
        call_prices = np.zeros((len(times_to_expiry), len(strikes)))
        put_prices = np.zeros((len(times_to_expiry), len(strikes)))
        
        for i, t in enumerate(times_to_expiry):
            for j, k in enumerate(strikes):
                vol = implied_vols[i, j]
                call_prices[i, j] = self._black_scholes_price(
                    underlying_price, k, t, risk_free_rate, vol, 'call'
                )
                put_prices[i, j] = self._black_scholes_price(
                    underlying_price, k, t, risk_free_rate, vol, 'put'
                )
        
        # Check for negative prices
        negative_calls = call_prices < -tolerance
        negative_puts = put_prices < -tolerance
        negative_prices = np.logical_or(negative_calls, negative_puts)
        
        has_negative_prices = np.any(negative_prices)
        
        if has_negative_prices:
            logger.warning(f"Negative option prices detected: min call = {call_prices.min()}, min put = {put_prices.min()}")
        
        return has_negative_prices, negative_prices
    
    def calculate_arbitrage_penalty(
        self,
        surface_data: Dict,
        underlying_price: float,
        risk_free_rate: float = 0.05
    ) -> float:
        """
        Calculate a penalty function quantifying arbitrage violations.
        
        Args:
            surface_data: Dictionary containing volatility surface data
            underlying_price: Current price of the underlying asset
            risk_free_rate: Risk-free interest rate (annualized)
            
        Returns:
            Penalty value (higher values indicate more arbitrage)
        """
        # Extract data
        X = surface_data['x_grid']  # Log moneyness
        Y = surface_data['y_grid']  # Time to expiry
        Z = surface_data['z_grid']  # Implied volatility
        
        # Convert log moneyness to strikes
        strikes = underlying_price * np.exp(X[0, :])
        times = Y[:, 0]
        
        # Initialize penalty
        penalty = 0.0
        
        # Check butterfly arbitrage for each expiry
        for i, t in enumerate(times):
            vols = Z[i, :]
            _, butterfly_spreads = self.check_butterfly_arbitrage(
                strikes, vols, t, underlying_price, risk_free_rate
            )
            
            # Add negative spreads to penalty (butterfly arbitrage)
            negative_spreads = butterfly_spreads[butterfly_spreads < 0]
            if len(negative_spreads) > 0:
                penalty += np.sum(np.abs(negative_spreads))
        
        # Check calendar spread arbitrage
        _, calendar_spreads = self.check_calendar_spread_arbitrage(
            Z, times, strikes, underlying_price, risk_free_rate
        )
        
        # Add negative spreads to penalty (calendar spread arbitrage)
        negative_spreads = calendar_spreads[calendar_spreads < 0]
        if len(negative_spreads) > 0:
            penalty += np.sum(np.abs(negative_spreads))
        
        # Check negative implied volatilities
        _, negative_vols = self.check_negative_implied_volatility(Z)
        if np.any(negative_vols):
            penalty += np.sum(np.abs(Z[negative_vols]))
        
        # Check negative option prices
        _, negative_prices = self.check_negative_option_prices(
            strikes, Z, times, underlying_price, risk_free_rate
        )
        
        if np.any(negative_prices):
            # Calculate option prices
            call_prices = np.zeros_like(Z)
            for i, t in enumerate(times):
                for j, k in enumerate(strikes):
                    vol = Z[i, j]
                    call_prices[i, j] = self._black_scholes_price(
                        underlying_price, k, t, risk_free_rate, vol, 'call'
                    )
            
            # Add negative prices to penalty
            penalty += np.sum(np.abs(call_prices[negative_prices]))
        
        return penalty
    
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

