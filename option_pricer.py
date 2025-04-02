# src/utils/option_pricer.py

import numpy as np
import pandas as pd
from scipy.stats import norm
from typing import Dict, List, Optional, Union, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OptionPricer:
    """
    A class for pricing options using various models.
    """

    def __init__(self, risk_free_rate: float = 0.05, dividend_yield: float = 0.0):
        """
        Initialize the OptionPricer.
        
        Args:
            risk_free_rate: Risk-free interest rate (annualized)
            dividend_yield: Dividend yield (annualized)
        """
        self.risk_free_rate = risk_free_rate
        self.dividend_yield = dividend_yield
    
    def black_scholes_price(
        self,
        option_type: str,
        underlying_price: float,
        strike_price: float,
        time_to_expiry: float,
        volatility: float
    ) -> float:
        """
        Calculate option price using the Black-Scholes model.
        
        Args:
            option_type: 'call' or 'put'
            underlying_price: Current price of the underlying asset
            strike_price: Strike price of the option
            time_to_expiry: Time to expiration in years
            volatility: Implied volatility
            
        Returns:
            Option price
        """
        if option_type.lower() not in ['call', 'put']:
            raise ValueError("Option type must be 'call' or 'put'")
        
        if time_to_expiry <= 0 or volatility <= 0:
            return max(0, underlying_price - strike_price) if option_type.lower() == 'call' else max(0, strike_price - underlying_price)
        
        d1 = (np.log(underlying_price / strike_price) + 
              (self.risk_free_rate - self.dividend_yield + 0.5 * volatility**2) * time_to_expiry) / (volatility * np.sqrt(time_to_expiry))
        d2 = d1 - volatility * np.sqrt(time_to_expiry)
        
        if option_type.lower() == 'call':
            price = (underlying_price * np.exp(-self.dividend_yield * time_to_expiry) * norm.cdf(d1) - 
                     strike_price * np.exp(-self.risk_free_rate * time_to_expiry) * norm.cdf(d2))
        else:  # put
            price = (strike_price * np.exp(-self.risk_free_rate * time_to_expiry) * norm.cdf(-d2) - 
                     underlying_price * np.exp(-self.dividend_yield * time_to_expiry) * norm.cdf(-d1))
        
        return price
    
    def implied_volatility(
        self,
        option_price: float,
        underlying_price: float,
        strike_price: float,
        time_to_expiry: float,
        initial_guess: float = 0.2,
        max_iterations: int = 100,
        precision: float = 1e-6
    ) -> float:
        """
        Calculate implied volatility using the Newton-Raphson method.
        
        Args:
            option_price: Market price of the option
            underlying_price: Current price of the underlying asset
            strike_price: Strike price of the option
            time_to_expiry: Time to expiration in years
            initial_guess: Initial guess for implied volatility
            max_iterations: Maximum number of iterations
            precision: Desired precision
            
        Returns:
            Implied volatility
        """
        if time_to_expiry <= 0:
            return 0.0
        
        # Determine option type based on moneyness
        option_type = 'call' if underlying_price >= strike_price else 'put'
        
        # Initial guess
        vol = initial_guess
        
        for i in range(max_iterations):
            # Calculate option price and vega
            price = self.black_scholes_price(option_type, underlying_price, strike_price, time_to_expiry, vol)
            vega = self.calculate_vega(underlying_price, strike_price, time_to_expiry, vol)
            
            # Calculate difference
            diff = option_price - price
            
            # Check if precision is reached
            if abs(diff) < precision:
                return vol
            
            # Update volatility estimate
            if abs(vega) < 1e-10:  # Avoid division by zero
                break
                
            vol = vol + diff / vega
            
            # Ensure volatility is within reasonable bounds
            if vol <= 0:
                vol = 0.001
            elif vol > 5:
                vol = 5.0
        
        logger.warning(f"Implied volatility calculation did not converge after {max_iterations} iterations")
        return vol
    
    def calculate_delta(
        self,
        option_type: str,
        underlying_price: float,
        strike_price: float,
        time_to_expiry: float,
        volatility: float
    ) -> float:
        """
        Calculate the delta of an option.
        
        Args:
            option_type: 'call' or 'put'
            underlying_price: Current price of the underlying asset
            strike_price: Strike price of the option
            time_to_expiry: Time to expiration in years
            volatility: Implied volatility
            
        Returns:
            Option delta
        """
        if time_to_expiry <= 0 or volatility <= 0:
            if option_type.lower() == 'call':
                return 1.0 if underlying_price > strike_price else 0.0
            else:  # put
                return -1.0 if underlying_price < strike_price else 0.0
        
        d1 = (np.log(underlying_price / strike_price) + 
              (self.risk_free_rate - self.dividend_yield + 0.5 * volatility**2) * time_to_expiry) / (volatility * np.sqrt(time_to_expiry))
        
        if option_type.lower() == 'call':
            delta = np.exp(-self.dividend_yield * time_to_expiry) * norm.cdf(d1)
        else:  # put
            delta = np.exp(-self.dividend_yield * time_to_expiry) * (norm.cdf(d1) - 1)
        
        return delta
    
    def calculate_gamma(
        self,
        underlying_price: float,
        strike_price: float,
        time_to_expiry: float,
        volatility: float
    ) -> float:
        """
        Calculate the gamma of an option.
        
        Args:
            underlying_price: Current price of the underlying asset
            strike_price: Strike price of the option
            time_to_expiry: Time to expiration in years
            volatility: Implied volatility
            
        Returns:
            Option gamma
        """
        if time_to_expiry <= 0 or volatility <= 0:
            return 0.0
        
        d1 = (np.log(underlying_price / strike_price) + 
              (self.risk_free_rate - self.dividend_yield + 0.5 * volatility**2) * time_to_expiry) / (volatility * np.sqrt(time_to_expiry))
        
        gamma = np.exp(-self.dividend_yield * time_to_expiry) * norm.pdf(d1) / (underlying_price * volatility * np.sqrt(time_to_expiry))
        
        return gamma
    
    def calculate_vega(
        self,
        underlying_price: float,
        strike_price: float,
        time_to_expiry: float,
        volatility: float
    ) -> float:
        """
        Calculate the vega of an option.
        
        Args:
            underlying_price: Current price of the underlying asset
            strike_price: Strike price of the option
            time_to_expiry: Time to expiration in years
            volatility: Implied volatility
            
        Returns:
            Option vega
        """
        if time_to_expiry <= 0 or volatility <= 0:
            return 0.0
        
        d1 = (np.log(underlying_price / strike_price) + 
              (self.risk_free_rate - self.dividend_yield + 0.5 * volatility**2) * time_to_expiry) / (volatility * np.sqrt(time_to_expiry))
        
        vega = underlying_price * np.exp(-self.dividend_yield * time_to_expiry) * norm.pdf(d1) * np.sqrt(time_to_expiry) / 100
        
        return vega
    
    def calculate_theta(
        self,
        option_type: str,
        underlying_price: float,
        strike_price: float,
        time_to_expiry: float,
        volatility: float
    ) -> float:
        """
        Calculate the theta of an option.
        
        Args:
            option_type: 'call' or 'put'
            underlying_price: Current price of the underlying asset
            strike_price: Strike price of the option
            time_to_expiry: Time to expiration in years
            volatility: Implied volatility
            
        Returns:
            Option theta
        """
        if time_to_expiry <= 0 or volatility <= 0:
            return 0.0
        
        d1 = (np.log(underlying_price / strike_price) + 
              (self.risk_free_rate - self.dividend_yield + 0.5 * volatility**2) * time_to_expiry) / (volatility * np.sqrt(time_to_expiry))
        d2 = d1 - volatility * np.sqrt(time_to_expiry)
        
        term1 = -underlying_price * np.exp(-self.dividend_yield * time_to_expiry) * norm.pdf(d1) * volatility / (2 * np.sqrt(time_to_expiry))
        
        if option_type.lower() == 'call':
            term2 = -self.risk_free_rate * strike_price * np.exp(-self.risk_free_rate * time_to_expiry) * norm.cdf(d2)
            term3 = self.dividend_yield * underlying_price * np.exp(-self.dividend_yield * time_to_expiry) * norm.cdf(d1)
        else:  # put
            term2 = self.risk_free_rate * strike_price * np.exp(-self.risk_free_rate * time_to_expiry) * norm.cdf(-d2)
            term3 = -self.dividend_yield * underlying_price * np.exp(-self.dividend_yield * time_to_expiry) * norm.cdf(-d1)
        
        theta = (term1 + term2 + term3) / 365.0  # Daily theta
        
        return theta
    
    def calculate_greeks(
        self,
        option_type: str,
        underlying_price: float,
        strike_price: float,
        time_to_expiry: float,
        volatility: float
    ) -> Dict[str, float]:
        """
        Calculate all Greeks for an option.
        
        Args:
            option_type: 'call' or 'put'
            underlying_price: Current price of the underlying asset
            strike_price: Strike price of the option
            time_to_expiry: Time to expiration in years
            volatility: Implied volatility
            
        Returns:
            Dictionary containing all Greeks
        """
        delta = self.calculate_delta(option_type, underlying_price, strike_price, time_to_expiry, volatility)
        gamma = self.calculate_gamma(underlying_price, strike_price, time_to_expiry, volatility)
        vega = self.calculate_vega(underlying_price, strike_price, time_to_expiry, volatility)
        theta = self.calculate_theta(option_type, underlying_price, strike_price, time_to_expiry, volatility)
        
        return {
            'delta': delta,
            'gamma': gamma,
            'vega': vega,
            'theta': theta
        }
