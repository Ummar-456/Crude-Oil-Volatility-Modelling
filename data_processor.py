# src/utils/data_processor.py

import pandas as pd
import numpy as np
import yfinance as yf
from typing import Dict, List, Optional, Union
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataProcessor:
    """
    A class for processing and cleaning commodity data from yfinance for volatility surface modeling.
    """

    def __init__(self):
        """Initialize the DataProcessor."""
        pass

    @staticmethod
    def fetch_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch data from yfinance.

        Args:
            ticker (str): Ticker symbol for the commodity.
            start_date (str): Start date in 'YYYY-MM-DD' format.
            end_date (str): End date in 'YYYY-MM-DD' format.

        Returns:
            pd.DataFrame: Fetched data.
        """
        try:
            df = yf.download(ticker, start=start_date, end=end_date)
            logger.info(f"Data fetched successfully for {ticker} from {start_date} to {end_date}")
            return df
        except Exception as e:
            logger.error(f"Error fetching data for {ticker}: {str(e)}")
            raise

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the input dataframe by removing duplicates and handling missing values.

        Args:
            df (pd.DataFrame): Input dataframe.

        Returns:
            pd.DataFrame: Cleaned dataframe.
        """
        logger.info("Starting data cleaning process")
        
        # Remove duplicates
        df_cleaned = df.drop_duplicates()
        logger.info(f"Removed {len(df) - len(df_cleaned)} duplicate rows")

        # Handle missing values
        df_cleaned = df_cleaned.dropna()
        logger.info(f"Removed rows with missing values. Remaining rows: {len(df_cleaned)}")

        return df_cleaned

    def calculate_returns(self, df: pd.DataFrame, price_column: str = 'Close') -> pd.DataFrame:
        """
        Calculate daily returns based on closing prices.

        Args:
            df (pd.DataFrame): Input dataframe.
            price_column (str): Name of the column containing price data.

        Returns:
            pd.DataFrame: Dataframe with an additional 'returns' column.
        """
        df['returns'] = df[price_column].pct_change()
        logger.info("Daily returns calculated")
        return df

    def calculate_historical_volatility(self, df: pd.DataFrame, window: int = 30) -> pd.DataFrame:
        """
        Calculate historical volatility using a rolling window.

        Args:
            df (pd.DataFrame): Input dataframe with a 'returns' column.
            window (int): Rolling window size for volatility calculation.

        Returns:
            pd.DataFrame: Dataframe with an additional 'hist_volatility' column.
        """
        df['hist_volatility'] = df['returns'].rolling(window=window).std() * np.sqrt(252)
        logger.info(f"Historical volatility calculated using a {window}-day window")
        return df
    
    
    def fetch_option_data(self, ticker: str) -> pd.DataFrame:
        """
    Fetch option data for a given ticker from yfinance.

    Args:
        ticker (str): Ticker symbol for the commodity.

    Returns:
        pd.DataFrame: Processed option data or empty DataFrame if no options available.
    """
        logger.info(f"Fetching option data for {ticker}")
    
        try:
            ticker_obj = yf.Ticker(ticker)
            all_options = []
        
        # Check if options are available
            if not ticker_obj.options:
                logger.warning(f"No options available for {ticker}")
            return pd.DataFrame()  # Return empty DataFrame
        
            for expiration in ticker_obj.options:
                options = ticker_obj.option_chain(expiration)
                calls = options.calls
                puts = options.puts
            
                calls['option_type'] = 'call'
                puts['option_type'] = 'put'
            
                all_options.append(calls)
                all_options.append(puts)
        
        # Check if all_options is empty before concatenating
            if not all_options:
                logger.warning(f"No option data found for {ticker}")
            return pd.DataFrame()
            
            df = pd.concat(all_options, ignore_index=True)
            df['expiry'] = pd.to_datetime(df['expiration'])
            df['underlying_price'] = ticker_obj.history(period="1d")['Close'].iloc[0]
        
            logger.info(f"Option data fetched successfully for {ticker}")
            return df
        except Exception as e:
            logger.error(f"Error fetching option data for {ticker}: {str(e)}")
        return pd.DataFrame()  # Return empty DataFrame instead of raising exception

    def process_option_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process option data for volatility surface modeling.

        Args:
            df (pd.DataFrame): Input dataframe with option data.

        Returns:
            pd.DataFrame: Processed option data.
        """
        logger.info("Processing option data")

        # Calculate time to expiry in years
        current_date = pd.Timestamp.now()
        df['time_to_expiry'] = (df['expiry'] - current_date).dt.days / 365.25

        # Calculate moneyness
        df['moneyness'] = df['underlying_price'] / df['strike']

        logger.info("Option data processed successfully")
        return df

    def filter_options(self, df: pd.DataFrame, min_days: int = 7, max_days: int = 365) -> pd.DataFrame:
        """
        Filter options based on time to expiry.

        Args:
            df (pd.DataFrame): Input dataframe with option data.
            min_days (int): Minimum number of days to expiry.
            max_days (int): Maximum number of days to expiry.

        Returns:
            pd.DataFrame: Filtered option data.
        """
        df_filtered = df[(df['time_to_expiry'] * 365.25 >= min_days) & (df['time_to_expiry'] * 365.25 <= max_days)]
        logger.info(f"Filtered options: {len(df_filtered)} out of {len(df)} remain")
        return df_filtered

    def calculate_implied_volatility(self, df: pd.DataFrame, pricing_model: callable) -> pd.DataFrame:
        """
        Calculate implied volatility for options.

        Args:
            df (pd.DataFrame): Input dataframe with option data.
            pricing_model (callable): Option pricing model function.

        Returns:
            pd.DataFrame: Dataframe with an additional 'implied_vol' column.
        """
        def implied_vol(row):
            try:
                return pricing_model.implied_volatility(
                    row['lastPrice'], row['underlying_price'], row['strike'],
                    row['time_to_expiry'], row['impliedVolatility']  # Using impliedVolatility from yfinance as initial guess
                )
            except:
                return np.nan

        df['implied_vol'] = df.apply(implied_vol, axis=1)
        df = df.dropna(subset=['implied_vol'])
        logger.info(f"Implied volatility calculated for {len(df)} options")
        return df

    def prepare_surface_data(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Prepare data for volatility surface modeling.

        Args:
            df (pd.DataFrame): Input dataframe with processed option data.

        Returns:
            Dict[str, np.ndarray]: Dictionary containing arrays for strikes, expiries, and implied volatilities.
        """
        strikes = df['strike'].unique()
        expiries = df['time_to_expiry'].unique()
        
        X, Y = np.meshgrid(strikes, expiries)
        Z = np.zeros_like(X)

        for i, expiry in enumerate(expiries):
            for j, strike in enumerate(strikes):
                matching_options = df[(df['time_to_expiry'] == expiry) & (df['strike'] == strike)]
                if not matching_options.empty:
                    Z[i, j] = matching_options['implied_vol'].mean()

        logger.info("Surface data prepared successfully")
        return {'strikes': X, 'expiries': Y, 'implied_vols': Z}
