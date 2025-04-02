# src/data/commodity_data.py

import pandas as pd
import numpy as np
import yfinance as yf
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Dictionary mapping commodity names to their Yahoo Finance ticker symbols
COMMODITY_TICKERS = {
    'gold': 'GC=F',
    'silver': 'SI=F',
    'crude_oil': 'CL=F',
    'natural_gas': 'NG=F',
    'copper': 'HG=F',
    'corn': 'ZC=F',
    'wheat': 'ZW=F',
    'soybeans': 'ZS=F',
    'cotton': 'CT=F',
    'coffee': 'KC=F',
    'sugar': 'SB=F',
    'platinum': 'PL=F',
    'palladium': 'PA=F'
}

class CommodityDataManager:
    """
    A class to manage the retrieval, storage, and processing of commodity futures data
    for volatility surface modeling.
    """
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize the CommodityDataManager.
        
        Args:
            data_dir: Directory to store downloaded data
        """
        self.data_dir = data_dir
        self.raw_dir = os.path.join(data_dir, "raw")
        self.processed_dir = os.path.join(data_dir, "processed")
        
        # Create directories if they don't exist
        for directory in [self.data_dir, self.raw_dir, self.processed_dir]:
            if not os.path.exists(directory):
                os.makedirs(directory)
                logger.info(f"Created directory: {directory}")
    
    def download_historical_data(
        self, 
        commodity: str, 
        start_date: str = None, 
        end_date: str = None,
        period: str = "5y",
        force_download: bool = False
    ) -> pd.DataFrame:
        """
        Download historical futures data for a specified commodity.
        
        Args:
            commodity: Name of the commodity (must be in COMMODITY_TICKERS)
            start_date: Start date in 'YYYY-MM-DD' format (overrides period if provided)
            end_date: End date in 'YYYY-MM-DD' format (defaults to today)
            period: Time period to download (e.g., '1d', '1mo', '1y', '5y', 'max')
            force_download: If True, download even if local file exists
            
        Returns:
            DataFrame containing historical price data
        """
        # Validate commodity name
        if commodity not in COMMODITY_TICKERS:
            raise ValueError(f"Unknown commodity: {commodity}. Available options: {list(COMMODITY_TICKERS.keys())}")
        
        ticker = COMMODITY_TICKERS[commodity]
        
        # Check if data already exists locally
        filename = f"{commodity}_historical.parquet"
        filepath = os.path.join(self.raw_dir, filename)
        
        if os.path.exists(filepath) and not force_download:
            logger.info(f"Loading {commodity} data from local file: {filepath}")
            return pd.read_parquet(filepath)
        
        # Set up date parameters
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        try:
            # Download data
            if start_date:
                logger.info(f"Downloading {commodity} data from {start_date} to {end_date}")
                data = yf.download(ticker, start=start_date, end=end_date)
            else:
                logger.info(f"Downloading {commodity} data for period: {period}")
                data = yf.download(ticker, period=period)
            
            if data.empty:
                logger.warning(f"No data found for {commodity} ({ticker})")
                return pd.DataFrame()
            
            # Add metadata columns
            data['commodity'] = commodity
            data['ticker'] = ticker
            
            # Save to parquet file
            data.to_parquet(filepath)
            logger.info(f"Saved {commodity} data to {filepath}")
            
            return data
            
        except Exception as e:
            logger.error(f"Error downloading {commodity} data: {e}")
            raise
    
    def download_options_data(self, commodity: str) -> Dict[str, pd.DataFrame]:
        """
        Download options chain data for a specified commodity.
        
        Args:
            commodity: Name of the commodity (must be in COMMODITY_TICKERS)
            
        Returns:
            Dictionary with 'calls' and 'puts' DataFrames for each expiration date
        """
        if commodity not in COMMODITY_TICKERS:
            raise ValueError(f"Unknown commodity: {commodity}. Available options: {list(COMMODITY_TICKERS.keys())}")
        
        ticker = COMMODITY_TICKERS[commodity]
        
        try:
            # Initialize the ticker object
            ticker_obj = yf.Ticker(ticker)
            
            # Get available expiration dates
            expirations = ticker_obj.options
            
            if not expirations:
                logger.warning(f"No options data available for {commodity} ({ticker})")
                return {}
            
            logger.info(f"Found {len(expirations)} expiration dates for {commodity}")
            
            # Fetch options data for each expiration
            options_data = {}
            for date in expirations:
                try:
                    # Get the options chain for this expiration date
                    opt = ticker_obj.option_chain(date)
                    
                    # Store calls and puts
                    calls_df = opt.calls
                    puts_df = opt.puts
                    
                    # Add metadata
                    for df in [calls_df, puts_df]:
                        df['commodity'] = commodity
                        df['ticker'] = ticker
                        df['expiration'] = date
                    
                    # Save to files
                    calls_file = os.path.join(self.raw_dir, f"{commodity}_calls_{date}.parquet")
                    puts_file = os.path.join(self.raw_dir, f"{commodity}_puts_{date}.parquet")
                    
                    calls_df.to_parquet(calls_file)
                    puts_df.to_parquet(puts_file)
                    
                    # Store in dictionary
                    options_data[date] = {
                        'calls': calls_df,
                        'puts': puts_df
                    }
                    
                    logger.info(f"Saved options data for {commodity}, expiration {date}")
                    
                except Exception as e:
                    logger.error(f"Error fetching options for {date}: {e}")
            
            return options_data
            
        except Exception as e:
            logger.error(f"Error downloading options data for {commodity}: {e}")
            raise
    
    def calculate_returns(self, data: pd.DataFrame, method: str = 'log') -> pd.DataFrame:
        """
        Calculate returns from price data.
        
        Args:
            data: DataFrame with price data (must have 'Close' column)
            method: 'simple' for percentage returns or 'log' for log returns
            
        Returns:
            DataFrame with returns added
        """
        if 'Close' not in data.columns:
            raise ValueError("DataFrame must contain 'Close' column")
        
        df = data.copy()
        
        if method == 'simple':
            df['returns'] = df['Close'].pct_change()
        elif method == 'log':
            df['returns'] = np.log(df['Close'] / df['Close'].shift(1))
        else:
            raise ValueError("Method must be 'simple' or 'log'")
        
        return df
    
    def calculate_realized_volatility(
        self, 
        data: pd.DataFrame, 
        window: int = 30, 
        annualize: bool = True
    ) -> pd.DataFrame:
        """
        Calculate realized volatility from returns data.
        
        Args:
            data: DataFrame with returns data (must have 'returns' column)
            window: Rolling window size in days
            annualize: Whether to annualize the volatility
            
        Returns:
            DataFrame with realized volatility added
        """
        if 'returns' not in data.columns:
            raise ValueError("DataFrame must contain 'returns' column")
        
        df = data.copy()
        
        # Calculate rolling standard deviation
        df['realized_vol'] = df['returns'].rolling(window=window).std()
        
        # Annualize if requested (assuming 252 trading days per year)
        if annualize:
            df['realized_vol'] = df['realized_vol'] * np.sqrt(252)
        
        return df
    
    def get_processed_data(
        self, 
        commodity: str, 
        start_date: str = None, 
        end_date: str = None,
        period: str = "5y",
        force_download: bool = False
    ) -> pd.DataFrame:
        """
        Get fully processed data for a commodity, including returns and volatility.
        
        Args:
            commodity: Name of the commodity
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            period: Time period to download
            force_download: If True, download even if local file exists
            
        Returns:
            Processed DataFrame with price, returns, and volatility data
        """
        # Get raw data
        raw_data = self.download_historical_data(
            commodity, 
            start_date, 
            end_date, 
            period, 
            force_download
        )
        
        if raw_data.empty:
            return pd.DataFrame()
        
        # Calculate returns
        data_with_returns = self.calculate_returns(raw_data)
        
        # Calculate volatility
        processed_data = self.calculate_realized_volatility(data_with_returns)
        
        # Save processed data
        filename = f"{commodity}_processed.parquet"
        filepath = os.path.join(self.processed_dir, filename)
        processed_data.to_parquet(filepath)
        logger.info(f"Saved processed {commodity} data to {filepath}")
        
        return processed_data

