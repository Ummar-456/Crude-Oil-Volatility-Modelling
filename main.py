# src/main.py

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import logging
import argparse
from typing import Dict, Optional

# Import project modules
from utils.data_processor import DataProcessor
from utils.option_pricer import OptionPricer
from models.volatility_surface import VolatilitySurface
from models.sabr_model import SABRModel
from models.local_volatility import LocalVolatilityModel
from models.stochastic_local_volatility import StochasticLocalVolatilityModel
from visualization.surface_visualizer import VolatilitySurfaceVisualizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("volatility_surface.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class CommodityVolatilityApp:
    """
    Main application class for commodity volatility surface modeling.
    """
    
    def __init__(
        self,
        commodity_ticker: str = "CL=F",
        data_lookback_days: int = 365,
        output_dir: str = "results",
        risk_free_rate: float = 0.05,
        dividend_yield: float = 0.0
    ):
        """
        Initialize the application.
        
        Args:
            commodity_ticker: Ticker symbol for the commodity
            data_lookback_days: Number of days of historical data to fetch
            output_dir: Directory to save results
            risk_free_rate: Risk-free interest rate (annualized)
            dividend_yield: Dividend yield (annualized)
        """
        self.commodity_ticker = commodity_ticker
        self.data_lookback_days = data_lookback_days
        self.output_dir = output_dir
        self.risk_free_rate = risk_free_rate
        self.dividend_yield = dividend_yield
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            logger.info(f"Created output directory: {output_dir}")
        
        # Initialize components
        self.data_processor = DataProcessor()
        self.option_pricer = OptionPricer(risk_free_rate=risk_free_rate, dividend_yield=dividend_yield)
        self.visualizer = VolatilitySurfaceVisualizer(output_dir=os.path.join(output_dir, "figures"))
        
        # Data containers
        self.historical_data = None
        self.option_data = None
        self.processed_options = None
        self.surface_data = None
        self.models = {}
    
    def run(self):
        """
        Run the complete workflow.
        """
        logger.info(f"Starting commodity volatility surface modeling for {self.commodity_ticker}")
        
        # Fetch and process data
        self._fetch_and_process_data()
        
        # Calibrate models
        self._calibrate_models()
        
        # Generate visualizations
        self._generate_visualizations()
        
        logger.info("Commodity volatility surface modeling completed successfully")
    
    def _fetch_and_process_data(self):
        """
        Fetch and process historical and option data.
        """
        # Calculate date range
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=self.data_lookback_days)).strftime('%Y-%m-%d')
        
        logger.info(f"Fetching historical data for {self.commodity_ticker} from {start_date} to {end_date}")
        
        # Fetch and process historical data
        self.historical_data = self.data_processor.fetch_data(self.commodity_ticker, start_date, end_date)
        self.historical_data = self.data_processor.clean_data(self.historical_data)
        self.historical_data = self.data_processor.calculate_returns(self.historical_data)
        self.historical_data = self.data_processor.calculate_historical_volatility(self.historical_data)
        
        # Get current price
        current_price = self.historical_data['Close'].iloc[-1]
        if isinstance(current_price, pd.Series):
            current_price = current_price.item()
        logger.info(f"Current price of {self.commodity_ticker}: {current_price:.2f}")
        
        # Fetch and process option data
        logger.info(f"Fetching option data for {self.commodity_ticker}")
        self.option_data = self.data_processor.fetch_option_data(self.commodity_ticker)
        
        if self.option_data.empty:
            logger.warning(f"No option data available for {self.commodity_ticker}. Using synthetic data.")
            self._generate_synthetic_option_data(current_price)
        else:
            self.processed_options = self.data_processor.process_option_data(self.option_data)
            self.processed_options = self.data_processor.filter_options(self.processed_options)
            self.processed_options = self.data_processor.calculate_implied_volatility(
                self.processed_options, self.option_pricer
            )
        
        # Prepare surface data
        self.surface_data = self.data_processor.prepare_surface_data(self.processed_options)
        
        logger.info("Data processing completed")
    
    def _generate_synthetic_option_data(self, current_price: float):
        """
        Generate synthetic option data when real data is not available.
        
        Args:
            current_price: Current price of the underlying asset
        """
        logger.info("Generating synthetic option data")
        
        # Create synthetic option data
        strikes = np.linspace(0.7 * current_price, 1.3 * current_price, 15)
        expiries = np.array([0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0])
        
        option_rows = []
        for expiry in expiries:
            for strike in strikes:
                # Calculate synthetic implied volatility with smile
                moneyness = strike / current_price
                implied_vol = 0.2 + 0.1 * (moneyness - 1)**2 + 0.05 * np.sqrt(expiry)
                
                # Calculate option price
                call_price = self.option_pricer.black_scholes_price(
                    'call', current_price, strike, expiry, implied_vol
                )
                put_price = self.option_pricer.black_scholes_price(
                    'put', current_price, strike, expiry, implied_vol
                )
                
                # Add call option
                option_rows.append({
                    'strike': strike,
                    'expiry': pd.Timestamp.now() + pd.Timedelta(days=int(expiry * 365)),
                    'time_to_expiry': expiry,
                    'option_type': 'call',
                    'lastPrice': call_price,
                    'underlying_price': current_price,
                    'impliedVolatility': implied_vol,
                    'implied_vol': implied_vol
                })
                
                # Add put option
                option_rows.append({
                    'strike': strike,
                    'expiry': pd.Timestamp.now() + pd.Timedelta(days=int(expiry * 365)),
                    'time_to_expiry': expiry,
                    'option_type': 'put',
                    'lastPrice': put_price,
                    'underlying_price': current_price,
                    'impliedVolatility': implied_vol,
                    'implied_vol': implied_vol
                })
        
        self.processed_options = pd.DataFrame(option_rows)
        logger.info(f"Generated {len(self.processed_options)} synthetic option data points")
    
    def _calibrate_models(self):
        """
        Calibrate various volatility models.
        """
        logger.info("Calibrating volatility models")
        
        current_price = self.historical_data['Close'].iloc[-1]
        
        # Calibrate SABR model
        logger.info("Calibrating SABR model")
        sabr_model = SABRModel()
        sabr_model.calibrate(self.processed_options, current_price)
        self.models['sabr'] = sabr_model
        
        # Calibrate local volatility model
        logger.info("Calibrating local volatility model")
        local_vol_model = LocalVolatilityModel()
        local_vol_model.calibrate(self.processed_options, current_price, self.risk_free_rate, self.dividend_yield)
        self.models['local_vol'] = local_vol_model
        
        # Calibrate stochastic-local volatility model
        logger.info("Calibrating stochastic-local volatility model")
        slv_model = StochasticLocalVolatilityModel()
        slv_model.calibrate(self.processed_options, current_price, self.risk_free_rate, self.dividend_yield)
        self.models['slv'] = slv_model
        
        logger.info("Model calibration completed")
    
    def _generate_visualizations(self):
        """
        Generate visualizations for the calibrated models.
        """
        logger.info("Generating visualizations")
        
        current_price = self.historical_data['Close'].iloc[-1]
        
        # Create dashboard for each model
        for model_name, model in self.models.items():
            logger.info(f"Generating visualizations for {model_name} model")
            
            # Get model-specific surface data
            if model_name == 'sabr':
                strikes = np.linspace(0.7 * current_price, 1.3 * current_price, 50)
                expiries = np.linspace(0.1, 2.0, 20)
                surface_data = model.get_volatility_surface(strikes, expiries)
            else:
                surface_data = model.plot_local_volatility_surface(show=False).canvas.tostring_rgb()
            
            # Create dashboard
            self.visualizer.create_dashboard(
                surface_data,
                self.historical_data['hist_volatility'].dropna(),
                self.processed_options,
                model.calibrated_params if hasattr(model, 'calibrated_params') else None,
                current_price,
                self.commodity_ticker,
                f"{model_name}_dashboard.pdf"
            )
        
        logger.info("Visualization generation completed")

def parse_arguments():
    """
    Parse command line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description='Commodity Volatility Surface Modeling')
    
    parser.add_argument('--ticker', type=str, default='CL=F',
                        help='Ticker symbol for the commodity (default: CL=F)')
    
    parser.add_argument('--lookback', type=int, default=365,
                        help='Number of days of historical data to fetch (default: 365)')
    
    parser.add_argument('--output-dir', type=str, default='results',
                        help='Directory to save results (default: results)')
    
    parser.add_argument('--risk-free-rate', type=float, default=0.05,
                        help='Risk-free interest rate (default: 0.05)')
    
    parser.add_argument('--dividend-yield', type=float, default=0.0,
                        help='Dividend yield (default: 0.0)')
    
    return parser.parse_args()

if __name__ == "__main__":
    # Parse command line arguments
    args = parse_arguments()
    
    # Create and run the application
    app = CommodityVolatilityApp(
        commodity_ticker=args.ticker,
        data_lookback_days=args.lookback,
        output_dir=args.output_dir,
        risk_free_rate=args.risk_free_rate,
        dividend_yield=args.dividend_yield
    )
    
    app.run()
