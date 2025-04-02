# src/visualization/surface_visualizer.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import logging
from typing import Dict, List, Optional, Tuple, Union
import os
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class VolatilitySurfaceVisualizer:
    """
    A class for visualizing volatility surfaces and related data for commodity options.
    
    This class provides various visualization methods for:
    - 3D volatility surfaces
    - 2D volatility slices (smiles and term structures)
    - Comparison of model vs. market data
    - Time series of volatility surface parameters
    """
    
    def __init__(self, output_dir: str = "results/figures"):
        """
        Initialize the VolatilitySurfaceVisualizer.
        
        Args:
            output_dir: Directory to save generated figures
        """
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            logger.info(f"Created output directory: {output_dir}")
        
        # Set default plot style
        plt.style.use('seaborn-v0_8-whitegrid')
        
        # Default color maps
        self.surface_cmap = cm.viridis
        self.smile_colors = plt.cm.tab10
    
    def plot_3d_surface(
        self,
        surface_data: Dict,
        title: str = "Commodity Volatility Surface",
        show_market_points: bool = True,
        azimuth: float = -60,
        elevation: float = 30,
        filename: Optional[str] = None,
        show: bool = True
    ) -> plt.Figure:
        """
        Create a 3D visualization of a volatility surface.
        
        Args:
            surface_data: Dictionary containing surface grid and labels
            title: Plot title
            show_market_points: Whether to show market data points
            azimuth: Horizontal viewing angle
            elevation: Vertical viewing angle
            filename: If provided, save figure to this filename
            show: Whether to display the figure
            
        Returns:
            Matplotlib figure object
        """
        # Extract data
        X = surface_data['x_grid']
        Y = surface_data['y_grid']
        Z = surface_data['z_grid']
        
        x_label = surface_data.get('x_label', 'Log Moneyness')
        y_label = surface_data.get('y_label', 'Time to Expiry')
        z_label = surface_data.get('z_label', 'Implied Volatility')
        
        # Create figure
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot the surface
        surf = ax.plot_surface(
            X, Y, Z, 
            cmap=self.surface_cmap,
            linewidth=0, 
            antialiased=True,
            alpha=0.8
        )
        
        # Add market data points if requested
        if show_market_points and 'x_raw' in surface_data:
            x_raw = surface_data['x_raw']
            y_raw = surface_data['y_raw']
            z_raw = surface_data['z_raw']
            
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
        ax.set_title(title)
        
        # Add colorbar
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
        
        # Set view angle
        ax.view_init(elevation, azimuth)
        
        # Save figure if filename is provided
        if filename:
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            logger.info(f"Saved figure to {filepath}")
        
        # Show or close figure
        if show:
            plt.tight_layout()
        else:
            plt.close(fig)
        
        return fig
    
    def plot_volatility_smile(
        self,
        surface_data: Dict,
        expiries: List[float] = None,
        title: str = "Volatility Smile at Different Expiries",
        filename: Optional[str] = None,
        show: bool = True
    ) -> plt.Figure:
        """
        Plot volatility smile (implied volatility vs. log moneyness) for different expiries.
        
        Args:
            surface_data: Dictionary containing surface grid and labels
            expiries: List of expiry times to plot (if None, selects a few representative ones)
            title: Plot title
            filename: If provided, save figure to this filename
            show: Whether to display the figure
            
        Returns:
            Matplotlib figure object
        """
        # Extract data
        X = surface_data['x_grid']
        Y = surface_data['y_grid']
        Z = surface_data['z_grid']
        
        x_label = surface_data.get('x_label', 'Log Moneyness')
        y_label = surface_data.get('y_label', 'Time to Expiry')
        z_label = surface_data.get('z_label', 'Implied Volatility')
        
        # If expiries not provided, select a few representative ones
        if expiries is None:
            y_unique = np.unique(Y[:, 0])
            num_expiries = min(5, len(y_unique))
            indices = np.linspace(0, len(y_unique) - 1, num_expiries, dtype=int)
            expiries = [y_unique[i] for i in indices]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot volatility smile for each expiry
        for i, expiry in enumerate(expiries):
            # Find closest row in the grid
            row_idx = np.abs(Y[:, 0] - expiry).argmin()
            
            # Extract data for this slice
            x_slice = X[row_idx, :]
            z_slice = Z[row_idx, :]
            
            # Sort by x value
            sort_idx = np.argsort(x_slice)
            x_slice = x_slice[sort_idx]
            z_slice = z_slice[sort_idx]
            
            # Plot this slice
            color = self.smile_colors(i % 10)
            ax.plot(x_slice, z_slice, '-', linewidth=2, color=color, label=f'T = {expiry:.2f}')
            
            # Add market points if available
            if 'x_raw' in surface_data and 'y_raw' in surface_data and 'z_raw' in surface_data:
                x_raw = surface_data['x_raw']
                y_raw = surface_data['y_raw']
                z_raw = surface_data['z_raw']
                
                # Filter points near this expiry
                mask = np.abs(y_raw - expiry) < 0.01
                if np.any(mask):
                    ax.scatter(x_raw[mask], z_raw[mask], color=color, marker='o', s=30, alpha=0.7)
        
        # Set labels and title
        ax.set_xlabel(x_label)
        ax.set_ylabel(z_label)
        ax.set_title(title)
        
        # Add legend and grid
        ax.legend()
        ax.grid(True)
        
        # Save figure if filename is provided
        if filename:
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            logger.info(f"Saved figure to {filepath}")
        
        # Show or close figure
        if show:
            plt.tight_layout()
        else:
            plt.close(fig)
        
        return fig
    
    def plot_term_structure(
        self,
        surface_data: Dict,
        moneyness_levels: List[float] = None,
        title: str = "Volatility Term Structure at Different Moneyness Levels",
        filename: Optional[str] = None,
        show: bool = True
    ) -> plt.Figure:
        """
        Plot volatility term structure (implied volatility vs. time to expiry) for different moneyness levels.
        
        Args:
            surface_data: Dictionary containing surface grid and labels
            moneyness_levels: List of log moneyness levels to plot (if None, selects a few representative ones)
            title: Plot title
            filename: If provided, save figure to this filename
            show: Whether to display the figure
            
        Returns:
            Matplotlib figure object
        """
        # Extract data
        X = surface_data['x_grid']
        Y = surface_data['y_grid']
        Z = surface_data['z_grid']
        
        x_label = surface_data.get('x_label', 'Log Moneyness')
        y_label = surface_data.get('y_label', 'Time to Expiry')
        z_label = surface_data.get('z_label', 'Implied Volatility')
        
        # If moneyness levels not provided, select a few representative ones
        if moneyness_levels is None:
            x_unique = np.unique(X[0, :])
            num_levels = min(5, len(x_unique))
            indices = np.linspace(0, len(x_unique) - 1, num_levels, dtype=int)
            moneyness_levels = [x_unique[i] for i in indices]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot term structure for each moneyness level
        for i, moneyness in enumerate(moneyness_levels):
            # Find closest column in the grid
            col_idx = np.abs(X[0, :] - moneyness).argmin()
            
            # Extract data for this slice
            y_slice = Y[:, col_idx]
            z_slice = Z[:, col_idx]
            
            # Sort by y value
            sort_idx = np.argsort(y_slice)
            y_slice = y_slice[sort_idx]
            z_slice = z_slice[sort_idx]
            
            # Plot this slice
            color = self.smile_colors(i % 10)
            label = 'ATM' if abs(moneyness) < 0.01 else f'K/S = {np.exp(moneyness):.2f}'
            ax.plot(y_slice, z_slice, '-', linewidth=2, color=color, label=label)
            
            # Add market points if available
            if 'x_raw' in surface_data and 'y_raw' in surface_data and 'z_raw' in surface_data:
                x_raw = surface_data['x_raw']
                y_raw = surface_data['y_raw']
                z_raw = surface_data['z_raw']
                
                # Filter points near this moneyness
                mask = np.abs(x_raw - moneyness) < 0.01
                if np.any(mask):
                    ax.scatter(y_raw[mask], z_raw[mask], color=color, marker='o', s=30, alpha=0.7)
        
        # Set labels and title
        ax.set_xlabel(y_label)
        ax.set_ylabel(z_label)
        ax.set_title(title)
        
        # Add legend and grid
        ax.legend()
        ax.grid(True)
        
        # Save figure if filename is provided
        if filename:
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            logger.info(f"Saved figure to {filepath}")
        
        # Show or close figure
        if show:
            plt.tight_layout()
        else:
            plt.close(fig)
        
        return fig
    
    def plot_model_vs_market(
        self,
        fitted_data: pd.DataFrame,
        title: str = "Model vs. Market Implied Volatilities",
        filename: Optional[str] = None,
        show: bool = True
    ) -> plt.Figure:
        """
        Plot model vs. market implied volatilities.
        
        Args:
            fitted_data: DataFrame containing market and fitted volatilities
            title: Plot title
            filename: If provided, save figure to this filename
            show: Whether to display the figure
            
        Returns:
            Matplotlib figure object
        """
        # Check if required columns exist
        required_cols = ['market_vol', 'fitted_vol']
        missing_cols = [col for col in required_cols if col not in fitted_data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Scatter plot of model vs. market
        ax1.scatter(
            fitted_data['market_vol'], 
            fitted_data['fitted_vol'],
            alpha=0.7
        )
        
        # Add diagonal line
        min_vol = min(fitted_data['market_vol'].min(), fitted_data['fitted_vol'].min())
        max_vol = max(fitted_data['market_vol'].max(), fitted_data['fitted_vol'].max())
        margin = 0.05 * (max_vol - min_vol)
        ax1.plot(
            [min_vol - margin, max_vol + margin], 
            [min_vol - margin, max_vol + margin], 
            'r--'
        )
        
        # Set labels
        ax1.set_xlabel('Market Implied Volatility')
        ax1.set_ylabel('Model Implied Volatility')
        ax1.set_title('Model vs. Market')
        
        # Calculate and plot errors
        fitted_data['error'] = fitted_data['fitted_vol'] - fitted_data['market_vol']
        fitted_data['rel_error'] = fitted_data['error'] / fitted_data['market_vol']
        
        # Histogram of relative errors
        ax2.hist(fitted_data['rel_error'] * 100, bins=20, alpha=0.7)
        ax2.set_xlabel('Relative Error (%)')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Distribution of Relative Errors')
        
        # Calculate error statistics
        mae = fitted_data['error'].abs().mean()
        rmse = np.sqrt((fitted_data['error'] ** 2).mean())
        mape = fitted_data['rel_error'].abs().mean() * 100
        
        # Add error statistics as text
        stats_text = f"MAE: {mae:.4f}\nRMSE: {rmse:.4f}\nMAPE: {mape:.2f}%"
        ax2.text(
            0.95, 0.95, stats_text,
            transform=ax2.transAxes,
            verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
        )
        
        # Set overall title
        fig.suptitle(title, fontsize=16)
        
        # Save figure if filename is provided
        if filename:
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            logger.info(f"Saved figure to {filepath}")
        
        # Show or close figure
        if show:
            plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for suptitle
        else:
            plt.close(fig)
        
        return fig
    
    def plot_parameter_evolution(
        self,
        parameters: Dict[str, Dict[str, float]],
        param_names: List[str] = None,
        title: str = "Evolution of Model Parameters",
        filename: Optional[str] = None,
        show: bool = True
    ) -> plt.Figure:
        """
        Plot the evolution of model parameters across different expiries.
        
        Args:
            parameters: Dictionary mapping expiry dates to parameter dictionaries
            param_names: List of parameter names to plot (if None, plots all)
            title: Plot title
            filename: If provided, save figure to this filename
            show: Whether to display the figure
            
        Returns:
            Matplotlib figure object
        """
        # Extract expiry dates and sort them
        expiries = sorted(parameters.keys())
        
        # Convert to datetime if they are strings
        if isinstance(expiries[0], str):
            expiries_dt = [pd.to_datetime(exp) for exp in expiries]
            # Sort again based on datetime
            expiries = [exp for _, exp in sorted(zip(expiries_dt, expiries))]
        
        # If no parameter names provided, use all from the first expiry
        if param_names is None and expiries:
            param_names = list(parameters[expiries[0]].keys())
        
        # Filter out 'time_to_expiry' or similar if present
        param_names = [p for p in param_names if p not in ['time_to_expiry', 'forward']]
        
        # Create figure with subplots for each parameter
        n_params = len(param_names)
        fig, axes = plt.subplots(n_params, 1, figsize=(10, 3 * n_params), sharex=True)
        
        # Handle case with only one parameter
        if n_params == 1:
            axes = [axes]
        
        # Plot each parameter
        for i, param_name in enumerate(param_names):
            ax = axes[i]
            
            # Extract parameter values
            values = [parameters[exp].get(param_name, np.nan) for exp in expiries]
            
            # Plot
            ax.plot(expiries, values, 'o-', linewidth=2)
            
            # Set labels
            ax.set_ylabel(param_name)
            ax.set_title(f"Evolution of {param_name}")
            ax.grid(True)
        
        # Set x-label for bottom subplot
        axes[-1].set_xlabel("Expiry Date")
        
        # Rotate x-tick labels if they are dates
        if isinstance(expiries[0], str):
            plt.setp(axes[-1].get_xticklabels(), rotation=45, ha='right')
        
        # Set overall title
        fig.suptitle(title, fontsize=16)
        
        # Save figure if filename is provided
        if filename:
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            logger.info(f"Saved figure to {filepath}")
        
        # Show or close figure
        if show:
            plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for suptitle
        else:
            plt.close(fig)
        
        return fig

        if show:
            plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for suptitle
        else:
                plt.close(fig)
        
        return fig
    
    def plot_historical_volatility_comparison(
        self,
        surface_data: Dict,
        historical_vol: pd.Series,
        title: str = "Implied vs. Historical Volatility",
        filename: Optional[str] = None,
        show: bool = True
    ) -> plt.Figure:
        """
        Compare implied volatility surface with historical realized volatility.
        
        Args:
            surface_data: Dictionary containing volatility surface data
            historical_vol: Series of historical realized volatility
            title: Plot title
            filename: If provided, save figure to this filename
            show: Whether to display the figure
            
        Returns:
            Matplotlib figure object
        """
        # Extract ATM volatility from surface
        X = surface_data['x_grid']
        Y = surface_data['y_grid']
        Z = surface_data['z_grid']
        
        # Find ATM slice (log moneyness closest to 0)
        atm_idx = np.abs(X[0, :]).argmin()
        
        # Extract term structure for ATM options
        expiries = Y[:, atm_idx]
        atm_vols = Z[:, atm_idx]
        
        # Sort by expiry
        sort_idx = np.argsort(expiries)
        expiries = expiries[sort_idx]
        atm_vols = atm_vols[sort_idx]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot ATM implied volatility term structure
        ax.plot(
            expiries, 
            atm_vols, 
            'b-', 
            linewidth=2, 
            label='ATM Implied Volatility'
        )
        
        # Plot historical volatility as a horizontal line
        hist_vol_value = historical_vol.mean()
        ax.axhline(
            y=hist_vol_value,
            color='r',
            linestyle='--',
            linewidth=2,
            label=f'Historical Volatility ({hist_vol_value:.2%})'
        )
        
        # Add historical volatility range (±1 std)
        hist_vol_std = historical_vol.std()
        ax.fill_between(
            expiries,
            hist_vol_value - hist_vol_std,
            hist_vol_value + hist_vol_std,
            color='r',
            alpha=0.2,
            label=f'Historical Vol ±1σ ({hist_vol_std:.2%})'
        )
        
        # Set labels and title
        ax.set_xlabel('Time to Expiry (years)')
        ax.set_ylabel('Volatility')
        ax.set_title(title)
        
        # Format y-axis as percentage
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
        
        # Add legend and grid
        ax.legend()
        ax.grid(True)
        
        # Save figure if filename is provided
        if filename:
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            logger.info(f"Saved figure to {filepath}")
        
        # Show or close figure
        if show:
            plt.tight_layout()
        else:
            plt.close(fig)
        
        return fig
    
    def plot_surface_evolution(
        self,
        surface_data_list: List[Dict],
        dates: List[str],
        title: str = "Evolution of Volatility Surface Over Time",
        filename: Optional[str] = None,
        show: bool = True
    ) -> plt.Figure:
        """
        Visualize the evolution of volatility surface over time.
        
        Args:
            surface_data_list: List of surface data dictionaries for different dates
            dates: List of date strings corresponding to each surface
            title: Plot title
            filename: If provided, save figure to this filename
            show: Whether to display the figure
            
        Returns:
            Matplotlib figure object
        """
        # Ensure we have at least 2 surfaces
        if len(surface_data_list) < 2 or len(dates) < 2:
            logger.error("Need at least 2 surfaces to visualize evolution")
            return None
        
        # Create figure with subplots
        n_surfaces = len(surface_data_list)
        n_cols = min(3, n_surfaces)
        n_rows = (n_surfaces + n_cols - 1) // n_cols
        
        fig = plt.figure(figsize=(6 * n_cols, 5 * n_rows))
        
        # Plot each surface
        for i, (surface_data, date) in enumerate(zip(surface_data_list, dates)):
            # Extract data
            X = surface_data['x_grid']
            Y = surface_data['y_grid']
            Z = surface_data['z_grid']
            
            # Create subplot
            ax = fig.add_subplot(n_rows, n_cols, i + 1, projection='3d')
            
            # Plot the surface
            surf = ax.plot_surface(
                X, Y, Z, 
                cmap=self.surface_cmap,
                linewidth=0, 
                antialiased=True,
                alpha=0.8
            )
            
            # Set labels
            ax.set_xlabel('Log Moneyness')
            ax.set_ylabel('Time to Expiry')
            ax.set_zlabel('Implied Vol')
            
            # Set title for this subplot
            ax.set_title(f"Date: {date}")
            
            # Set view angle
            ax.view_init(30, -60)
        
        # Set overall title
        fig.suptitle(title, fontsize=16)
        
        # Save figure if filename is provided
        if filename:
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            logger.info(f"Saved figure to {filepath}")
        
        # Show or close figure
        if show:
            plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for suptitle
        else:
            plt.close(fig)
        
        return fig
    
    def plot_risk_metrics(
        self,
        surface_data: Dict,
        underlying_price: float,
        title: str = "Volatility Surface Risk Metrics",
        filename: Optional[str] = None,
        show: bool = True
    ) -> plt.Figure:
        """
        Calculate and visualize risk metrics derived from the volatility surface.
        
        Args:
            surface_data: Dictionary containing volatility surface data
            underlying_price: Current price of the underlying asset
            title: Plot title
            filename: If provided, save figure to this filename
            show: Whether to display the figure
            
        Returns:
            Matplotlib figure object
        """
        # Extract data
        X = surface_data['x_grid']  # Log moneyness
        Y = surface_data['y_grid']  # Time to expiry
        Z = surface_data['z_grid']  # Implied volatility
        
        # Create figure with 2x2 subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # 1. Skew (dVol/dK) - First derivative with respect to strike
        skew = np.zeros_like(Z)
        for i in range(Z.shape[0]):
            # Convert log moneyness to strike
            strikes = underlying_price * np.exp(X[i, :])
            # Calculate derivative
            skew[i, :] = np.gradient(Z[i, :], strikes)
        
        # Plot skew surface
        ax = axes[0, 0]
        cs = ax.contourf(X[0, :], Y[:, 0], skew, cmap='coolwarm')
        ax.set_xlabel('Log Moneyness')
        ax.set_ylabel('Time to Expiry')
        ax.set_title('Volatility Skew (dVol/dK)')
        fig.colorbar(cs, ax=ax)
        
        # 2. Convexity (d²Vol/dK²) - Second derivative with respect to strike
        convexity = np.zeros_like(Z)
        for i in range(Z.shape[0]):
            # Convert log moneyness to strike
            strikes = underlying_price * np.exp(X[i, :])
            # Calculate first derivative
            first_deriv = np.gradient(Z[i, :], strikes)
            # Calculate second derivative
            convexity[i, :] = np.gradient(first_deriv, strikes)
        
        # Plot convexity surface
        ax = axes[0, 1]
        cs = ax.contourf(X[0, :], Y[:, 0], convexity, cmap='coolwarm')
        ax.set_xlabel('Log Moneyness')
        ax.set_ylabel('Time to Expiry')
        ax.set_title('Volatility Convexity (d²Vol/dK²)')
        fig.colorbar(cs, ax=ax)
        
        # 3. Term Structure Slope (dVol/dT) - Derivative with respect to time
        term_slope = np.zeros_like(Z)
        for j in range(Z.shape[1]):
            term_slope[:, j] = np.gradient(Z[:, j], Y[:, j])
        
        # Plot term structure slope
        ax = axes[1, 0]
        cs = ax.contourf(X[0, :], Y[:, 0], term_slope, cmap='coolwarm')
        ax.set_xlabel('Log Moneyness')
        ax.set_ylabel('Time to Expiry')
        ax.set_title('Term Structure Slope (dVol/dT)')
        fig.colorbar(cs, ax=ax)
        
        # 4. Volatility of Volatility (standard deviation of local vol)
        # Calculate local volatility (simplified Dupire formula)
        local_vol = np.zeros_like(Z)
        for i in range(1, Z.shape[0] - 1):
            for j in range(1, Z.shape[1] - 1):
                # Extract values
                vol = Z[i, j]
                t = Y[i, j]
                k = X[i, j]
                
                # Calculate derivatives (central difference)
                dvol_dt = (Z[i+1, j] - Z[i-1, j]) / (Y[i+1, j] - Y[i-1, j])
                dvol_dk = (Z[i, j+1] - Z[i, j-1]) / (X[i, j+1] - X[i, j-1])
                d2vol_dk2 = (Z[i, j+1] - 2*Z[i, j] + Z[i, j-1]) / ((X[i, j+1] - X[i, j-1])/2)**2
                
                # Simplified Dupire formula
                if t > 0 and vol > 0:
                    local_vol[i, j] = vol / np.sqrt(1 + 2*t*dvol_dt/vol + t*k*dvol_dk + 0.25*t*k**2*d2vol_dk2)
                else:
                    local_vol[i, j] = vol
        
        # Calculate volatility of volatility (standard deviation of local vol)
        vol_of_vol = np.std(local_vol, axis=1)
        
        # Plot volatility of volatility
        ax = axes[1, 1]
        ax.plot(Y[:, 0], vol_of_vol, 'b-', linewidth=2)
        ax.set_xlabel('Time to Expiry')
        ax.set_ylabel('Volatility of Volatility')
        ax.set_title('Volatility of Volatility vs. Time to Expiry')
        ax.grid(True)
        
        # Set overall title
        fig.suptitle(title, fontsize=16)
        
        # Save figure if filename is provided
        if filename:
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            logger.info(f"Saved figure to {filepath}")
        
        # Show or close figure
        if show:
            plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for suptitle
        else:
            plt.close(fig)
        
        return fig
    
    def create_dashboard(
        self,
        surface_data: Dict,
        historical_vol: pd.Series = None,
        fitted_data: pd.DataFrame = None,
        parameters: Dict = None,
        underlying_price: float = None,
        commodity_name: str = "Commodity",
        filename: str = "volatility_dashboard.pdf",
        show: bool = False
    ) -> None:
        """
        Create a comprehensive dashboard with multiple visualizations.
        
        Args:
            surface_data: Dictionary containing volatility surface data
            historical_vol: Series of historical realized volatility (optional)
            fitted_data: DataFrame with model vs. market data (optional)
            parameters: Dictionary of model parameters (optional)
            underlying_price: Current price of the underlying asset (optional)
            commodity_name: Name of the commodity
            filename: Filename for the dashboard PDF
            show: Whether to display the figures during creation
        """
        from matplotlib.backends.backend_pdf import PdfPages
        
        # Create PDF
        filepath = os.path.join(self.output_dir, filename)
        with PdfPages(filepath) as pdf:
            # 1. 3D Surface
            fig = self.plot_3d_surface(
                surface_data,
                title=f"{commodity_name} Implied Volatility Surface",
                show=show
            )
            pdf.savefig(fig)
            plt.close(fig)
            
            # 2. Volatility Smile
            fig = self.plot_volatility_smile(
                surface_data,
                title=f"{commodity_name} Volatility Smile at Different Expiries",
                show=show
            )
            pdf.savefig(fig)
            plt.close(fig)
            
            # 3. Term Structure
            fig = self.plot_term_structure(
                surface_data,
                title=f"{commodity_name} Volatility Term Structure",
                show=show
            )
            pdf.savefig(fig)
            plt.close(fig)
            
            # 4. Historical Comparison (if provided)
            if historical_vol is not None:
                fig = self.plot_historical_volatility_comparison(
                    surface_data,
                    historical_vol,
                    title=f"{commodity_name} Implied vs. Historical Volatility",
                    show=show
                )
                pdf.savefig(fig)
                plt.close(fig)
            
            # 5. Model vs. Market (if provided)
            if fitted_data is not None:
                fig = self.plot_model_vs_market(
                    fitted_data,
                    title=f"{commodity_name} Model vs. Market Implied Volatilities",
                    show=show
                )
                pdf.savefig(fig)
                plt.close(fig)
            
            # 6. Parameter Evolution (if provided)
            if parameters is not None:
                fig = self.plot_parameter_evolution(
                    parameters,
                    title=f"{commodity_name} Model Parameter Evolution",
                    show=show
                )
                pdf.savefig(fig)
                plt.close(fig)
            
            # 7. Risk Metrics (if underlying price provided)
            if underlying_price is not None:
                fig = self.plot_risk_metrics(
                    surface_data,
                    underlying_price,
                    title=f"{commodity_name} Volatility Surface Risk Metrics",
                    show=show
                )
                pdf.savefig(fig)
                plt.close(fig)
            
            # Add metadata
            d = pdf.infodict()
            d['Title'] = f'{commodity_name} Volatility Surface Analysis'
            d['Author'] = 'Volatility Surface Analyzer'
            d['Subject'] = 'Commodity Volatility Analysis'
            d['Keywords'] = 'volatility surface, options, commodities'
            d['CreationDate'] = datetime.now()
            d['ModDate'] = datetime.now()
        
        logger.info(f"Created dashboard PDF at {filepath}")


# Example usage
if __name__ == "__main__":
    # This is a simplified example - in production, you would:
    # 1. Load real volatility surface data
    # 2. Generate visualizations based on your specific needs
    
    # Create synthetic data for demonstration
    np.random.seed(42)
    
    # Generate a grid
    x = np.linspace(-0.5, 0.5, 50)  # Log moneyness
    y = np.linspace(0.1, 2.0, 50)   # Time to expiry
    X, Y = np.meshgrid(x, y)
    
    # Generate a volatility surface with smile and term structure
    Z = 0.2 + 0.1 * X**2 + 0.05 * np.sqrt(Y)
    
    # Add some random market data points
    n_points = 100
    x_raw = np.random.uniform(-0.4, 0.4, n_points)
    y_raw = np.random.uniform(0.1, 1.9, n_points)
    z_raw = 0.2 + 0.1 * x_raw**2 + 0.05 * np.sqrt(y_raw) + np.random.normal(0, 0.01, n_points)
    
    # Create surface data dictionary
    surface_data = {
        'x_grid': X,
        'y_grid': Y,
        'z_grid': Z,
        'x_raw': x_raw,
        'y_raw': y_raw,
        'z_raw': z_raw,
        'x_label': 'Log Moneyness (log(K/S))',
        'y_label': 'Time to Expiry (years)',
        'z_label': 'Implied Volatility'
    }
    
    # Create visualizer
    visualizer = VolatilitySurfaceVisualizer()
    
    # Generate visualizations
    visualizer.plot_3d_surface(surface_data, title="Example Commodity Volatility Surface")
    visualizer.plot_volatility_smile(surface_data)
    visualizer.plot_term_structure(surface_data)
    
    # Show plots
    plt.show()

