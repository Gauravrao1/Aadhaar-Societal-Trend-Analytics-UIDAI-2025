"""
Visualization Module
Creates visualizations for Aadhaar enrolment analytics
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional
import plotly.graph_objects as go
import plotly.express as px


class EnrolmentVisualizer:
    """Creates visualizations for enrolment data analysis"""
    
    def __init__(self, style: str = 'seaborn-v0_8-darkgrid'):
        """
        Initialize visualizer
        
        Args:
            style: Matplotlib style to use
        """
        try:
            plt.style.use(style)
        except:
            pass  # Use default style if not available
        sns.set_palette("husl")
    
    def plot_time_series(self, data: pd.DataFrame, title: str = "Enrolment Trends Over Time",
                        save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot time series of enrolments
        
        Args:
            data: DataFrame with 'date' and 'enrolments' columns
            title: Plot title
            save_path: Optional path to save figure
        
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.plot(data['date'], data['enrolments'], linewidth=2)
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Enrolments', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_seasonal_decomposition(self, decomposition: Dict,
                                   save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot seasonal decomposition components
        
        Args:
            decomposition: Dictionary with trend, seasonal, residual components
            save_path: Optional path to save figure
        
        Returns:
            Matplotlib figure
        """
        if 'error' in decomposition:
            print(f"Cannot plot decomposition: {decomposition['error']}")
            return None
        
        fig, axes = plt.subplots(4, 1, figsize=(12, 10))
        
        components = ['observed', 'trend', 'seasonal', 'residual']
        titles = ['Observed', 'Trend', 'Seasonal', 'Residual']
        
        for ax, component, title in zip(axes, components, titles):
            if component in decomposition:
                comp_data = pd.Series(decomposition[component])
                ax.plot(comp_data.index, comp_data.values)
                ax.set_ylabel(title, fontsize=10)
                ax.grid(True, alpha=0.3)
        
        axes[0].set_title('Seasonal Decomposition', fontsize=14, fontweight='bold')
        axes[-1].set_xlabel('Date', fontsize=12)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_district_comparison(self, metrics: pd.DataFrame, metric: str = 'avg_daily_enrolments',
                                top_n: int = 10, save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot comparison of districts by a metric
        
        Args:
            metrics: DataFrame with district metrics
            metric: Metric to compare
            top_n: Number of top districts to show
            save_path: Optional path to save figure
        
        Returns:
            Matplotlib figure
        """
        top_districts = metrics.nlargest(top_n, metric)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.barh(top_districts['district'], top_districts[metric])
        ax.set_xlabel(metric.replace('_', ' ').title(), fontsize=12)
        ax.set_ylabel('District', fontsize=12)
        ax.set_title(f'Top {top_n} Districts by {metric.replace("_", " ").title()}',
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_monthly_heatmap(self, data: pd.DataFrame,
                            save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot heatmap of monthly enrolments
        
        Args:
            data: DataFrame with 'date' and 'enrolments' columns
            save_path: Optional path to save figure
        
        Returns:
            Matplotlib figure
        """
        # Prepare data for heatmap
        data = data.copy()
        data['year'] = data['date'].dt.year
        data['month'] = data['date'].dt.month
        
        pivot_data = data.groupby(['year', 'month'])['enrolments'].sum().reset_index()
        pivot_table = pivot_data.pivot(index='year', columns='month', values='enrolments')
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        sns.heatmap(pivot_table, annot=True, fmt='.0f', cmap='YlOrRd',
                   cbar_kws={'label': 'Enrolments'}, ax=ax)
        
        ax.set_xlabel('Month', fontsize=12)
        ax.set_ylabel('Year', fontsize=12)
        ax.set_title('Monthly Enrolment Heatmap', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_predictions(self, historical: pd.DataFrame, predictions: pd.DataFrame,
                        save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot historical data with predictions
        
        Args:
            historical: DataFrame with historical data
            predictions: DataFrame with predicted data
            save_path: Optional path to save figure
        
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # Plot historical
        ax.plot(historical['date'], historical['enrolments'],
               label='Historical', linewidth=2, color='blue')
        
        # Plot predictions
        ax.plot(predictions['date'], predictions['predicted_enrolments'],
               label='Predicted', linewidth=2, color='red', linestyle='--')
        
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Enrolments', fontsize=12)
        ax.set_title('Historical Data and Predictions', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def create_interactive_dashboard(self, data: pd.DataFrame,
                                    metrics: pd.DataFrame) -> go.Figure:
        """
        Create interactive dashboard with Plotly
        
        Args:
            data: Enrolment time series data
            metrics: District metrics
        
        Returns:
            Plotly figure
        """
        from plotly.subplots import make_subplots
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Time Series', 'District Comparison',
                          'Monthly Trends', 'Pressure Score'),
            specs=[[{"type": "scatter"}, {"type": "bar"}],
                   [{"type": "scatter"}, {"type": "bar"}]]
        )
        
        # Time series
        fig.add_trace(
            go.Scatter(x=data['date'], y=data['enrolments'],
                      mode='lines', name='Enrolments'),
            row=1, col=1
        )
        
        # District comparison
        top_districts = metrics.nlargest(10, 'total_enrolments')
        fig.add_trace(
            go.Bar(x=top_districts['district'], y=top_districts['total_enrolments'],
                  name='Total Enrolments'),
            row=1, col=2
        )
        
        # Monthly trends
        monthly = data.groupby(data['date'].dt.to_period('M'))['enrolments'].sum()
        fig.add_trace(
            go.Scatter(x=monthly.index.astype(str), y=monthly.values,
                      mode='lines+markers', name='Monthly'),
            row=2, col=1
        )
        
        # Pressure scores
        if 'pressure_score' in metrics.columns:
            top_pressure = metrics.nlargest(10, 'pressure_score')
            fig.add_trace(
                go.Bar(x=top_pressure['district'], y=top_pressure['pressure_score'],
                      name='Pressure Score'),
                row=2, col=2
            )
        
        fig.update_layout(height=800, showlegend=True,
                         title_text="Aadhaar Enrolment Analytics Dashboard")
        
        return fig
