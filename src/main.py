"""
Main Module for Aadhaar Societal Trend Analytics
Provides unified interface for all analysis modules
"""

import pandas as pd
from typing import Dict, Optional
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models.enrolment_data import EnrolmentDataset
from analysis.seasonal_trends import SeasonalTrendDetector
from analysis.district_pressure import DistrictPressureAnalyzer
from analysis.predictive_demand import DemandPredictor
from visualization.plots import EnrolmentVisualizer
from utils.data_utils import generate_sample_data, save_data_to_csv


class AadhaarAnalytics:
    """
    Main class for Aadhaar enrolment analytics
    Provides unified interface for all analysis modules
    """
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize analytics with enrolment data
        
        Args:
            data: DataFrame with enrolment data
        """
        self.dataset = EnrolmentDataset(data)
        self.visualizer = EnrolmentVisualizer()
    
    def analyze_seasonal_trends(self, district: Optional[str] = None) -> Dict:
        """
        Analyze seasonal trends in enrolment data
        
        Args:
            district: Optional district name for district-specific analysis
        
        Returns:
            Dictionary with seasonal insights
        """
        if district:
            data = self.dataset.filter_by_district(district)
        else:
            data = self.dataset.data
        
        detector = SeasonalTrendDetector(data)
        return detector.get_seasonal_summary()
    
    def analyze_district_pressure(self) -> Dict:
        """
        Analyze district-level pressure and capacity
        
        Returns:
            Dictionary with district pressure insights
        """
        analyzer = DistrictPressureAnalyzer(self.dataset.data)
        
        metrics = analyzer.calculate_district_metrics()
        high_pressure = analyzer.identify_high_pressure_districts()
        surges = analyzer.detect_surges()
        
        return {
            'district_metrics': metrics.to_dict('records'),
            'high_pressure_districts': high_pressure,
            'surge_events': surges[:10]  # Top 10 surges
        }
    
    def predict_demand(self, district: Optional[str] = None, days: int = 30) -> Dict:
        """
        Predict future enrolment demand
        
        Args:
            district: Optional district name for district-specific prediction
            days: Number of days to predict ahead
        
        Returns:
            Dictionary with demand predictions and indicators
        """
        if district:
            data = self.dataset.filter_by_district(district)
        else:
            data = self.dataset.data
        
        predictor = DemandPredictor(data)
        
        # Train model
        model_performance = predictor.train_linear_model()
        
        # Get predictions
        predictions = predictor.predict_next_period(days=days, model_type='linear')
        
        # Get demand indicators
        indicators = predictor.calculate_demand_indicators()
        
        # Get peak periods
        peak_periods = predictor.identify_peak_demand_periods(future_days=days)
        
        return {
            'model_performance': model_performance,
            'predictions': predictions.to_dict('records'),
            'demand_indicators': indicators,
            'peak_periods': peak_periods[:5]  # Top 5 peaks
        }
    
    def generate_comprehensive_report(self) -> Dict:
        """
        Generate comprehensive analytics report
        
        Returns:
            Dictionary with all analytics results
        """
        report = {
            'data_summary': {
                'total_districts': len(self.dataset.get_districts()),
                'total_states': len(self.dataset.get_states()),
                'date_range': {
                    'start': str(self.dataset.get_date_range()[0]),
                    'end': str(self.dataset.get_date_range()[1])
                },
                'total_enrolments': int(self.dataset.data['enrolments'].sum())
            },
            'seasonal_trends': self.analyze_seasonal_trends(),
            'district_pressure': self.analyze_district_pressure(),
            'demand_predictions': self.predict_demand(days=30)
        }
        
        return report
    
    def create_visualizations(self, output_dir: str = 'outputs'):
        """
        Create and save all visualizations
        
        Args:
            output_dir: Directory to save visualizations
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Time series plot
        self.visualizer.plot_time_series(
            self.dataset.data.groupby('date')['enrolments'].sum().reset_index(),
            save_path=f'{output_dir}/time_series.png'
        )
        
        # District comparison
        analyzer = DistrictPressureAnalyzer(self.dataset.data)
        metrics = analyzer.calculate_district_metrics()
        self.visualizer.plot_district_comparison(
            metrics,
            save_path=f'{output_dir}/district_comparison.png'
        )
        
        # Monthly heatmap
        self.visualizer.plot_monthly_heatmap(
            self.dataset.data,
            save_path=f'{output_dir}/monthly_heatmap.png'
        )
        
        print(f"Visualizations saved to {output_dir}/")


def main():
    """Main entry point for demonstration"""
    print("=" * 60)
    print("Aadhaar Societal Trend Analytics - UIDAI 2026")
    print("=" * 60)
    print()
    
    # Generate sample data
    print("Generating sample data...")
    districts = ['Mumbai', 'Delhi', 'Bangalore', 'Chennai', 'Kolkata']
    states = ['Maharashtra', 'Delhi', 'Karnataka', 'Tamil Nadu', 'West Bengal']
    
    sample_data = generate_sample_data(
        districts=districts,
        states=states,
        start_date='2024-01-01',
        end_date='2025-12-31',
        base_enrolments=150
    )
    
    print(f"Generated {len(sample_data)} records")
    print()
    
    # Initialize analytics
    print("Initializing analytics engine...")
    analytics = AadhaarAnalytics(sample_data)
    print()
    
    # Generate comprehensive report
    print("Generating comprehensive analytics report...")
    report = analytics.generate_comprehensive_report()
    
    # Display summary
    print("\n" + "=" * 60)
    print("DATA SUMMARY")
    print("=" * 60)
    for key, value in report['data_summary'].items():
        print(f"{key}: {value}")
    
    print("\n" + "=" * 60)
    print("SEASONAL TRENDS")
    print("=" * 60)
    trends = report['seasonal_trends']
    print(f"Seasonality Strength: {trends['seasonality_strength']:.2f}")
    print(f"Average Monthly Enrolments: {trends['average_monthly_enrolments']:.0f}")
    print(f"\nTop 3 Peak Months:")
    for i, (month, avg) in enumerate(list(trends['peak_months'].items())[:3], 1):
        print(f"  {i}. Month {month}: {avg:.0f} enrolments")
    
    print("\n" + "=" * 60)
    print("DISTRICT PRESSURE")
    print("=" * 60)
    pressure = report['district_pressure']
    print(f"\nTop 3 High Pressure Districts:")
    for i, district in enumerate(pressure['high_pressure_districts'][:3], 1):
        print(f"  {i}. {district['district']} (Pressure Score: {district['pressure_score']:.2f})")
    
    print("\n" + "=" * 60)
    print("DEMAND PREDICTIONS")
    print("=" * 60)
    predictions = report['demand_predictions']
    indicators = predictions['demand_indicators']
    print(f"Current 30-day Average: {indicators['current_30d_avg']:.0f}")
    print(f"Growth Rate: {indicators['growth_rate_percentage']:.1f}%")
    print(f"Trend: {indicators['trend']}")
    print(f"Predicted 30-day Total: {indicators['predicted_30d_total']:.0f}")
    
    # Create visualizations
    print("\n" + "=" * 60)
    print("GENERATING VISUALIZATIONS")
    print("=" * 60)
    analytics.create_visualizations()
    
    print("\n" + "=" * 60)
    print("Analysis Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
