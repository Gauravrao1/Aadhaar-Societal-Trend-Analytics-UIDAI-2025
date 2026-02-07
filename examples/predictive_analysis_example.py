"""
Example: Predictive Demand Analysis
Demonstrates how to predict future enrolment demand
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils.data_utils import generate_sample_data
from analysis.predictive_demand import DemandPredictor
from visualization.plots import EnrolmentVisualizer


def main():
    """Run predictive demand analysis example"""
    print("Predictive Demand Analysis Example")
    print("=" * 60)
    
    # Generate sample data
    print("\n1. Generating historical enrolment data...")
    districts = ['Mumbai']
    states = ['Maharashtra']
    
    data = generate_sample_data(
        districts=districts,
        states=states,
        start_date='2023-01-01',
        end_date='2025-12-31',
        base_enrolments=200
    )
    
    # Aggregate for overall prediction
    aggregated = data.groupby('date')['enrolments'].sum().reset_index()
    print(f"   Generated {len(aggregated)} days of historical data")
    
    # Initialize predictor
    print("\n2. Training prediction models...")
    predictor = DemandPredictor(aggregated)
    
    # Train linear model
    model_performance = predictor.train_linear_model()
    
    if 'error' not in model_performance:
        print(f"\n   Model Performance:")
        print(f"   - Model Type: {model_performance['model_type']}")
        print(f"   - R² Score: {model_performance['r2_score']:.4f}")
        print(f"   - Mean Absolute Error: {model_performance['mae']:.2f}")
    else:
        print(f"   Error: {model_performance['error']}")
    
    # Calculate demand indicators
    print("\n3. Calculating demand indicators...")
    indicators = predictor.calculate_demand_indicators()
    
    print(f"\n   Current Demand Indicators:")
    print(f"   - Current 30-day Average: {indicators['current_30d_avg']:.0f}")
    print(f"   - Historical Average: {indicators['historical_avg']:.0f}")
    print(f"   - Growth Rate: {indicators['growth_rate_percentage']:+.1f}%")
    print(f"   - Volatility: {indicators['volatility_coefficient']:.3f}")
    print(f"   - Trend: {indicators['trend']}")
    print(f"   - Current Daily Average: {indicators['current_daily_avg']:.0f}")
    print(f"   - Peak Daily: {indicators['peak_daily']}")
    
    # Predict next 30 days
    print("\n4. Predicting next 30 days...")
    predictions_linear = predictor.predict_next_period(days=30, model_type='linear')
    predictions_ma = predictor.predict_next_period(days=30, model_type='ma')
    predictions_exp = predictor.predict_next_period(days=30, model_type='exponential')
    
    print(f"\n   30-Day Predictions:")
    print(f"   - Linear Model: {predictions_linear['predicted_enrolments'].sum():.0f} total")
    print(f"   - Moving Average: {predictions_ma['predicted_enrolments'].sum():.0f} total")
    print(f"   - Exponential: {predictions_exp['predicted_enrolments'].sum():.0f} total")
    
    print(f"\n   Predicted Daily Averages:")
    print(f"   - Linear Model: {predictions_linear['predicted_enrolments'].mean():.0f}")
    print(f"   - Moving Average: {predictions_ma['predicted_enrolments'].mean():.0f}")
    print(f"   - Exponential: {predictions_exp['predicted_enrolments'].mean():.0f}")
    
    # Identify peak demand periods
    print("\n5. Identifying peak demand periods...")
    peak_periods = predictor.identify_peak_demand_periods(future_days=90)
    
    if peak_periods:
        print(f"\n   Expected Peak Periods (next 90 days):")
        for i, peak in enumerate(peak_periods[:5], 1):
            print(f"   {i}. {peak['date']} (Month {peak['month']}): "
                  f"{peak['predicted_enrolments']:.0f} enrolments")
    
    # Create visualizations
    print("\n6. Creating visualizations...")
    visualizer = EnrolmentVisualizer()
    
    # Historical and predictions plot
    visualizer.plot_predictions(
        aggregated.tail(90),  # Last 90 days of historical
        predictions_linear,
        save_path='outputs/demand_predictions.png'
    )
    print("   - Saved predictions plot")
    
    # Time series with trends
    visualizer.plot_time_series(
        aggregated,
        title="Historical Enrolment Trends",
        save_path='outputs/historical_trends.png'
    )
    print("   - Saved historical trends plot")
    
    print("\n" + "=" * 60)
    print("Analysis complete! Check 'outputs/' directory for visualizations.")
    print("=" * 60)


if __name__ == "__main__":
    # Create outputs directory
    os.makedirs('outputs', exist_ok=True)
    main()
