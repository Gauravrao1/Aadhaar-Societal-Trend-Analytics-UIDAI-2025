"""
Example: Seasonal Trend Analysis
Demonstrates how to analyze seasonal patterns in Aadhaar enrolment data
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils.data_utils import generate_sample_data
from analysis.seasonal_trends import SeasonalTrendDetector
from visualization.plots import EnrolmentVisualizer
import matplotlib.pyplot as plt


def main():
    """Run seasonal trend analysis example"""
    print("Seasonal Trend Analysis Example")
    print("=" * 60)
    
    # Generate sample data
    print("\n1. Generating sample data...")
    districts = ['Mumbai', 'Delhi']
    states = ['Maharashtra', 'Delhi']
    
    data = generate_sample_data(
        districts=districts,
        states=states,
        start_date='2023-01-01',
        end_date='2025-12-31',
        base_enrolments=200
    )
    
    # Aggregate data for overall trends
    aggregated = data.groupby('date')['enrolments'].sum().reset_index()
    print(f"   Generated {len(aggregated)} days of data")
    
    # Initialize detector
    print("\n2. Analyzing seasonal patterns...")
    detector = SeasonalTrendDetector(aggregated)
    
    # Get seasonal summary
    summary = detector.get_seasonal_summary()
    
    print("\n   Seasonal Insights:")
    print(f"   - Seasonality Strength: {summary['seasonality_strength']:.3f}")
    print(f"   - Total Enrolments: {summary['total_enrolments']:,}")
    print(f"   - Average Monthly: {summary['average_monthly_enrolments']:.0f}")
    
    print("\n   Peak Months (Top 5):")
    for i, (month, avg) in enumerate(list(summary['peak_months'].items())[:5], 1):
        month_names = ['', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        print(f"   {i}. {month_names[month]}: {avg:.0f} avg enrolments")
    
    print("\n   Peak Quarters:")
    for quarter, avg in summary['peak_quarters'].items():
        print(f"   Q{quarter}: {avg:.0f} avg enrolments")
    
    # Detect anomalies
    if summary['anomalous_periods']:
        print(f"\n   Detected {len(summary['anomalous_periods'])} anomalous periods:")
        for anomaly in summary['anomalous_periods'][:3]:
            print(f"   - {anomaly['date']}: {anomaly['type']} "
                  f"(z-score: {anomaly['z_score']:.2f})")
    
    # Perform seasonal decomposition
    print("\n3. Performing seasonal decomposition...")
    decomposition = detector.detect_seasonal_pattern(period=12)
    
    if 'error' not in decomposition:
        print("   Decomposition successful!")
        
        # Create visualizations
        print("\n4. Creating visualizations...")
        visualizer = EnrolmentVisualizer()
        
        # Time series plot
        fig1 = visualizer.plot_time_series(aggregated, 
                                          title="Aadhaar Enrolment Time Series")
        plt.savefig('outputs/seasonal_example_timeseries.png', dpi=300, bbox_inches='tight')
        print("   - Saved time series plot")
        
        # Seasonal decomposition plot
        fig2 = visualizer.plot_seasonal_decomposition(decomposition)
        if fig2:
            plt.savefig('outputs/seasonal_example_decomposition.png', 
                       dpi=300, bbox_inches='tight')
            print("   - Saved decomposition plot")
        
        # Monthly heatmap
        fig3 = visualizer.plot_monthly_heatmap(aggregated)
        plt.savefig('outputs/seasonal_example_heatmap.png', dpi=300, bbox_inches='tight')
        print("   - Saved monthly heatmap")
        
        plt.close('all')
    else:
        print(f"   Error: {decomposition['error']}")
    
    print("\n" + "=" * 60)
    print("Analysis complete! Check 'outputs/' directory for visualizations.")
    print("=" * 60)


if __name__ == "__main__":
    # Create outputs directory
    os.makedirs('outputs', exist_ok=True)
    main()
