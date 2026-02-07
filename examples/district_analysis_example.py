"""
Example: District Pressure Analysis
Demonstrates how to analyze district-level pressure and capacity
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils.data_utils import generate_sample_data
from analysis.district_pressure import DistrictPressureAnalyzer
from visualization.plots import EnrolmentVisualizer


def main():
    """Run district pressure analysis example"""
    print("District Pressure Analysis Example")
    print("=" * 60)
    
    # Generate sample data for multiple districts
    print("\n1. Generating sample data for 10 districts...")
    districts = ['Mumbai', 'Delhi', 'Bangalore', 'Chennai', 'Kolkata',
                'Hyderabad', 'Pune', 'Ahmedabad', 'Jaipur', 'Lucknow']
    states = ['Maharashtra', 'Delhi', 'Karnataka', 'Tamil Nadu', 'West Bengal',
             'Telangana', 'Maharashtra', 'Gujarat', 'Rajasthan', 'Uttar Pradesh']
    
    data = generate_sample_data(
        districts=districts,
        states=states,
        start_date='2024-01-01',
        end_date='2025-12-31',
        base_enrolments=150
    )
    
    print(f"   Generated {len(data)} records")
    
    # Initialize analyzer
    print("\n2. Analyzing district-level pressure...")
    analyzer = DistrictPressureAnalyzer(data)
    
    # Calculate district metrics
    metrics = analyzer.calculate_district_metrics()
    print(f"\n   Analyzed {len(metrics)} districts")
    
    print("\n   Top 5 Districts by Total Enrolments:")
    top_districts = metrics.nlargest(5, 'total_enrolments')
    for i, row in enumerate(top_districts.iterrows(), 1):
        district = row[1]
        print(f"   {i}. {district['district']} ({district['state']}): "
              f"{district['total_enrolments']:,} total enrolments")
    
    # Identify high pressure districts
    print("\n3. Identifying high-pressure districts...")
    high_pressure = analyzer.identify_high_pressure_districts(threshold_percentile=70)
    
    print(f"\n   Found {len(high_pressure)} high-pressure districts:")
    for i, district in enumerate(high_pressure[:5], 1):
        print(f"   {i}. {district['district']} - Pressure Score: {district['pressure_score']:.2f}")
        print(f"      Current Avg: {district['current_rolling_avg']:.0f}, "
              f"Trend: {district['trend_percentage']:+.1f}%")
    
    # Calculate capacity utilization
    print("\n4. Calculating capacity requirements...")
    capacity = analyzer.calculate_capacity_utilization(capacity_per_centre=100)
    
    print("\n   Capacity Analysis (Top 5 districts):")
    top_capacity = capacity.nlargest(5, 'estimated_centres_needed')
    for i, row in enumerate(top_capacity.iterrows(), 1):
        district = row[1]
        print(f"   {i}. {district['district']}:")
        print(f"      - Estimated Centres: {district['estimated_centres_needed']:.0f}")
        print(f"      - Peak Centres: {district['peak_centres_needed']:.0f}")
    
    # Detect surges
    print("\n5. Detecting enrolment surges...")
    surges = analyzer.detect_surges(surge_threshold=2.0)
    
    if surges:
        print(f"\n   Detected {len(surges)} surge events (showing top 5):")
        for i, surge in enumerate(surges[:5], 1):
            print(f"   {i}. {surge['district']} on {surge['date']}")
            print(f"      Excess: +{surge['excess_percentage']:.1f}% above expected")
    
    # Get district-specific summary
    print("\n6. Detailed summary for Mumbai...")
    summary = analyzer.get_district_summary('Mumbai')
    
    print(f"\n   Mumbai District Summary:")
    print(f"   - Total Enrolments: {summary['total_enrolments']:,}")
    print(f"   - Average Daily: {summary['avg_daily_enrolments']:.0f}")
    print(f"   - Recent 7-day Avg: {summary['recent_7day_avg']:.0f}")
    print(f"   - Trend: {summary['trend_percentage']:+.1f}%")
    print(f"   - Pressure Status: {summary['pressure_status']}")
    
    # Create visualizations
    print("\n7. Creating visualizations...")
    visualizer = EnrolmentVisualizer()
    
    # District comparison
    visualizer.plot_district_comparison(
        metrics, 
        metric='avg_daily_enrolments',
        save_path='outputs/district_comparison.png'
    )
    print("   - Saved district comparison plot")
    
    # Pressure score comparison
    metrics_with_pressure = capacity.copy()
    visualizer.plot_district_comparison(
        metrics_with_pressure,
        metric='estimated_centres_needed',
        save_path='outputs/capacity_requirements.png'
    )
    print("   - Saved capacity requirements plot")
    
    print("\n" + "=" * 60)
    print("Analysis complete! Check 'outputs/' directory for visualizations.")
    print("=" * 60)


if __name__ == "__main__":
    # Create outputs directory
    os.makedirs('outputs', exist_ok=True)
    main()
