"""
District-Level Pressure Analysis Module
Analyzes enrolment pressure and capacity at district level
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from datetime import datetime, timedelta


class DistrictPressureAnalyzer:
    """Analyzes district-level enrolment pressure and capacity"""
    
    def __init__(self, data: pd.DataFrame, population_data: pd.DataFrame = None):
        """
        Initialize with enrolment data
        
        Args:
            data: DataFrame with 'district', 'date', 'enrolments' columns
            population_data: Optional DataFrame with 'district' and 'population' columns
        """
        self.data = data.copy()
        self.population_data = population_data
        self._prepare_data()
    
    def _prepare_data(self):
        """Prepare data for analysis"""
        if not pd.api.types.is_datetime64_any_dtype(self.data['date']):
            self.data['date'] = pd.to_datetime(self.data['date'])
        
        self.data = self.data.sort_values(['district', 'date'])
    
    def calculate_district_metrics(self, window_days: int = 30) -> pd.DataFrame:
        """
        Calculate key metrics for each district
        
        Args:
            window_days: Rolling window for metrics calculation
        
        Returns:
            DataFrame with district-level metrics
        """
        metrics = []
        
        for district in self.data['district'].unique():
            district_data = self.data[self.data['district'] == district].copy()
            
            # Calculate various metrics
            total_enrolments = district_data['enrolments'].sum()
            avg_daily = district_data['enrolments'].mean()
            max_daily = district_data['enrolments'].max()
            std_daily = district_data['enrolments'].std()
            
            # Calculate rolling metrics
            district_data['rolling_avg'] = district_data['enrolments'].rolling(
                window=min(window_days, len(district_data)), 
                min_periods=1
            ).mean()
            
            current_avg = district_data['rolling_avg'].iloc[-1] if len(district_data) > 0 else 0
            
            # Calculate trend (recent vs overall average)
            recent_period = district_data.tail(window_days)
            recent_avg = recent_period['enrolments'].mean() if len(recent_period) > 0 else 0
            trend = ((recent_avg - avg_daily) / avg_daily * 100) if avg_daily > 0 else 0
            
            metrics.append({
                'district': district,
                'state': district_data['state'].iloc[0],
                'total_enrolments': int(total_enrolments),
                'avg_daily_enrolments': float(avg_daily),
                'max_daily_enrolments': int(max_daily),
                'std_daily_enrolments': float(std_daily),
                'current_rolling_avg': float(current_avg),
                'trend_percentage': float(trend),
                'coefficient_of_variation': float(std_daily / avg_daily) if avg_daily > 0 else 0
            })
        
        return pd.DataFrame(metrics)
    
    def identify_high_pressure_districts(self, threshold_percentile: float = 75) -> List[Dict]:
        """
        Identify districts experiencing high enrolment pressure
        
        Args:
            threshold_percentile: Percentile threshold for high pressure
        
        Returns:
            List of high-pressure districts with details
        """
        metrics = self.calculate_district_metrics()
        
        # Calculate pressure score based on multiple factors
        metrics['pressure_score'] = (
            metrics['current_rolling_avg'] / metrics['avg_daily_enrolments'] * 50 +
            metrics['coefficient_of_variation'] * 30 +
            (metrics['trend_percentage'] / 100) * 20
        )
        
        threshold = metrics['pressure_score'].quantile(threshold_percentile / 100)
        high_pressure = metrics[metrics['pressure_score'] >= threshold]
        
        return high_pressure.sort_values('pressure_score', ascending=False).to_dict('records')
    
    def calculate_capacity_utilization(self, capacity_per_centre: int = 100) -> pd.DataFrame:
        """
        Calculate capacity utilization for districts
        
        Args:
            capacity_per_centre: Daily capacity per enrolment centre
        
        Returns:
            DataFrame with capacity metrics
        """
        metrics = self.calculate_district_metrics()
        
        # Estimate required centres based on average daily enrolments
        metrics['estimated_centres_needed'] = np.ceil(
            metrics['avg_daily_enrolments'] / capacity_per_centre
        )
        
        metrics['peak_centres_needed'] = np.ceil(
            metrics['max_daily_enrolments'] / capacity_per_centre
        )
        
        metrics['utilization_variance'] = (
            (metrics['peak_centres_needed'] - metrics['estimated_centres_needed']) / 
            metrics['estimated_centres_needed'] * 100
        ).fillna(0)
        
        return metrics
    
    def detect_surges(self, surge_threshold: float = 2.0) -> List[Dict]:
        """
        Detect sudden surges in district enrolments
        
        Args:
            surge_threshold: Multiplier threshold for surge detection
        
        Returns:
            List of surge events
        """
        surges = []
        
        for district in self.data['district'].unique():
            district_data = self.data[self.data['district'] == district].copy()
            
            # Calculate rolling statistics
            district_data['rolling_mean'] = district_data['enrolments'].rolling(
                window=7, min_periods=1
            ).mean()
            district_data['rolling_std'] = district_data['enrolments'].rolling(
                window=7, min_periods=1
            ).std()
            
            # Detect surges
            district_data['surge'] = (
                district_data['enrolments'] > 
                district_data['rolling_mean'] + surge_threshold * district_data['rolling_std']
            )
            
            surge_days = district_data[district_data['surge']]
            
            for _, row in surge_days.iterrows():
                surges.append({
                    'district': district,
                    'state': row['state'],
                    'date': row['date'].strftime('%Y-%m-%d'),
                    'enrolments': int(row['enrolments']),
                    'expected': float(row['rolling_mean']),
                    'excess_percentage': float(
                        (row['enrolments'] - row['rolling_mean']) / row['rolling_mean'] * 100
                    )
                })
        
        return sorted(surges, key=lambda x: x['excess_percentage'], reverse=True)
    
    def compare_districts(self, metric: str = 'avg_daily_enrolments') -> pd.DataFrame:
        """
        Compare districts by a specific metric
        
        Args:
            metric: Metric to compare districts by
        
        Returns:
            DataFrame with district comparisons
        """
        metrics = self.calculate_district_metrics()
        return metrics.sort_values(metric, ascending=False)
    
    def get_district_summary(self, district: str) -> Dict:
        """
        Get comprehensive summary for a specific district
        
        Args:
            district: District name
        
        Returns:
            Dictionary with district insights
        """
        district_data = self.data[self.data['district'] == district].copy()
        
        if len(district_data) == 0:
            return {'error': f'No data found for district: {district}'}
        
        metrics = self.calculate_district_metrics()
        district_metrics = metrics[metrics['district'] == district].iloc[0]
        
        # Get recent trend
        recent_7days = district_data.tail(7)['enrolments'].mean()
        recent_30days = district_data.tail(30)['enrolments'].mean()
        
        return {
            'district': district,
            'state': district_metrics['state'],
            'total_enrolments': int(district_metrics['total_enrolments']),
            'avg_daily_enrolments': float(district_metrics['avg_daily_enrolments']),
            'recent_7day_avg': float(recent_7days),
            'recent_30day_avg': float(recent_30days),
            'trend_percentage': float(district_metrics['trend_percentage']),
            'pressure_status': 'High' if district_metrics['coefficient_of_variation'] > 0.5 else 'Normal',
            'data_points': len(district_data)
        }
