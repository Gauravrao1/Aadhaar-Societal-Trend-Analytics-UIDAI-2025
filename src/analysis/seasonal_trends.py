"""
Seasonal Trend Detection Module
Analyzes seasonal patterns in Aadhaar enrolment data
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, List
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy import stats


class SeasonalTrendDetector:
    """Detects and analyzes seasonal trends in enrolment data"""
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize with enrolment data
        
        Args:
            data: DataFrame with 'date' and 'enrolments' columns
        """
        self.data = data.copy()
        self.data = self.data.sort_values('date')
        self._prepare_data()
    
    def _prepare_data(self):
        """Prepare data for seasonal analysis"""
        # Ensure date is datetime
        if not pd.api.types.is_datetime64_any_dtype(self.data['date']):
            self.data['date'] = pd.to_datetime(self.data['date'])
        
        # Add time-based features
        self.data['month'] = self.data['date'].dt.month
        self.data['quarter'] = self.data['date'].dt.quarter
        self.data['year'] = self.data['date'].dt.year
        self.data['week'] = self.data['date'].dt.isocalendar().week
        self.data['day_of_year'] = self.data['date'].dt.dayofyear
    
    def detect_seasonal_pattern(self, period: int = 12) -> Dict:
        """
        Detect seasonal patterns using time series decomposition
        
        Args:
            period: Seasonality period (default: 12 for monthly data)
        
        Returns:
            Dictionary with trend, seasonal, and residual components
        """
        # Aggregate by month for consistency
        monthly_data = self.data.groupby(
            pd.Grouper(key='date', freq='M')
        )['enrolments'].sum()
        
        # Need at least 2 complete cycles
        if len(monthly_data) < 2 * period:
            return {
                'error': 'Insufficient data for seasonal decomposition',
                'required_periods': 2 * period,
                'available_periods': len(monthly_data)
            }
        
        # Perform seasonal decomposition
        decomposition = seasonal_decompose(
            monthly_data, 
            model='additive', 
            period=period,
            extrapolate_trend='freq'
        )
        
        return {
            'trend': decomposition.trend.to_dict(),
            'seasonal': decomposition.seasonal.to_dict(),
            'residual': decomposition.resid.to_dict(),
            'observed': decomposition.observed.to_dict()
        }
    
    def identify_peak_months(self) -> Dict[int, float]:
        """
        Identify months with highest average enrolments
        
        Returns:
            Dictionary mapping month number to average enrolments
        """
        monthly_avg = self.data.groupby('month')['enrolments'].mean()
        return monthly_avg.sort_values(ascending=False).to_dict()
    
    def identify_peak_quarters(self) -> Dict[int, float]:
        """
        Identify quarters with highest average enrolments
        
        Returns:
            Dictionary mapping quarter number to average enrolments
        """
        quarterly_avg = self.data.groupby('quarter')['enrolments'].mean()
        return quarterly_avg.sort_values(ascending=False).to_dict()
    
    def calculate_seasonality_strength(self) -> float:
        """
        Calculate strength of seasonality using coefficient of variation
        
        Returns:
            Seasonality strength score (0-1, higher = stronger seasonality)
        """
        monthly_avg = self.data.groupby('month')['enrolments'].mean()
        cv = monthly_avg.std() / monthly_avg.mean()
        # Normalize to 0-1 scale
        return min(cv, 1.0)
    
    def detect_anomalous_periods(self, threshold: float = 2.0) -> List[Dict]:
        """
        Detect periods with anomalous enrolment patterns
        
        Args:
            threshold: Number of standard deviations for anomaly detection
        
        Returns:
            List of anomalous periods with details
        """
        # Calculate monthly totals
        monthly = self.data.groupby(
            pd.Grouper(key='date', freq='M')
        )['enrolments'].sum().reset_index()
        
        # Calculate z-scores
        mean_enrolments = monthly['enrolments'].mean()
        std_enrolments = monthly['enrolments'].std()
        monthly['z_score'] = (monthly['enrolments'] - mean_enrolments) / std_enrolments
        
        # Identify anomalies
        anomalies = monthly[np.abs(monthly['z_score']) > threshold]
        
        return [
            {
                'date': row['date'].strftime('%Y-%m'),
                'enrolments': int(row['enrolments']),
                'z_score': float(row['z_score']),
                'type': 'spike' if row['z_score'] > 0 else 'drop'
            }
            for _, row in anomalies.iterrows()
        ]
    
    def get_seasonal_summary(self) -> Dict:
        """
        Get comprehensive seasonal trend summary
        
        Returns:
            Dictionary with seasonal insights
        """
        peak_months = self.identify_peak_months()
        peak_quarters = self.identify_peak_quarters()
        seasonality_strength = self.calculate_seasonality_strength()
        anomalies = self.detect_anomalous_periods()
        
        return {
            'peak_months': peak_months,
            'peak_quarters': peak_quarters,
            'seasonality_strength': seasonality_strength,
            'anomalous_periods': anomalies,
            'total_enrolments': int(self.data['enrolments'].sum()),
            'average_monthly_enrolments': float(
                self.data.groupby(
                    pd.Grouper(key='date', freq='M')
                )['enrolments'].sum().mean()
            )
        }
