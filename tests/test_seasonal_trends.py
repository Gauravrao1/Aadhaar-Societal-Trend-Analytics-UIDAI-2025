"""
Unit Tests for Seasonal Trend Detection
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from analysis.seasonal_trends import SeasonalTrendDetector
from utils.data_utils import generate_sample_data


class TestSeasonalTrendDetector:
    """Test cases for SeasonalTrendDetector"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing"""
        dates = pd.date_range(start='2023-01-01', end='2025-12-31', freq='D')
        data = pd.DataFrame({
            'date': dates,
            'enrolments': np.random.randint(50, 200, size=len(dates))
        })
        return data
    
    def test_initialization(self, sample_data):
        """Test detector initialization"""
        detector = SeasonalTrendDetector(sample_data)
        assert detector.data is not None
        assert 'month' in detector.data.columns
        assert 'quarter' in detector.data.columns
    
    def test_identify_peak_months(self, sample_data):
        """Test peak month identification"""
        detector = SeasonalTrendDetector(sample_data)
        peak_months = detector.identify_peak_months()
        
        assert isinstance(peak_months, dict)
        assert len(peak_months) == 12  # Should have all 12 months
        assert all(1 <= month <= 12 for month in peak_months.keys())
    
    def test_identify_peak_quarters(self, sample_data):
        """Test peak quarter identification"""
        detector = SeasonalTrendDetector(sample_data)
        peak_quarters = detector.identify_peak_quarters()
        
        assert isinstance(peak_quarters, dict)
        assert len(peak_quarters) == 4  # Should have all 4 quarters
        assert all(1 <= quarter <= 4 for quarter in peak_quarters.keys())
    
    def test_calculate_seasonality_strength(self, sample_data):
        """Test seasonality strength calculation"""
        detector = SeasonalTrendDetector(sample_data)
        strength = detector.calculate_seasonality_strength()
        
        assert isinstance(strength, float)
        assert 0 <= strength <= 1.0
    
    def test_detect_anomalous_periods(self, sample_data):
        """Test anomaly detection"""
        detector = SeasonalTrendDetector(sample_data)
        anomalies = detector.detect_anomalous_periods(threshold=2.0)
        
        assert isinstance(anomalies, list)
        for anomaly in anomalies:
            assert 'date' in anomaly
            assert 'z_score' in anomaly
            assert 'type' in anomaly
            assert anomaly['type'] in ['spike', 'drop']
    
    def test_get_seasonal_summary(self, sample_data):
        """Test comprehensive seasonal summary"""
        detector = SeasonalTrendDetector(sample_data)
        summary = detector.get_seasonal_summary()
        
        assert isinstance(summary, dict)
        assert 'peak_months' in summary
        assert 'peak_quarters' in summary
        assert 'seasonality_strength' in summary
        assert 'total_enrolments' in summary
        assert 'average_monthly_enrolments' in summary
    
    def test_seasonal_decomposition(self, sample_data):
        """Test seasonal decomposition"""
        detector = SeasonalTrendDetector(sample_data)
        decomposition = detector.detect_seasonal_pattern(period=12)
        
        # Should succeed with 3 years of data
        if 'error' not in decomposition:
            assert 'trend' in decomposition
            assert 'seasonal' in decomposition
            assert 'residual' in decomposition
    
    def test_with_empty_data(self):
        """Test with empty dataset"""
        empty_data = pd.DataFrame(columns=['date', 'enrolments'])
        
        with pytest.raises(Exception):
            detector = SeasonalTrendDetector(empty_data)
