"""
Unit Tests for District Pressure Analysis
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from analysis.district_pressure import DistrictPressureAnalyzer
from utils.data_utils import generate_sample_data


class TestDistrictPressureAnalyzer:
    """Test cases for DistrictPressureAnalyzer"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing"""
        return generate_sample_data(
            districts=['Mumbai', 'Delhi', 'Bangalore'],
            states=['Maharashtra', 'Delhi', 'Karnataka'],
            start_date='2024-01-01',
            end_date='2024-12-31',
            base_enrolments=100
        )
    
    def test_initialization(self, sample_data):
        """Test analyzer initialization"""
        analyzer = DistrictPressureAnalyzer(sample_data)
        assert analyzer.data is not None
        assert len(analyzer.data) > 0
    
    def test_calculate_district_metrics(self, sample_data):
        """Test district metrics calculation"""
        analyzer = DistrictPressureAnalyzer(sample_data)
        metrics = analyzer.calculate_district_metrics()
        
        assert isinstance(metrics, pd.DataFrame)
        assert len(metrics) == 3  # Three districts
        
        required_cols = ['district', 'state', 'total_enrolments', 
                        'avg_daily_enrolments', 'max_daily_enrolments']
        for col in required_cols:
            assert col in metrics.columns
    
    def test_identify_high_pressure_districts(self, sample_data):
        """Test high pressure district identification"""
        analyzer = DistrictPressureAnalyzer(sample_data)
        high_pressure = analyzer.identify_high_pressure_districts(threshold_percentile=50)
        
        assert isinstance(high_pressure, list)
        assert len(high_pressure) > 0
        
        for district in high_pressure:
            assert 'district' in district
            assert 'pressure_score' in district
    
    def test_calculate_capacity_utilization(self, sample_data):
        """Test capacity utilization calculation"""
        analyzer = DistrictPressureAnalyzer(sample_data)
        capacity = analyzer.calculate_capacity_utilization(capacity_per_centre=100)
        
        assert isinstance(capacity, pd.DataFrame)
        assert 'estimated_centres_needed' in capacity.columns
        assert 'peak_centres_needed' in capacity.columns
    
    def test_detect_surges(self, sample_data):
        """Test surge detection"""
        analyzer = DistrictPressureAnalyzer(sample_data)
        surges = analyzer.detect_surges(surge_threshold=2.0)
        
        assert isinstance(surges, list)
        for surge in surges:
            assert 'district' in surge
            assert 'date' in surge
            assert 'enrolments' in surge
            assert 'excess_percentage' in surge
    
    def test_compare_districts(self, sample_data):
        """Test district comparison"""
        analyzer = DistrictPressureAnalyzer(sample_data)
        comparison = analyzer.compare_districts(metric='avg_daily_enrolments')
        
        assert isinstance(comparison, pd.DataFrame)
        assert len(comparison) == 3
        assert comparison['avg_daily_enrolments'].is_monotonic_decreasing
    
    def test_get_district_summary(self, sample_data):
        """Test district summary generation"""
        analyzer = DistrictPressureAnalyzer(sample_data)
        summary = analyzer.get_district_summary('Mumbai')
        
        assert isinstance(summary, dict)
        assert 'district' in summary
        assert summary['district'] == 'Mumbai'
        assert 'total_enrolments' in summary
        assert 'avg_daily_enrolments' in summary
    
    def test_invalid_district(self, sample_data):
        """Test with invalid district name"""
        analyzer = DistrictPressureAnalyzer(sample_data)
        summary = analyzer.get_district_summary('NonExistent')
        
        assert 'error' in summary
