"""
Unit Tests for Predictive Demand Models
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from analysis.predictive_demand import DemandPredictor
from utils.data_utils import generate_sample_data


class TestDemandPredictor:
    """Test cases for DemandPredictor"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing"""
        return generate_sample_data(
            districts=['Mumbai'],
            states=['Maharashtra'],
            start_date='2023-01-01',
            end_date='2025-12-31',
            base_enrolments=150
        )
    
    def test_initialization(self, sample_data):
        """Test predictor initialization"""
        predictor = DemandPredictor(sample_data)
        assert predictor.data is not None
        assert 'year' in predictor.data.columns
        assert 'month' in predictor.data.columns
        assert 'date_ordinal' in predictor.data.columns
    
    def test_train_linear_model(self, sample_data):
        """Test linear model training"""
        aggregated = sample_data.groupby('date')['enrolments'].sum().reset_index()
        predictor = DemandPredictor(aggregated)
        
        performance = predictor.train_linear_model()
        
        if 'error' not in performance:
            assert 'model_type' in performance
            assert 'r2_score' in performance
            assert 'mae' in performance
            assert performance['trained'] == True
    
    def test_predict_next_period_ma(self, sample_data):
        """Test moving average prediction"""
        aggregated = sample_data.groupby('date')['enrolments'].sum().reset_index()
        predictor = DemandPredictor(aggregated)
        
        predictions = predictor.predict_next_period(days=30, model_type='ma')
        
        assert isinstance(predictions, pd.DataFrame)
        assert len(predictions) == 30
        assert 'date' in predictions.columns
        assert 'predicted_enrolments' in predictions.columns
        assert all(predictions['predicted_enrolments'] >= 0)
    
    def test_predict_next_period_exponential(self, sample_data):
        """Test exponential smoothing prediction"""
        aggregated = sample_data.groupby('date')['enrolments'].sum().reset_index()
        predictor = DemandPredictor(aggregated)
        
        predictions = predictor.predict_next_period(days=30, model_type='exponential')
        
        assert isinstance(predictions, pd.DataFrame)
        assert len(predictions) == 30
        assert 'predicted_enrolments' in predictions.columns
    
    def test_calculate_demand_indicators(self, sample_data):
        """Test demand indicators calculation"""
        aggregated = sample_data.groupby('date')['enrolments'].sum().reset_index()
        predictor = DemandPredictor(aggregated)
        
        indicators = predictor.calculate_demand_indicators()
        
        assert isinstance(indicators, dict)
        assert 'current_30d_avg' in indicators
        assert 'growth_rate_percentage' in indicators
        assert 'volatility_coefficient' in indicators
        assert 'trend' in indicators
        assert indicators['trend'] in ['Increasing', 'Stable', 'Decreasing']
    
    def test_identify_peak_demand_periods(self, sample_data):
        """Test peak demand period identification"""
        aggregated = sample_data.groupby('date')['enrolments'].sum().reset_index()
        predictor = DemandPredictor(aggregated)
        
        peaks = predictor.identify_peak_demand_periods(future_days=60)
        
        assert isinstance(peaks, list)
        for peak in peaks:
            assert 'date' in peak
            assert 'month' in peak
            assert 'predicted_enrolments' in peak
    
    def test_create_features(self, sample_data):
        """Test feature creation"""
        aggregated = sample_data.groupby('date')['enrolments'].sum().reset_index()
        predictor = DemandPredictor(aggregated)
        
        features = predictor.create_features(predictor.data)
        
        assert 'enrolments_lag_1' in features.columns
        assert 'enrolments_rolling_mean_7' in features.columns
        assert 'trend' in features.columns
