"""
Predictive Demand Indicators Module
Predicts future enrolment demand using statistical models
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime, timedelta


class DemandPredictor:
    """Predicts future enrolment demand using various models"""
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize with historical enrolment data
        
        Args:
            data: DataFrame with 'date' and 'enrolments' columns
        """
        self.data = data.copy()
        self._prepare_data()
        self.models = {}
    
    def _prepare_data(self):
        """Prepare data for prediction models"""
        if not pd.api.types.is_datetime64_any_dtype(self.data['date']):
            self.data['date'] = pd.to_datetime(self.data['date'])
        
        self.data = self.data.sort_values('date')
        
        # Add time-based features
        self.data['year'] = self.data['date'].dt.year
        self.data['month'] = self.data['date'].dt.month
        self.data['day'] = self.data['date'].dt.day
        self.data['day_of_week'] = self.data['date'].dt.dayofweek
        self.data['day_of_year'] = self.data['date'].dt.dayofyear
        self.data['quarter'] = self.data['date'].dt.quarter
        
        # Create ordinal date for time series
        self.data['date_ordinal'] = self.data['date'].map(datetime.toordinal)
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features for prediction models
        
        Args:
            df: Input DataFrame
        
        Returns:
            DataFrame with additional features
        """
        df = df.copy()
        
        # Lagged features
        for lag in [1, 7, 30]:
            df[f'enrolments_lag_{lag}'] = df['enrolments'].shift(lag)
        
        # Rolling statistics
        for window in [7, 30]:
            df[f'enrolments_rolling_mean_{window}'] = df['enrolments'].rolling(
                window=window, min_periods=1
            ).mean()
            df[f'enrolments_rolling_std_{window}'] = df['enrolments'].rolling(
                window=window, min_periods=1
            ).std()
        
        # Trend feature
        df['trend'] = range(len(df))
        
        return df
    
    def train_linear_model(self) -> Dict:
        """
        Train a linear regression model
        
        Returns:
            Dictionary with model performance metrics
        """
        # Prepare features
        data_with_features = self.create_features(self.data)
        data_with_features = data_with_features.dropna()
        
        if len(data_with_features) < 10:
            return {'error': 'Insufficient data for model training'}
        
        # Split features and target
        feature_cols = ['year', 'month', 'day_of_week', 'quarter', 'trend',
                       'enrolments_lag_1', 'enrolments_rolling_mean_7']
        X = data_with_features[feature_cols].fillna(0)
        y = data_with_features['enrolments']
        
        # Train model
        model = LinearRegression()
        model.fit(X, y)
        
        # Calculate performance
        train_score = model.score(X, y)
        predictions = model.predict(X)
        mae = np.mean(np.abs(y - predictions))
        
        self.models['linear'] = {
            'model': model,
            'features': feature_cols,
            'score': train_score
        }
        
        return {
            'model_type': 'Linear Regression',
            'r2_score': float(train_score),
            'mae': float(mae),
            'trained': True
        }
    
    def predict_next_period(self, days: int = 30, model_type: str = 'linear') -> pd.DataFrame:
        """
        Predict enrolments for next period
        
        Args:
            days: Number of days to predict
            model_type: Type of model to use ('linear', 'ma', 'exponential')
        
        Returns:
            DataFrame with predictions
        """
        if model_type == 'linear' and 'linear' in self.models:
            return self._predict_with_linear(days)
        elif model_type == 'ma':
            return self._predict_with_moving_average(days)
        elif model_type == 'exponential':
            return self._predict_with_exponential(days)
        else:
            # Default to moving average
            return self._predict_with_moving_average(days)
    
    def _predict_with_linear(self, days: int) -> pd.DataFrame:
        """Predict using linear regression model"""
        model_info = self.models['linear']
        model = model_info['model']
        feature_cols = model_info['features']
        
        # Get last date and create future dates
        last_date = self.data['date'].max()
        future_dates = pd.date_range(
            start=last_date + timedelta(days=1),
            periods=days,
            freq='D'
        )
        
        # Create future DataFrame
        future_df = pd.DataFrame({'date': future_dates})
        future_df['year'] = future_df['date'].dt.year
        future_df['month'] = future_df['date'].dt.month
        future_df['day_of_week'] = future_df['date'].dt.dayofweek
        future_df['quarter'] = future_df['date'].dt.quarter
        
        # Use last known values for lagged features
        last_enrolment = self.data['enrolments'].iloc[-1]
        last_rolling_mean = self.data['enrolments'].tail(7).mean()
        
        future_df['enrolments_lag_1'] = last_enrolment
        future_df['enrolments_rolling_mean_7'] = last_rolling_mean
        future_df['trend'] = range(len(self.data), len(self.data) + days)
        
        # Make predictions
        X_future = future_df[feature_cols].fillna(0)
        predictions = model.predict(X_future)
        
        future_df['predicted_enrolments'] = np.maximum(predictions, 0)  # Ensure non-negative
        
        return future_df[['date', 'predicted_enrolments']]
    
    def _predict_with_moving_average(self, days: int, window: int = 30) -> pd.DataFrame:
        """Predict using moving average"""
        last_date = self.data['date'].max()
        avg_enrolments = self.data['enrolments'].tail(window).mean()
        
        future_dates = pd.date_range(
            start=last_date + timedelta(days=1),
            periods=days,
            freq='D'
        )
        
        return pd.DataFrame({
            'date': future_dates,
            'predicted_enrolments': avg_enrolments
        })
    
    def _predict_with_exponential(self, days: int, alpha: float = 0.3) -> pd.DataFrame:
        """Predict using exponential smoothing"""
        last_date = self.data['date'].max()
        
        # Calculate exponentially weighted moving average
        ewm = self.data['enrolments'].ewm(alpha=alpha).mean()
        base_prediction = ewm.iloc[-1]
        
        future_dates = pd.date_range(
            start=last_date + timedelta(days=1),
            periods=days,
            freq='D'
        )
        
        return pd.DataFrame({
            'date': future_dates,
            'predicted_enrolments': base_prediction
        })
    
    def calculate_demand_indicators(self) -> Dict:
        """
        Calculate key demand indicators
        
        Returns:
            Dictionary with demand indicators
        """
        # Calculate growth rate
        recent_avg = self.data['enrolments'].tail(30).mean()
        older_avg = self.data['enrolments'].head(30).mean()
        growth_rate = ((recent_avg - older_avg) / older_avg * 100) if older_avg > 0 else 0
        
        # Calculate volatility
        volatility = self.data['enrolments'].std() / self.data['enrolments'].mean()
        
        # Predict next 30 days
        predictions_30d = self._predict_with_moving_average(30)
        predicted_total = predictions_30d['predicted_enrolments'].sum()
        
        # Calculate capacity requirements
        daily_avg = self.data['enrolments'].mean()
        peak_daily = self.data['enrolments'].max()
        
        return {
            'current_30d_avg': float(recent_avg),
            'historical_avg': float(older_avg),
            'growth_rate_percentage': float(growth_rate),
            'volatility_coefficient': float(volatility),
            'predicted_30d_total': float(predicted_total),
            'predicted_daily_avg': float(predicted_total / 30),
            'current_daily_avg': float(daily_avg),
            'peak_daily': int(peak_daily),
            'trend': 'Increasing' if growth_rate > 5 else 'Stable' if growth_rate > -5 else 'Decreasing'
        }
    
    def identify_peak_demand_periods(self, future_days: int = 90) -> List[Dict]:
        """
        Identify predicted peak demand periods
        
        Args:
            future_days: Days to predict ahead
        
        Returns:
            List of peak demand periods
        """
        # Use historical patterns to predict peaks
        monthly_avg = self.data.groupby(
            self.data['date'].dt.month
        )['enrolments'].mean()
        
        # Generate future months
        last_date = self.data['date'].max()
        future_dates = pd.date_range(
            start=last_date + timedelta(days=1),
            periods=future_days,
            freq='D'
        )
        
        # Map historical monthly averages to future dates
        future_months = pd.Series([d.month for d in future_dates])
        predicted_enrolments = future_months.map(monthly_avg)
        
        # Find peaks (top 20% of predicted values)
        threshold = predicted_enrolments.quantile(0.8)
        peak_indices = predicted_enrolments[predicted_enrolments >= threshold].index
        
        peaks = []
        for idx in peak_indices:
            peaks.append({
                'date': future_dates[idx].strftime('%Y-%m-%d'),
                'month': future_dates[idx].month,
                'predicted_enrolments': float(predicted_enrolments[idx]),
                'is_peak': True
            })
        
        return peaks
