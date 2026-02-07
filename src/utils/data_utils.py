"""
Utility Functions
Helper functions for data processing and analysis
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import random


def generate_sample_data(
    districts: List[str],
    states: List[str],
    start_date: str = '2024-01-01',
    end_date: str = '2025-12-31',
    base_enrolments: int = 100
) -> pd.DataFrame:
    """
    Generate sample Aadhaar enrolment data for testing
    
    Args:
        districts: List of district names
        states: List of state names (should match districts)
        start_date: Start date for data generation
        end_date: End date for data generation
        base_enrolments: Base number of daily enrolments
    
    Returns:
        DataFrame with sample enrolment data
    """
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    data = []
    
    for district, state in zip(districts, states):
        for date in date_range:
            # Add seasonal variation
            month = date.month
            seasonal_factor = 1.0 + 0.3 * np.sin(2 * np.pi * month / 12)
            
            # Add random variation
            random_factor = random.uniform(0.7, 1.3)
            
            # Add day-of-week effect (lower on weekends)
            weekday_factor = 0.7 if date.dayofweek >= 5 else 1.0
            
            enrolments = int(base_enrolments * seasonal_factor * random_factor * weekday_factor)
            
            data.append({
                'district': district,
                'state': state,
                'date': date,
                'enrolments': enrolments
            })
    
    return pd.DataFrame(data)


def load_data_from_csv(file_path: str) -> pd.DataFrame:
    """
    Load enrolment data from CSV file
    
    Args:
        file_path: Path to CSV file
    
    Returns:
        DataFrame with enrolment data
    """
    data = pd.read_csv(file_path)
    
    # Convert date column to datetime
    if 'date' in data.columns:
        data['date'] = pd.to_datetime(data['date'])
    
    return data


def save_data_to_csv(data: pd.DataFrame, file_path: str):
    """
    Save enrolment data to CSV file
    
    Args:
        data: DataFrame with enrolment data
        file_path: Path to save CSV file
    """
    data.to_csv(file_path, index=False)


def aggregate_by_district(data: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate enrolment data by district
    
    Args:
        data: DataFrame with enrolment data
    
    Returns:
        DataFrame with district-level aggregates
    """
    return data.groupby(['district', 'state'])['enrolments'].sum().reset_index()


def aggregate_by_month(data: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate enrolment data by month
    
    Args:
        data: DataFrame with enrolment data
    
    Returns:
        DataFrame with monthly aggregates
    """
    data = data.copy()
    data['month'] = data['date'].dt.to_period('M')
    return data.groupby('month')['enrolments'].sum().reset_index()


def calculate_growth_rate(data: pd.DataFrame, period: str = 'M') -> pd.DataFrame:
    """
    Calculate period-over-period growth rate
    
    Args:
        data: DataFrame with 'date' and 'enrolments' columns
        period: Period for grouping ('D', 'W', 'M', 'Q', 'Y')
    
    Returns:
        DataFrame with growth rates
    """
    # Aggregate by period
    aggregated = data.groupby(
        pd.Grouper(key='date', freq=period)
    )['enrolments'].sum().reset_index()
    
    # Calculate growth rate
    aggregated['growth_rate'] = aggregated['enrolments'].pct_change() * 100
    
    return aggregated


def detect_outliers(data: pd.DataFrame, column: str = 'enrolments',
                   method: str = 'iqr', threshold: float = 1.5) -> pd.DataFrame:
    """
    Detect outliers in data
    
    Args:
        data: DataFrame with data
        column: Column to check for outliers
        method: Method to use ('iqr' or 'zscore')
        threshold: Threshold for outlier detection
    
    Returns:
        DataFrame with outliers marked
    """
    data = data.copy()
    
    if method == 'iqr':
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        
        data['is_outlier'] = (data[column] < lower_bound) | (data[column] > upper_bound)
    
    elif method == 'zscore':
        mean = data[column].mean()
        std = data[column].std()
        
        data['z_score'] = (data[column] - mean) / std
        data['is_outlier'] = np.abs(data['z_score']) > threshold
    
    return data


def get_summary_statistics(data: pd.DataFrame, column: str = 'enrolments') -> Dict:
    """
    Calculate summary statistics for a column
    
    Args:
        data: DataFrame with data
        column: Column to summarize
    
    Returns:
        Dictionary with summary statistics
    """
    return {
        'count': int(data[column].count()),
        'mean': float(data[column].mean()),
        'median': float(data[column].median()),
        'std': float(data[column].std()),
        'min': float(data[column].min()),
        'max': float(data[column].max()),
        'q25': float(data[column].quantile(0.25)),
        'q75': float(data[column].quantile(0.75))
    }


def filter_date_range(data: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Filter data by date range
    
    Args:
        data: DataFrame with 'date' column
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
    
    Returns:
        Filtered DataFrame
    """
    data = data.copy()
    if not pd.api.types.is_datetime64_any_dtype(data['date']):
        data['date'] = pd.to_datetime(data['date'])
    
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    
    return data[(data['date'] >= start) & (data['date'] <= end)]


def resample_data(data: pd.DataFrame, frequency: str = 'W') -> pd.DataFrame:
    """
    Resample time series data to different frequency
    
    Args:
        data: DataFrame with 'date' and 'enrolments' columns
        frequency: Resampling frequency ('D', 'W', 'M', 'Q', 'Y')
    
    Returns:
        Resampled DataFrame
    """
    data = data.copy()
    data = data.set_index('date')
    resampled = data.resample(frequency)['enrolments'].sum().reset_index()
    return resampled
