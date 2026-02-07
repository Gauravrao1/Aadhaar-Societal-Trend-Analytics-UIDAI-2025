"""
Aadhaar Societal Trend Analytics
Data models for Aadhaar enrolment data
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Dict, List
import pandas as pd


@dataclass
class EnrolmentRecord:
    """Represents a single Aadhaar enrolment record"""
    district: str
    state: str
    date: datetime
    enrolments: int
    age_group: Optional[str] = None
    gender: Optional[str] = None
    centre_id: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convert record to dictionary"""
        return {
            'district': self.district,
            'state': self.state,
            'date': self.date,
            'enrolments': self.enrolments,
            'age_group': self.age_group,
            'gender': self.gender,
            'centre_id': self.centre_id
        }


class EnrolmentDataset:
    """Manages Aadhaar enrolment dataset"""
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize dataset with pandas DataFrame
        
        Args:
            data: DataFrame with columns: district, state, date, enrolments
        """
        self.data = data
        self._validate_data()
    
    def _validate_data(self):
        """Validate required columns exist"""
        required_cols = ['district', 'state', 'date', 'enrolments']
        missing = set(required_cols) - set(self.data.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        # Convert date to datetime if not already
        if not pd.api.types.is_datetime64_any_dtype(self.data['date']):
            self.data['date'] = pd.to_datetime(self.data['date'])
    
    def get_date_range(self) -> tuple:
        """Get min and max dates in dataset"""
        return self.data['date'].min(), self.data['date'].max()
    
    def get_districts(self) -> List[str]:
        """Get unique districts in dataset"""
        return sorted(self.data['district'].unique().tolist())
    
    def get_states(self) -> List[str]:
        """Get unique states in dataset"""
        return sorted(self.data['state'].unique().tolist())
    
    def filter_by_district(self, district: str) -> pd.DataFrame:
        """Filter data by district"""
        return self.data[self.data['district'] == district].copy()
    
    def filter_by_state(self, state: str) -> pd.DataFrame:
        """Filter data by state"""
        return self.data[self.data['state'] == state].copy()
    
    def filter_by_date_range(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Filter data by date range"""
        return self.data[
            (self.data['date'] >= start_date) & 
            (self.data['date'] <= end_date)
        ].copy()
    
    def aggregate_by_period(self, period: str = 'M') -> pd.DataFrame:
        """
        Aggregate enrolments by time period
        
        Args:
            period: Pandas period string ('D', 'W', 'M', 'Q', 'Y')
        
        Returns:
            DataFrame with aggregated enrolments
        """
        return self.data.groupby([
            pd.Grouper(key='date', freq=period),
            'district',
            'state'
        ])['enrolments'].sum().reset_index()
