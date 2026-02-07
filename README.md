# Aadhaar Societal Trend Analytics - UIDAI 2026

Data-driven analysis of Aadhaar enrolment patterns to detect seasonal trends, district-level pressure, and predictive demand indicators for proactive UIDAI governance.

## Overview

This project provides comprehensive analytics tools for analyzing Aadhaar enrolment data to support proactive governance and resource allocation decisions. The system includes:

- **Seasonal Trend Detection**: Identifies seasonal patterns and peak enrolment periods
- **District-Level Pressure Analysis**: Monitors capacity and pressure at district level
- **Predictive Demand Indicators**: Forecasts future enrolment demand using statistical models

## Features

### 1. Seasonal Trend Analysis
- Time series decomposition (trend, seasonal, residual components)
- Peak month and quarter identification
- Seasonality strength measurement
- Anomalous period detection

### 2. District Pressure Analysis
- District-level enrolment metrics calculation
- High-pressure district identification
- Capacity utilization analysis
- Surge detection and monitoring
- Inter-district comparison

### 3. Predictive Demand Modeling
- Multiple prediction models (Linear Regression, Moving Average, Exponential Smoothing)
- Demand growth rate analysis
- Peak period prediction
- Capacity requirement forecasting

### 4. Visualization
- Time series plots
- Seasonal decomposition charts
- District comparison graphs
- Monthly heatmaps
- Interactive dashboards

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/Gauravrao1/Aadhaar-Societal-Trend-Analytics-UIDAI-2026.git
cd Aadhaar-Societal-Trend-Analytics-UIDAI-2026
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Quick Start

Run the main analytics script:
```bash
python src/main.py
```

This will:
1. Generate sample data
2. Perform all analyses (seasonal, district pressure, predictive)
3. Create visualizations
4. Display comprehensive report

### Example Scripts

Three example scripts demonstrate different analysis modules:

#### 1. Seasonal Trend Analysis
```bash
python examples/seasonal_analysis_example.py
```

Demonstrates:
- Seasonal pattern detection
- Peak month identification
- Anomaly detection
- Time series decomposition

#### 2. District Pressure Analysis
```bash
python examples/district_analysis_example.py
```

Demonstrates:
- District metrics calculation
- High-pressure district identification
- Capacity requirement estimation
- Surge detection

#### 3. Predictive Demand Analysis
```bash
python examples/predictive_analysis_example.py
```

Demonstrates:
- Model training
- Future demand prediction
- Peak period identification
- Multiple prediction methods

### Using as a Library

```python
import pandas as pd
from src.main import AadhaarAnalytics
from src.utils.data_utils import generate_sample_data

# Generate or load your data
data = generate_sample_data(
    districts=['Mumbai', 'Delhi'],
    states=['Maharashtra', 'Delhi'],
    start_date='2024-01-01',
    end_date='2025-12-31'
)

# Initialize analytics
analytics = AadhaarAnalytics(data)

# Analyze seasonal trends
seasonal = analytics.analyze_seasonal_trends()
print(f"Seasonality Strength: {seasonal['seasonality_strength']}")

# Analyze district pressure
pressure = analytics.analyze_district_pressure()
high_pressure_districts = pressure['high_pressure_districts']

# Predict demand
predictions = analytics.predict_demand(days=30)
print(f"Predicted 30-day total: {predictions['demand_indicators']['predicted_30d_total']}")

# Generate comprehensive report
report = analytics.generate_comprehensive_report()

# Create visualizations
analytics.create_visualizations(output_dir='outputs')
```

## Project Structure

```
Aadhaar-Societal-Trend-Analytics-UIDAI-2026/
├── src/
│   ├── models/
│   │   └── enrolment_data.py      # Data models and dataset management
│   ├── analysis/
│   │   ├── seasonal_trends.py      # Seasonal trend detection
│   │   ├── district_pressure.py    # District pressure analysis
│   │   └── predictive_demand.py    # Demand prediction models
│   ├── visualization/
│   │   └── plots.py                # Visualization functions
│   ├── utils/
│   │   └── data_utils.py           # Utility functions
│   └── main.py                     # Main entry point
├── examples/
│   ├── seasonal_analysis_example.py
│   ├── district_analysis_example.py
│   └── predictive_analysis_example.py
├── tests/                          # Unit tests
├── data/
│   ├── raw/                        # Raw data files
│   └── processed/                  # Processed data files
├── docs/                           # Documentation
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## Data Format

The system expects enrolment data in the following format:

```csv
district,state,date,enrolments
Mumbai,Maharashtra,2024-01-01,150
Delhi,Delhi,2024-01-01,200
...
```

Required columns:
- `district`: District name
- `state`: State name
- `date`: Date (YYYY-MM-DD format)
- `enrolments`: Number of enrolments

Optional columns:
- `age_group`: Age group classification
- `gender`: Gender classification
- `centre_id`: Enrolment centre identifier

## Key Algorithms

### Seasonal Decomposition
Uses statsmodels' seasonal_decompose with additive model to separate:
- **Trend**: Long-term progression
- **Seasonal**: Repeating patterns
- **Residual**: Random variations

### District Pressure Score
Calculated using:
```
Pressure Score = (Current/Average * 50) + (CV * 30) + (Trend * 20)
```
Where:
- Current/Average: Current rolling average vs. historical average
- CV: Coefficient of variation (volatility)
- Trend: Recent trend percentage

### Demand Prediction
Three methods available:
1. **Linear Regression**: Uses time-based features and lagged values
2. **Moving Average**: Simple averaging of recent periods
3. **Exponential Smoothing**: Weighted moving average with decay

## Output

### Console Reports
- Data summary statistics
- Seasonal insights (peak months, quarters, anomalies)
- High-pressure districts with scores
- Demand predictions and growth rates

### Visualizations
- Time series plots (PNG format)
- Seasonal decomposition charts
- District comparison bar charts
- Monthly heatmaps
- Prediction plots
- Interactive dashboards (HTML)

All visualizations are saved to the `outputs/` directory.

## Testing

Run unit tests:
```bash
pytest tests/
```

## Dependencies

Core dependencies:
- pandas: Data manipulation
- numpy: Numerical computations
- matplotlib: Static visualizations
- seaborn: Statistical visualizations
- scikit-learn: Machine learning models
- statsmodels: Time series analysis
- plotly: Interactive visualizations

See `requirements.txt` for complete list with versions.

## Use Cases

1. **Capacity Planning**: Identify districts requiring additional enrolment centres
2. **Resource Allocation**: Predict peak periods for staff deployment
3. **Performance Monitoring**: Track district-level trends and anomalies
4. **Policy Decisions**: Data-driven insights for UIDAI governance
5. **Budget Planning**: Forecast future resource requirements

## Future Enhancements

- Real-time data integration
- Machine learning models (LSTM, Prophet)
- Geographic visualization with maps
- Multi-variate analysis
- API for integration with other systems
- Web-based dashboard

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

This project is for educational and analytical purposes.

## Contact

For questions or support, please open an issue on GitHub.

## Acknowledgments

Developed for UIDAI (Unique Identification Authority of India) 2026 analytics requirements.