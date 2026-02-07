# Technical Documentation

## Architecture Overview

The Aadhaar Societal Trend Analytics system follows a modular architecture with clear separation of concerns:

```
┌─────────────────────────────────────────────────┐
│           User Interface Layer                   │
│  (main.py, examples/, notebooks/)                │
└────────────────┬────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────┐
│         Analytics Engine Layer                   │
│  (AadhaarAnalytics - Unified Interface)          │
└────────────────┬────────────────────────────────┘
                 │
      ┌──────────┴──────────┬──────────────────┐
      │                     │                  │
┌─────▼──────┐    ┌─────────▼────┐   ┌────────▼─────────┐
│ Seasonal   │    │  District     │   │  Predictive      │
│ Trends     │    │  Pressure     │   │  Demand          │
└─────┬──────┘    └──────┬────────┘   └────────┬─────────┘
      │                  │                     │
      └──────────────────┴─────────────────────┘
                         │
            ┌────────────▼────────────┐
            │  Data Models Layer       │
            │  (EnrolmentDataset)      │
            └────────────┬─────────────┘
                         │
            ┌────────────▼─────────────┐
            │  Utilities & Viz Layer   │
            │  (data_utils, plots)     │
            └──────────────────────────┘
```

## Module Dependencies

### Core Analysis Modules

**seasonal_trends.py**
- Dependencies: pandas, numpy, statsmodels, scipy
- Purpose: Time series analysis and seasonal pattern detection
- Key Algorithms: STL decomposition, z-score anomaly detection

**district_pressure.py**
- Dependencies: pandas, numpy
- Purpose: District-level capacity and pressure monitoring
- Key Algorithms: Rolling statistics, pressure scoring, surge detection

**predictive_demand.py**
- Dependencies: pandas, numpy, scikit-learn
- Purpose: Future demand forecasting
- Key Algorithms: Linear regression, moving average, exponential smoothing

### Supporting Modules

**enrolment_data.py**
- Purpose: Data models and dataset management
- Features: Validation, filtering, aggregation

**plots.py**
- Dependencies: matplotlib, seaborn, plotly
- Purpose: Data visualization
- Features: Static and interactive charts

**data_utils.py**
- Purpose: Utility functions for data processing
- Features: Loading, saving, aggregation, outlier detection

## Data Flow

1. **Input**: CSV files or DataFrames with enrolment data
2. **Validation**: EnrolmentDataset validates required columns
3. **Analysis**: Three parallel analysis pipelines process data
4. **Aggregation**: Results combined in comprehensive report
5. **Output**: Reports, visualizations, predictions

## Key Algorithms

### 1. Seasonal Decomposition

Uses STL (Seasonal and Trend decomposition using Loess):
- **Trend**: Long-term progression (smoothed using LOESS)
- **Seasonal**: Repeating patterns (extracted via moving averages)
- **Residual**: Random variations (observed - trend - seasonal)

Implementation:
```python
decomposition = seasonal_decompose(
    monthly_data, 
    model='additive',
    period=12,
    extrapolate_trend='freq'
)
```

### 2. Pressure Score Calculation

Composite score based on three factors:

```
Pressure Score = (Current/Baseline × 50) + (Volatility × 30) + (Trend × 20)
```

Where:
- Current/Baseline: Ratio of current to historical average
- Volatility: Coefficient of variation
- Trend: Recent percentage change

### 3. Demand Prediction

Three complementary methods:

**Linear Regression:**
- Features: time, month, day_of_week, lagged values
- Target: enrolments
- Training: Ordinary Least Squares

**Moving Average:**
```
Prediction = Average(Last N days)
```

**Exponential Smoothing:**
```
Prediction = α × Latest + (1-α) × Previous_Prediction
```

### 4. Anomaly Detection

Z-score based method:
```
z_score = (value - mean) / std
anomaly = |z_score| > threshold
```

## Performance Considerations

### Data Size
- **Small** (<10K records): All analyses run in <1 second
- **Medium** (10K-100K records): 1-5 seconds per analysis
- **Large** (>100K records): Consider aggregation to weekly/monthly

### Memory Usage
- Base: ~50MB for module imports
- Per 10K records: ~5MB additional
- Visualizations: ~20MB per chart

### Optimization Tips
1. Aggregate to daily level if finer granularity not needed
2. Filter to specific districts for targeted analysis
3. Use appropriate rolling window sizes
4. Cache model training results

## Testing Strategy

### Unit Tests
- 23 test cases covering all core functionality
- Test fixtures use generated sample data
- Tests validate both correctness and error handling

### Test Coverage
- Seasonal trends: 8 tests
- District pressure: 8 tests
- Predictive demand: 7 tests

### Running Tests
```bash
pytest tests/ -v                    # All tests
pytest tests/test_seasonal_trends.py # Specific module
pytest tests/ --cov=src              # With coverage
```

## Extension Points

### Adding New Analysis Modules

1. Create new file in `src/analysis/`
2. Implement class with consistent interface
3. Add method to `AadhaarAnalytics` class
4. Add unit tests in `tests/`

Example:
```python
class CustomAnalyzer:
    def __init__(self, data: pd.DataFrame):
        self.data = data
    
    def analyze(self) -> Dict:
        # Your analysis logic
        return results
```

### Adding New Visualizations

1. Add method to `EnrolmentVisualizer` class
2. Follow existing pattern: accept data, return figure
3. Support save_path parameter

### Adding New Prediction Models

1. Add method to `DemandPredictor` class
2. Follow pattern: train, predict, evaluate
3. Return standardized format (date, predicted_enrolments)

## Configuration

### Adjustable Parameters

**Seasonal Analysis:**
- `period`: Seasonality period (default: 12)
- `threshold`: Anomaly detection threshold (default: 2.0)

**District Pressure:**
- `window_days`: Rolling window (default: 30)
- `threshold_percentile`: High pressure threshold (default: 75)
- `surge_threshold`: Surge detection multiplier (default: 2.0)
- `capacity_per_centre`: Centre capacity (default: 100)

**Demand Prediction:**
- `days`: Prediction horizon (default: 30)
- `model_type`: 'linear', 'ma', or 'exponential'
- `window`: Moving average window (default: 30)
- `alpha`: Exponential smoothing factor (default: 0.3)

## Deployment Considerations

### Production Setup

1. **Data Pipeline**: Set up automated data ingestion
2. **Scheduling**: Run analyses daily/weekly using cron
3. **Monitoring**: Track analysis execution and errors
4. **Storage**: Store results in database
5. **API**: Wrap in REST API for integration

### Scalability

For large-scale deployment:
- Use Dask for distributed processing
- Store preprocessed aggregates
- Cache frequently accessed results
- Use parallel processing for multiple districts

### Security

- Sanitize input data
- Validate file uploads
- Implement access controls
- Audit trail for analysis runs
- Secure API endpoints

## Troubleshooting Guide

### Common Issues

**Import Errors:**
- Verify all dependencies installed: `pip install -r requirements.txt`
- Check Python version: `python --version` (must be 3.8+)

**Memory Errors:**
- Reduce date range
- Aggregate to weekly/monthly
- Process districts in batches

**Poor Predictions:**
- Ensure sufficient historical data (2+ years)
- Check for data quality issues
- Validate against known patterns
- Try different models

**Visualization Errors:**
- Install matplotlib backend: `pip install PyQt5`
- Use Agg backend for headless: `matplotlib.use('Agg')`

## Version History

**v1.0.0** (2026-02-07)
- Initial release
- Three core analysis modules
- Comprehensive test coverage
- Example scripts and documentation

## Future Enhancements

### Planned Features
1. Real-time data streaming support
2. Advanced ML models (LSTM, Prophet)
3. Geographic visualization with maps
4. Multi-variate analysis
5. API server implementation
6. Web-based dashboard
7. Automated report generation
8. Alert system for anomalies

### Research Directions
1. Causal analysis of enrolment patterns
2. Policy impact assessment
3. Resource optimization algorithms
4. Cross-state pattern analysis
