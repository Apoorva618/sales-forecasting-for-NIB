# NIB Sales Forecasting Model

A machine learning solution for accurate sales forecasting across multiple product lines in a low-tech environment. Built for National Industries for the Blind (NIB) as a capstone project.

## Problem

NIB needed a forecasting model that could:
- Predict sales across five product lines with varying seasonal patterns
- Run standalone without Python, cloud dependencies, or data science expertise
- Handle a 15-year historical dataset with multiple sub-product categories
- Align forecasts to NIB's fiscal calendar (offset from calendar by 3 months)

A bad forecast meant misaligned production planning for a nonprofit serving vulnerable populations.

## Key Challenges

1. **Seasonality varies by product line**: Textile and commodity products showed regular 12-month cycles, while service lines had irregular demand patterns.

2. **Structural anomaly**: COVID-19 caused 40-60% revenue swings that don't reflect normal business dynamics. Including this data would teach the model that catastrophic collapse is a plausible seasonal pattern.

3. **Deployment constraints**: The solution had to be a standalone executable with no external dependencies, suitable for non-technical users.

## Solution Approach

### Phase 1: Fiscal Calendar Alignment
Converted all dates from calendar months to fiscal months so the model learns seasonality on the correct business timeline.

```python
fiscal_month_num = (month_num + 3 - 1) % 12 + 1
fiscal_year = calendar_year + (1 if fiscal_month_num <= 3 else 0)
```

### Phase 2: Data Cleaning and Structural Decomposition
Applied Simple Exponential Smoothing with smoothing level 0.2 to reduce noise while preserving underlying trend and seasonality patterns.

### Phase 3: Seasonality Detection
Used STL decomposition on each product line separately to extract:
- Trend component
- Seasonal component (confirmed 12-month period via ACF analysis)
- Residual component

Checked ACF of residuals to confirm no significant autocorrelation remained.

### Phase 4: COVID-19 Removal
Split data into three periods. Trained a Holt-Winters model on pre-COVID data only and used it to forecast what sales would have been during COVID. Replaced actual COVID-period values with these counterfactual forecasts.

This prevents the model from learning that a 50% collapse is part of normal seasonality.

### Phase 5: 1D CNN Forecasting
Trained separate convolutional neural networks for each of 30 sub-product lines.

**Architecture:**
- Conv1D: 32 filters, kernel size 3 (learns 3-month local patterns)
- MaxPooling1D: pool size 2 (downsampling)
- Flatten layer
- Dense: 16 units with ReLU
- Dense: 1 unit (output)

**Key decisions:**
- Input sequence length: 12 months (one full fiscal year, one complete seasonal cycle)
- Lightweight architecture appropriate for monthly data with limited observations
- CNN over LSTM because STL already extracted global trend and seasonality; only local residual structure remains
- MinMax scaling fit on training data only (prevents leakage)

### Phase 6: Multi-Step Forecasting
Recursive prediction loop: generate one month, append to input window, drop oldest month, predict next. Produces 24-month forecasts.

Important: Error compounds over the horizon. Near-term forecasts (3-6 months) are operationalized directly. Longer horizons inform planning direction but require quarterly refresh.

## Results

MAPE (Mean Absolute Percentage Error): **4.67%** across all product lines

Packaged as standalone PyInstaller executable with no external dependencies.

## Dataset and Features

- 15 years of historical monthly sales data
- 5 Lines of Business (LoB): Textile, Niche, Commodity, Services, Military Resale
- 30+ sub-LoB categories
- Forecasts generated at both LoB and sub-LoB granularity

## Evaluation Metrics

Four metrics computed for each forecast:
- MAE: Interpretable absolute error
- RMSE: Highlights large misses
- MAPE: Percentage-based comparison across different scales
- R-squared: Variance explained

## Technical Stack

- Python 3.x
- pandas: data manipulation
- scikit-learn: scaling, train-test split, metrics
- statsmodels: STL decomposition, Holt-Winters, exponential smoothing
- TensorFlow/Keras: CNN model architecture and training
- NumPy: numerical operations
- PyInstaller: standalone executable packaging

## Deployment

Model packaged as standalone executable via PyInstaller. No Python installation, cloud access, or data science team required to run forecasts. Users can adjust parameters and generate 24-month projections through a simple interface.

## Key Methodological Decisions

1. **Fiscal calendar alignment first**: Training on misaligned time axes produces systematically shifted seasonal forecasts.

2. **STL with adaptive smoothing**: Iteratively calibrated seasonal smoothing parameter until residuals showed no significant autocorrelation. Prevents overfitting the seasonal component.

3. **Per-LoB decomposition**: Each product line has different seasonal amplitude; one universal decomposition would miss critical variation.

4. **COVID counterfactual**: More rigorous than simply removing the data. Holt-Winters projects what "normal" would have looked like, giving the model a cleaner signal.

5. **CNN over alternatives**: LSTM captures long-range dependencies, but those are already handled by STL. CNN's local sliding filters are ideal for residual structure and train much faster on 30 separate models.

6. **Input window equals seasonal period**: 12-month input contains exactly one full fiscal cycle, giving the model complete seasonal context.

## How to Use

1. Prepare your sales data in CSV format with columns: date, product_line, sales_value
2. Adjust fiscal month offset if needed (default: 3 months)
3. Run the model to generate 24-month forecasts
4. Review near-term forecasts (3-6 months) for immediate planning
5. Use longer forecasts directionally and refresh quarterly

## Future Enhancements

- Per-LoB COVID recovery date calibration instead of uniform cutoff
- Confidence intervals for forecast ranges
- Automated retraining on new data
- Interactive dashboard for stakeholders

## Acknowledgments

Built by Apoorva Goel, Atharva, Harshal, Ning, and Homer as an Arizona State University capstone project in collaboration with National Industries for the Blind.

## License

MIT

---

For detailed technical walkthrough, see the accompanying .py file
