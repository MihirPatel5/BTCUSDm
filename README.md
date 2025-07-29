# BTCUSD Forex Prediction Model

## Project Overview
A comprehensive machine learning project for predicting BTCUSD forex movements using advanced ML techniques with 5-minute intervals and 15-30 minute prediction horizons. This project implements a complete pipeline from data collection to production deployment with continuous improvement.

## Key Specifications
- **Data Source**: MetaTrader 5 (primary), Yahoo Finance, Alpha Vantage, Binance API (alternatives)
- **Time Interval**: 5-minute candles (recommended over 1-minute for better signal-to-noise ratio)
- **Historical Data**: 3 years of BTCUSD data
- **Prediction Target**: Binary classification (up/down) with 15-30 minute horizon
- **Success Metrics**: 55-60% accuracy, Sharpe ratio >1.5, Max drawdown <15%

## Project Structure
```
iwill/
├── data/
│   ├── raw/                    # Raw OHLCV data from MT5/APIs
│   ├── processed/              # Cleaned and feature-engineered data
│   ├── external/               # Economic calendar, sentiment data
│   └── backtest/               # Backtesting datasets
├── src/
│   ├── data_collection/        # MT5, Yahoo, Alpha Vantage, Binance collectors
│   ├── preprocessing/          # Data cleaning, gap handling, outlier removal
│   ├── feature_engineering/    # Technical indicators, statistical features
│   ├── target_definition/      # Target variable creation
│   ├── models/                 # XGBoost, LSTM, ensemble implementations
│   ├── training/               # Training strategies, class imbalance handling
│   ├── evaluation/             # Financial metrics, classification metrics
│   ├── tuning/                 # Hyperparameter optimization
│   ├── backtesting/            # Historical simulation framework
│   ├── validation/             # Model validation and robustness testing
│   ├── deployment/             # Production deployment pipeline
│   ├── monitoring/             # Real-time monitoring and alerting
│   ├── improvement/            # Continuous improvement and retraining
│   ├── advanced_techniques/    # Advanced modeling techniques
│   └── utils/                  # Helper functions, data validation
├── notebooks/
│   ├── phase1_data_exploration/     # EDA and data quality analysis
│   ├── phase2_feature_engineering/  # Feature creation and selection
│   ├── phase3_model_development/    # Model training and comparison
│   ├── phase4_backtesting/          # Strategy backtesting
│   └── phase5_analysis/             # Results analysis and optimization
├── config/
│   ├── config.yaml             # Main configuration
│   ├── model_params.yaml       # Model hyperparameters
│   └── data_sources.yaml       # API configurations
├── tests/                      # Unit tests for all modules
├── docs/                       # Documentation and reports
├── mt5_integration/            # MetaTrader 5 Expert Advisors
├── requirements.txt            # Python dependencies
├── environment.yml             # Conda environment
└── main.py                     # Main execution pipeline
```

## Success Targets
- **Accuracy**: 55-60% (realistic and profitable)
- **Sharpe Ratio**: >1.5 (risk-adjusted returns)
- **Maximum Drawdown**: <15% (risk control)
- **Win Rate**: 45-55% (sustainable)
- **Profit Factor**: >1.3 (gross profit/loss ratio)

## Implementation Phases

### Phase 1: Data Collection & Preprocessing
- Implemented data collection from MetaTrader 5, Yahoo Finance, Alpha Vantage, and Binance APIs
- Developed comprehensive data preprocessing pipeline with cleaning, gap handling, and outlier removal
- Created data validation modules to ensure data quality

### Phase 2: Feature Engineering
- Implemented technical indicators (SMA, EMA, RSI, MACD, Bollinger Bands, ATR)
- Added statistical features (rolling means, standard deviations, correlations)
- Created time-based features (hour of day, day of week, etc.)
- Developed feature selection mechanisms

### Phase 3: Target Definition
- Defined binary classification target with 15-30 minute prediction horizon
- Implemented target creation with configurable horizons

### Phase 4: Model Development
- Implemented XGBoost baseline model
- Developed LSTM neural network model
- Created ensemble methods combining multiple models

### Phase 5: Training Strategy
- Implemented time-based splits to prevent data leakage
- Added class imbalance handling (SMOTE, undersampling, oversampling)
- Developed walk-forward validation framework
- Added temporal stability analysis

### Phase 6: Model Evaluation
- Implemented comprehensive classification metrics (accuracy, precision, recall, F1, ROC AUC)
- Added financial metrics (total/annualized return, volatility, Sharpe ratio, max drawdown, win rate, profit factor)
- Created visualization tools for equity curves, drawdowns, and confusion matrices

### Phase 7: Hyperparameter Tuning
- Implemented grid search and Bayesian optimization (Optuna) for XGBoost and LSTM models
- Added cross-validation with time series splits
- Created optimization result visualization

### Phase 8: Backtesting Framework
- Developed historical replay backtesting with realistic trading simulation
- Implemented position sizing, slippage, and transaction costs
- Added walk-forward backtesting capability
- Created performance metrics calculation and visualization

### Phase 9: Model Validation & Robustness
- Implemented out-of-sample validation
- Added temporal stability analysis
- Developed data drift detection
- Created stress scenario testing

### Phase 10: Production Deployment
- Built real-time BTCUSD prediction pipeline
- Integrated with MetaTrader 5 for live data and trade execution
- Added fallback to Yahoo Finance for data
- Implemented risk management rules
- Created performance logging and graceful shutdown

### Phase 11: Monitoring & Continuous Improvement
- Developed real-time monitoring for model performance
- Implemented data quality monitoring
- Added model drift detection
- Created automated retraining pipeline
- Built feature update mechanisms

### Phase 12: Advanced Techniques
- Implemented advanced ensemble methods (Random Forest, Gradient Boosting, SVM)
- Developed market regime detection using clustering
- Added feature importance analysis using SHAP
- Created dimensionality reduction techniques (PCA, t-SNE)

### Phase 13: Advanced Feature Engineering and Improved Training
- Implemented advanced feature engineering with volatility regimes, momentum divergence, support/resistance levels, order flow approximations, and fractal patterns
- Developed improved training strategy with multiple validation approaches (standard time series splits and expanding window splits)
- Created advanced ensemble training with multiple model comparison
- Added comprehensive model performance visualization and comparison

## Key Features

### Risk Management
- Position sizing based on account equity and volatility
- Stop-loss and take-profit mechanisms
- Maximum drawdown protection
- Position correlation monitoring

### Data Quality
- Automated data validation and cleaning
- Gap detection and handling
- Outlier detection and treatment
- Data freshness monitoring

### Model Robustness
- Walk-forward validation to prevent overfitting
- Temporal stability analysis
- Out-of-sample testing
- Stress testing under various market conditions
- Advanced ensemble methods for improved predictions

### Production Ready
- Real-time prediction pipeline
- Live trading integration with MetaTrader 5
- Comprehensive monitoring and alerting
- Automated model retraining
- Graceful error handling and recovery
- Advanced feature engineering for enhanced model performance

## Dependencies

Key libraries used in this project:
- pandas, numpy for data manipulation
- scikit-learn, xgboost, tensorflow for machine learning
- matplotlib, seaborn for visualization
- ta-lib for technical indicators
- imblearn for handling class imbalance
- optuna for hyperparameter tuning
- joblib for model persistence
- MetaTrader5 API for live trading
- yfinance for alternative data source

See `requirements.txt` for complete list of dependencies. Additional dependencies for advanced features include scikit-learn ensemble methods and support vector machines.

## Usage

To run the complete pipeline:

```bash
python src/main.py
```

To run specific phases, execute the corresponding modules directly:

```bash
# Data collection
python src/data_collection/mt5_collector.py

# Feature engineering
python src/feature_engineering/feature_engineer.py

# Model training
python src/models/xgboost_model.py

# Backtesting
python src/backtesting/backtester.py

# Deployment
python src/deployment/deployer.py
```

## Configuration

The project is configured through `config/config.yaml` which allows customization of:
- Data sources and collection parameters
- Feature engineering parameters
- Model hyperparameters
- Training strategies
- Risk management rules
- Monitoring thresholds

## Results

This implementation provides a complete, production-ready pipeline for BTCUSD prediction with:
- Real-time prediction capabilities
- Automated trading integration
- Comprehensive monitoring and alerting
- Continuous improvement mechanisms
- Advanced modeling techniques

The system is designed to be robust, maintainable, and extensible for future enhancements.
