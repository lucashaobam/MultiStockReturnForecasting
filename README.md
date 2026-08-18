# Multi-Stock Return Forecasting & Portfolio Management

A leakage-safe machine learning pipeline for forecasting next-day returns across a 6-stock Indian equity universe (RELIANCE, HDFCBANK, INFY, M&M, BHARTIARTL, HUL), combining technical, fundamental, macroeconomic, and news sentiment data, validated with nested walk-forward cross-validation, and used to construct a volatility-scaled long-short portfolio.

Built as a course project for the Data Science and AI in Finance course.

## Overview

The pipeline covers:

- **Data acquisition**: daily OHLCV (Yahoo Finance), macro indicators (USD/INR, India VIX, Brent crude, US 10Y yield, India 10Y G-Sec yield via FRED), RBI policy repo rate, CPI inflation, quarterly fundamentals, and news sentiment (FinBERT-scored headlines).
- **Feature engineering**: technical indicators (SMA/EMA, RSI, MACD, Bollinger Bands, volume features), lagged returns, and macro-derived features, all constructed to avoid look-ahead bias by design.
- **Feature selection**: domain-grouped mutual information pruning followed by Recursive Feature Elimination with L1 regularization, computed strictly on the pre-holdout training window.
- **Modeling**: LightGBM and XGBoost regressors (ensembled), plus a LightGBM classifier for direction, tuned via nested walk-forward cross-validation (an outer loop of independent held-out folds, each with its own inner hyperparameter search).
- **Portfolio construction**: confidence-weighted, inverse-volatility-scaled long-short positioning across all six stocks, benchmarked against an equal-weight buy-and-hold strategy over a held-out Oct-Dec 2025 forward-test window.

## Results

Individual-stock R2 is consistently at or below zero across every validation method used (forward-test holdout, per-ticker breakdown, and an independent nested walk-forward cross-validation), and directional accuracy sits close to 50%. This is consistent with the efficient-market view that daily equity returns are largely unpredictable from public technical, fundamental, and sentiment data at this horizon, and is treated in this project as a real, validated finding rather than a shortfall to be hidden.

The portfolio construction step outperformed an equal-weight benchmark in the specific forward-test window evaluated (Sharpe 4.33 vs. 3.20), but given the weak underlying per-stock signal and the short evaluation window (61 trading days), this should be read as a demonstration of the portfolio construction methodology rather than evidence of a validated, tradable strategy.

Full results, per-ticker breakdowns, and discussion are in the notebook.

## Limitations

- India 10-Year G-Sec yield is a monthly FRED series, forward-filled to daily.
- Quarterly fundamentals are joined using a 45-day reporting-lag assumption (SEBI's maximum disclosure window), not each company's actual verified disclosure date.
- News sentiment coverage is uneven across tickers and years; missing sentiment-days are filled as neutral (0) rather than dropped.
- The forward-test evaluation covers a single ~3-month window, too short to treat any single performance metric as a demonstrated, repeatable edge.
- The backtest does not model transaction costs, slippage, market impact, or correlation between the six stocks.

## Tech Stack

Python, pandas, scikit-learn, LightGBM, XGBoost, Optuna, feature-engine, yfinance, pandas-datareader, transformers (FinBERT)

## Running the Notebook

Open `MultiStockReturnForecasting.ipynb` in Google Colab (or any environment with internet access) and run all cells in order. Data fetching, feature engineering, model training, and portfolio evaluation are fully reproducible end to end. Expect the news sentiment fetching step to take several minutes on first run.

## License

MIT
