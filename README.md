# 📊 QuantEye: Cryptocurrency Portfolio Optimization

QuantEye is a data-driven cryptocurrency portfolio optimization platform designed for institutional investors. It integrates structured OHLCV data and unstructured news sentiment to generate weekly rebalanced portfolios. This repository contains the codebase for two main pipeline stages and a standalone sentiment scoring module:

- `Station 1`: Full data pipeline (ETL + Feature Engineering + Sentiment)
- `Station 3`: Portfolio optimization and backtesting
- (Optional) `run_sentiment_analysis.py`: Standalone sentiment scoring utility

---

## 📁 Project Structure

```
.
├── Station1.py                # Combined implementation of Stage 1 (ETL) and Stage 2 (Feature Engineering)
├── run_station1.py           # Runs Station 1 (structured + unstructured ETL + feature construction)
├── run_sentiment_analysis.py # Optional standalone sentiment analysis module
├── station3.py               # Portfolio optimization and risk control
├── run_station3.py           # Run Station 3 (optimize + backtest)
├── results/                  # Output directory
└── README.md                 # This documentation
```

---

## 🧱 Station 1 – Unified ETL & Feature Engineering

### Purpose

Executes both Stage 1 and Stage 2:
- Extracts & cleans OHLCV and news data
- Performs sentiment scoring with VADER
- Constructs all required features (momentum, volatility, liquidity, sentiment)

### Key Functions

- **Structured Data**: Top 200 cryptos from CryptoCompare (OHLCV, volume)
- **Unstructured Data**: CoinDesk API (headline + body) over a rolling 180-day window
- **Sentiment**: VADER NLP with crypto-specific lexicon
- **Feature Engineering**:
  - Momentum, Reversal
  - Volatility, VaR, Drawdown
  - Volume ratio, Amihud illiquidity
  - Sentiment scores (compound, pos_ratio, news_count)
- **Output**:
  - `results/station2/station2_feature_matrix.csv`: Final modeling input
  - Intermediate results in `results/crypto/data` and `results/news/data`

### Run Command

```bash
python run_station1.py
```

Optional flags:
- `SKIP_CRYPTO`, `SKIP_NEWS`, `SKIP_STAGE2`

---

## 🗞 Optional: Sentiment-Only Runner

`run_sentiment_analysis.py` is a modular utility for testing or debugging sentiment logic independently.

### What it does:

- Loads cleaned news
- Applies custom VADER scoring
- Aggregates to weekly token and macro sentiment indices

### Run Command

```bash
python run_sentiment_analysis.py
```

Outputs:
- `token_sentiment.csv`
- `macro_sentiment_index.csv`

⚠️ This module is already included inside Station 1. Use it only for development or modular testing.

---

## 📈 Station 3 – Portfolio Optimization & Backtesting

### Purpose

Uses the engineered feature matrix to build weekly optimized portfolios with volatility-aware controls.

### Strategy Logic

- Top-N asset selection by score
- Ledoit-Wolf shrinkage for covariance
- Mean-variance optimization
- Constraints: max weight, liquidity filter, drawdown limit
- Weekly rebalancing (W-WED)

### Run Command

```bash
python run_station3.py
```

### Config Parameters

Defined in `station3.py > Config`:
```python
GAMMA = 1.5
TOP_N = 8
LOOKBACK_WEEKS = 4
MAX_WEIGHT = 0.3
MIN_LIQUIDITY = 0.5
```

### Output

- `cumulative_returns.png`
- `drawdown.png`
- `allocation_heatmap.png`
- `ls_returns.csv`, `long_returns.csv`, `short_returns.csv`

---

## 📦 Requirements

Install all dependencies using:

```bash
pip install -r requirements.txt
```

Or ensure these libraries:

- `pandas`, `numpy`, `matplotlib`, `seaborn`
- `nltk` (with `vader_lexicon`)
- `sklearn`

---

## 🧠 NLP & Sentiment Notes

- VADER model extended with crypto terms
- T+1 sentiment aggregation prevents lookahead bias
- Fallback to MARKET_WIDE when token-specific sentiment is missing

---

## 🔬 Backtest Overview

Each week:
1. Filter tradable assets
2. Score via engineered features
3. Estimate risk, optimize weights
4. Generate portfolio, log metrics

---

## 🚀 Roadmap

- Live streaming integration (WebSocket)
- Model ensembling (e.g., Ridge + GBDT)
- Parameter tuning UI
- Streamlit-based front-end

