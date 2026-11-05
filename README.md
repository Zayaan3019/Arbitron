# Arbitron — Autonomous Quant Research Agent

> An LLM-powered agent that autonomously discovers, tests, and validates trading signals using the Claude API, walk-forward backtesting, Ledoit-Wolf portfolio optimization, and HMM regime detection.

---

## Overview

Arbitron is a full-stack quantitative research platform combining:

- **Autonomous Quant Research Lab** - Ranks factor and foundation-model candidates, audits them with deployment gates, and emits a research memo
- **Statistical Defense Layer** - Probabilistic Sharpe Ratio, Deflated Sharpe Ratio, Probability of Backtest Overfitting, drawdown gates, and regime robustness checks
- **Market Foundation Modeling** - Self-supervised market-state embeddings plus a walk-forward Ridge adapter that produces a cross-sectional alpha signal

- **Autonomous LLM Agent** — Llama 3.3 70B (via Groq free API) uses function calling in a ReAct loop to plan and execute research tasks end-to-end
- **Quant Factor Library** — 13 named factors (momentum, reversal, volatility, RSI, MACD, Bollinger, price acceleration, ...)
- **Walk-Forward Backtesting** — Purged and embargoed splits (López de Prado, 2018), quintile L/S portfolio with transaction costs
- **Portfolio Optimization** — Mean-Variance (SLSQP), Max Sharpe (analytic + SLSQP), Risk Parity (ERC), Ledoit-Wolf covariance shrinkage
- **Risk Analytics** — VaR/CVaR (historical, parametric, Monte Carlo), factor decomposition, MRC/CRC, Ulcer Index
- **Regime Detection** — Gaussian HMM on (return, log-vol) features; rule-based fallback (SMA + vol percentile)
- **ML Return Predictor** — Random Forest with rolling Sharpe / beta / regime features, walk-forward fitting
- **FastAPI Backend** — REST + WebSocket streaming for agent events
- **React Dashboard** — Agent chat, factor browser, backtest viewer, portfolio optimizer, risk dashboard

---

## Flagship Research Lab

The new `/api/lab/run` workflow turns Arbitron into an autonomous quant research lab:

1. Computes a research queue of candidate factors.
2. Builds an optional market-foundation signal from causal multi-scale price features.
3. Runs walk-forward quintile backtests with transaction costs.
4. Audits every candidate with PSR, DSR, PBO, drawdown, and sample-size gates.
5. Checks whether performance survives bull/bear regime splits.
6. Ranks candidates by a defense-aware research score.
7. Returns a champion, candidate queue, foundation diagnostics, equity curve, and research memo.

This is designed to avoid the usual student-project trap: picking the prettiest backtest. The lab explicitly asks whether the result survives multiple-testing, overfitting, non-normal returns, and market-regime instability.

---

## Architecture

```
arbitron/
├── arbitron/
│   ├── agent/
│   │   ├── loop.py         # ArbitronAgent — ReAct loop using Claude tool_use
│   │   └── tools.py        # 8 tool schemas + executor
│   ├── data/
│   │   └── market.py       # yfinance + Parquet cache, universe definitions
│   ├── factors/
│   │   ├── library.py      # 13 factors + FACTOR_REGISTRY
│   │   └── evaluation.py   # IC, IC-IR, Fama-MacBeth (Newey-West HAC)
│   ├── backtest/
│   │   ├── engine.py       # Walk-forward L/S quintile backtest
│   │   └── metrics.py      # Sharpe, Sortino, Calmar, CAGR, Omega Ratio, ...
│   ├── portfolio/
│   │   └── optimizer.py    # MVO, Max Sharpe, Risk Parity, efficient frontier
│   ├── risk/
│   │   └── analytics.py    # VaR, CVaR, MRC, CRC, factor beta, tail metrics
│   └── models/
│       ├── regime.py       # GaussianHMM + SimpleRegimeDetector
│       └── predictor.py    # RandomForest return predictor
├── api/
│   ├── main.py             # FastAPI app + WebSocket endpoint
│   ├── schemas.py          # Pydantic request/response models
│   └── routers/            # data, factors, backtest, portfolio, risk, agent
├── frontend/               # React + Vite + Tailwind dashboard
│   └── src/
│       ├── pages/          # Agent, Factors, Backtest, Portfolio, Risk
│       ├── api.ts           # Typed API client + WebSocket helper
│       └── store.ts         # Zustand global state
├── tests/                  # 84 tests (factors, backtest, portfolio, risk)
├── research.py             # CLI entry point
└── pyproject.toml
```

---

## Statistical Defense Layer

Arbitron now separates "looks good" from "survives audit":

| Defense | Why it matters |
|---------|----------------|
| Probabilistic Sharpe Ratio | Estimates whether Sharpe is statistically above a hurdle after skew/kurtosis adjustment |
| Deflated Sharpe Ratio | Raises the Sharpe hurdle when many strategy variants were tested |
| Probability of Backtest Overfitting | Uses CSCV-style train/test splits to measure if the selected champion collapses out of sample |
| Regime Robustness | Reports strategy behavior in bull and bear regimes |
| Drawdown Gate | Blocks high-Sharpe strategies that hide unacceptable path risk |

---

## Market Foundation Signal

The local foundation module does not require PyTorch or paid model APIs. It learns representations from:

- Multi-horizon asset returns
- Realized volatility and volatility ratios
- Rolling drawdowns
- Market beta and correlation
- Cross-sectional rank/z-score structure
- Date-level market breadth, dispersion, drawdown, skew, and kurtosis

The learned latent state is used by a walk-forward Ridge adapter to predict cross-sectional forward returns. This gives Arbitron a foundation-model-style representation layer while keeping the project reproducible on a laptop.

---

## Quick Start

### 1. Install

```bash
pip install -e ".[dev]"
pip install hmmlearn     # optional HMM regime detector
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env — add GROQ_API_KEY
# Get a free key at https://console.groq.com (email signup, no credit card)
```

### 3. Run tests

```bash
pytest tests/ -v
# → 143 passed
```

### 4. Start the API server

```bash
uvicorn api.main:app --reload --port 8000
# → http://localhost:8000/docs
```

### 5. Start the frontend

```bash
cd frontend
npm install
npm run dev
# → http://localhost:5173
```

### 6. CLI research

```bash
# Quick factor screen
python research.py --screen --universe SP500

# Full pipeline
python research.py --pipeline --universe TECH --start 2021-01-01 --end 2024-12-31

# Custom task
python research.py "Evaluate momentum and reversal factors; backtest the winner"
```

---

## Agent Tools

The agent has access to 8 tools that map directly to the quant library:

| Tool | Description |
|------|-------------|
| `fetch_universe` | Get tickers for a sector or default S&P 500 universe |
| `compute_factor` | Compute any of the 13 named factors |
| `evaluate_factor` | IC analysis + Fama-MacBeth regression |
| `run_backtest` | Walk-forward quintile L/S backtest |
| `optimize_portfolio` | MVO / Max Sharpe / Risk Parity weights |
| `compute_risk_metrics` | VaR, CVaR, drawdown, factor beta, tail ratio |
| `detect_regime` | Current market regime (Bull/Bear) |
| `compare_factors` | Side-by-side IC comparison for multiple factors |
| `run_quant_lab` | Full research lab with statistical defenses + market foundation signal |

---

## Factor Library

| Factor | Description | Frequency |
|--------|-------------|-----------|
| `momentum_252_21` | 12-month momentum (skip 1 month) | Daily |
| `momentum_126_21` | 6-month momentum (skip 1 month) | Daily |
| `momentum_63_5` | 3-month momentum (skip 1 week) | Daily |
| `reversal_21` | Short-term reversal (1-month) | Daily |
| `reversal_5` | Weekly reversal | Daily |
| `volatility_63` | Realised volatility (3-month, negated) | Daily |
| `volatility_21` | Realised volatility (1-month, negated) | Daily |
| `rsi_14` | RSI contrarian (centred, negated) | Daily |
| `rsi_7` | Short RSI contrarian | Daily |
| `macd` | MACD histogram | Daily |
| `bollinger` | Bollinger Band position | Daily |
| `price_to_high_252` | Price vs 52-week high | Daily |
| `acceleration_63` | Price acceleration (2nd derivative) | Daily |

---

## Portfolio Optimizers

| Optimizer | Method | Use Case |
|-----------|--------|----------|
| `max_sharpe` | Analytic + SLSQP | When return estimates are reliable |
| `min_variance` | SLSQP | When return estimates are unreliable |
| `risk_parity` | ERC (Maillard et al., 2010) | Robust, diversified |

All use Ledoit-Wolf shrinkage covariance (sklearn) by default.

---

## Risk Analytics

- **VaR/CVaR** at 90%, 95%, 99%: historical simulation, parametric (Gaussian), Monte Carlo (Student-t)
- **Factor decomposition**: OLS betas with HAC standard errors
- **Marginal Risk Contribution** (MRC) and **Component Risk Contribution** (CRC)
- **Drawdown analytics**: max duration, average duration, number of drawdown periods
- **Ulcer Index**: root mean squared drawdown
- **Tail Ratio**: avg 5th percentile gain / avg 5th percentile loss

---

## References

- Fama & MacBeth (1973) — Cross-sectional regression
- Ledoit & Wolf (2004) — Covariance shrinkage
- Maillard, Roncalli & Teïletche (2010) — Risk Parity
- López de Prado (2018) — *Advances in Financial Machine Learning*: purged CV, embargo
- Hamilton (1989) — Hidden Markov Model for regime detection
- Yao et al. (2023) — ReAct: Synergizing Reasoning and Acting in Language Models

---

Additional research basis for the flagship lab:

- Bailey & Lopez de Prado (2014) - Deflated Sharpe Ratio
- Bailey, Borwein, Lopez de Prado & Zhu (2014) - Probability of Backtest Overfitting
- Ansari et al. (2024) - Chronos: time-series foundation models
- Das et al. (2024) - TimesFM: decoder-only time-series foundation model

---

## Author

**Mohamed Zayaan S** — IIT Madras (Civil Engineering + CS Minor)
