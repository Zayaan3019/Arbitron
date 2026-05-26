"""
ML-based return predictor using Random Forest.

Features: cross-sectional factor values + regime indicator + macro proxies.
Target: next-period return (21-day ahead).

Walk-forward fitting: model is re-trained on rolling windows to ensure
no lookahead bias.  Purged train/test splits (Lopez de Prado method) are
used to avoid contamination from overlapping return windows.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


class ReturnPredictor:
    """
    Random Forest return predictor with walk-forward cross-validation.

    Features per stock per date:
      - Factor values (momentum, reversal, volatility, RSI, MACD, ...)
      - Rolling Sharpe (lookback 63d)
      - Market beta (rolling 63d)
      - Regime dummy (from RegimeDetector)

    Target: 21-day forward return (sign or magnitude).

    The predictor is fitted per-date in a walk-forward fashion:
      - Training: all data from [start, t - embargo]
      - Prediction: t
    """

    def __init__(
        self,
        n_estimators:    int   = 100,
        max_depth:       int   = 5,
        min_samples_leaf: int  = 10,
        n_jobs:          int   = -1,
        random_state:    int   = 42,
    ) -> None:
        self.n_estimators     = n_estimators
        self.max_depth        = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.n_jobs           = n_jobs
        self.random_state     = random_state
        self._model           = None
        self._feature_names: list[str] = []

    # ------------------------------------------------------------------
    # Feature engineering
    # ------------------------------------------------------------------

    def _build_features(
        self,
        factors:    dict[str, pd.DataFrame],
        prices:     pd.DataFrame,
        market_ret: pd.Series,
        regime:     Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        """
        Build a stacked (date × ticker, features) panel DataFrame.

        Each row is a (date, ticker) observation.
        """
        rows = []
        tickers = prices.columns.tolist()

        # Rolling Sharpe and beta per ticker
        returns = prices.pct_change()
        roll_sr   = {}
        roll_beta = {}
        for col in tickers:
            r = returns[col]
            roll_sr[col]   = (r.rolling(63).mean() / r.rolling(63).std() * np.sqrt(252)).fillna(0)
            cov = r.rolling(63).cov(market_ret)
            var = market_ret.rolling(63).var()
            roll_beta[col] = (cov / var.replace(0, np.nan)).fillna(1.0)

        # Build panel
        for date in prices.index:
            for ticker in tickers:
                row: dict = {"date": date, "ticker": ticker}

                # Factor features
                for fname, fdf in factors.items():
                    if date in fdf.index and ticker in fdf.columns:
                        val = fdf.at[date, ticker]
                        row[f"f_{fname}"] = float(val) if not pd.isna(val) else 0.0
                    else:
                        row[f"f_{fname}"] = 0.0

                # Rolling Sharpe
                row["rolling_sharpe"] = float(roll_sr[ticker].get(date, 0.0) or 0.0)

                # Rolling beta
                row["rolling_beta"] = float(roll_beta[ticker].get(date, 1.0) or 1.0)

                # Regime
                if regime is not None and date in regime.index:
                    row["regime_bull"] = 1.0 if regime[date] == "Bull" else 0.0
                else:
                    row["regime_bull"] = 0.5

                rows.append(row)

        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.set_index(["date", "ticker"])
        return df

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        X:      pd.DataFrame,   # (n_samples, n_features)
        y:      pd.Series,      # (n_samples,) target returns
    ) -> "ReturnPredictor":
        """Fit the Random Forest on provided features and targets."""
        try:
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.preprocessing import StandardScaler
        except ImportError:
            raise ImportError("scikit-learn is required: pip install scikit-learn")

        # Remove NaNs
        mask = y.notna() & X.notna().all(axis=1)
        X_c, y_c = X.loc[mask], y.loc[mask]
        if len(X_c) < 50:
            return self

        self._feature_names = X_c.columns.tolist()
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X_c.values)

        self._model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
        )
        self._model.fit(X_scaled, y_c.values)
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Predict expected returns for the given feature matrix."""
        if self._model is None:
            return pd.Series(0.0, index=X.index)

        X_valid = X[self._feature_names].fillna(0)
        X_scaled = self._scaler.transform(X_valid.values)
        preds = self._model.predict(X_scaled)
        return pd.Series(preds, index=X.index)

    def feature_importance(self) -> pd.Series:
        """Return feature importances from the fitted model."""
        if self._model is None:
            return pd.Series(dtype=float)
        return pd.Series(
            self._model.feature_importances_,
            index=self._feature_names,
        ).sort_values(ascending=False)

    # ------------------------------------------------------------------
    # Walk-forward signal generation
    # ------------------------------------------------------------------

    def walk_forward_signals(
        self,
        factors:        dict[str, pd.DataFrame],
        prices:         pd.DataFrame,
        market_ret:     pd.Series,
        horizon:        int = 21,
        train_window:   int = 504,   # ~2 years
        refit_freq:     int = 63,    # Refit every ~quarter
        embargo:        int = 21,
        regime:         Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        """
        Generate predicted return signals using walk-forward fitting.

        For each test date:
          1. Fit on [t - train_window - embargo, t - embargo]
          2. Predict for date t
          3. Advance by refit_freq days and repeat

        Returns:
            (test_dates × tickers) signal DataFrame.
        """
        returns  = prices.pct_change()
        fwd_ret  = (prices.shift(-horizon) / prices - 1)

        all_dates = sorted(prices.index)
        signals   = {}

        for i in range(train_window + embargo, len(all_dates), refit_freq):
            test_date   = all_dates[i]
            train_end   = all_dates[i - embargo]
            train_start = all_dates[max(0, i - train_window - embargo)]

            train_dates = all_dates[
                all_dates.index(train_start): all_dates.index(train_end) + 1
            ]

            # Build training features and labels
            panel = self._build_features(factors, prices, market_ret, regime)
            if panel.empty:
                continue

            # Select train rows
            try:
                train_X = panel.loc[(train_dates, slice(None)), :]
                # Build target: forward returns for training dates (offset by horizon)
                y_rows = []
                for td in train_dates:
                    if td not in fwd_ret.index:
                        continue
                    for tk in prices.columns:
                        idx = (td, tk)
                        if idx in train_X.index:
                            y_rows.append((idx, fwd_ret.at[td, tk] if tk in fwd_ret.columns else np.nan))

                if not y_rows:
                    continue

                y_idx, y_vals = zip(*y_rows)
                y_train = pd.Series(list(y_vals), index=pd.MultiIndex.from_tuples(list(y_idx)))
                self.fit(train_X.loc[list(y_idx)], y_train)

                # Predict test date
                if (test_date,) in panel.index.droplevel(1) or \
                   any(test_date == idx[0] for idx in panel.index):
                    test_X = panel.loc[(test_date, slice(None)), :]
                    pred   = self.predict(test_X)
                    signals[test_date] = pred.droplevel(0)
            except (KeyError, ValueError):
                continue

        if not signals:
            return pd.DataFrame()

        return pd.DataFrame(signals).T
