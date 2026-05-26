"""
Market regime detection using a Gaussian Hidden Markov Model.

A 2-state HMM fitted on (daily return, realised volatility) captures
bull/bear/low-vol/high-vol regimes.  Regime labels inform factor selection:
  - Bull/low-vol: momentum works well
  - Bear/high-vol: reversal / low-vol factors preferred

Reference: Hamilton (1989) "A New Approach to the Economic Analysis of
           Nonstationary Time Series and the Business Cycle"
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Gaussian HMM regime detector
# ---------------------------------------------------------------------------

class RegimeDetector:
    """
    Gaussian HMM fitted on (return, log-volatility) feature pairs.

    After fitting, regimes are labelled:
      0 = "Low-vol / Bull"  (higher mean return, lower vol)
      1 = "High-vol / Bear" (lower mean return, higher vol)
    """

    N_COMPONENTS: int = 2

    def __init__(self, n_components: int = 2, random_state: int = 42) -> None:
        self.n_components  = n_components
        self.random_state  = random_state
        self._model        = None
        self._fitted       = False
        self._regime_labels: dict[int, str] = {}

    def _build_features(self, returns: pd.Series, vol_window: int = 21) -> np.ndarray:
        """
        Build (return, log-vol) feature matrix.

        Using log-volatility makes the distribution more symmetric.
        """
        r   = returns.dropna().values
        vol = pd.Series(r).rolling(vol_window, min_periods=5).std().values
        vol = np.where(vol > 0, np.log(vol + 1e-8), np.log(1e-8))
        # Drop the warm-up rows
        valid = ~np.isnan(vol)
        return np.column_stack([r[valid], vol[valid]]), returns.index[np.where(valid)[0]]

    def fit(self, returns: pd.Series, vol_window: int = 21) -> "RegimeDetector":
        """
        Fit the HMM on historical returns.

        Args:
            returns:    Daily return series.
            vol_window: Window for realised vol feature.
        """
        try:
            from hmmlearn.hmm import GaussianHMM
        except ImportError:
            raise ImportError("hmmlearn is required: pip install hmmlearn")

        X, _  = self._build_features(returns, vol_window)

        self._model = GaussianHMM(
            n_components=self.n_components,
            covariance_type="full",
            n_iter=200,
            random_state=self.random_state,
            tol=1e-4,
        )
        self._model.fit(X)
        self._fitted = True

        # Assign semantic labels based on mean returns per state
        # State with higher mean return = "Bull"
        means     = self._model.means_[:, 0]   # first feature = return
        bull_state = int(np.argmax(means))
        bear_state = int(np.argmin(means))

        self._regime_labels = {
            bull_state: "Bull",
            bear_state: "Bear",
        }
        if self.n_components > 2:
            for s in range(self.n_components):
                if s not in self._regime_labels:
                    self._regime_labels[s] = f"Neutral-{s}"

        return self

    def predict(
        self,
        returns:    pd.Series,
        vol_window: int = 21,
    ) -> pd.Series:
        """
        Predict regime labels for each date.

        Returns:
            pd.Series indexed by date with string labels ('Bull'/'Bear').
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before predict()")

        X, idx = self._build_features(returns, vol_window)
        state_seq = self._model.predict(X)
        labels    = [self._regime_labels.get(s, str(s)) for s in state_seq]
        return pd.Series(labels, index=idx, name="regime")

    def predict_proba(
        self,
        returns:    pd.Series,
        vol_window: int = 21,
    ) -> pd.DataFrame:
        """
        Posterior probabilities of being in each state.

        Returns:
            (dates × n_components) DataFrame.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before predict_proba()")

        X, idx = self._build_features(returns, vol_window)
        proba  = self._model.predict_proba(X)   # (T, n_states)
        cols   = [self._regime_labels.get(i, str(i)) for i in range(self.n_components)]
        return pd.DataFrame(proba, index=idx, columns=cols)

    def current_regime(
        self,
        returns:    pd.Series,
        vol_window: int = 21,
    ) -> str:
        """Return the most recently predicted regime label."""
        labels = self.predict(returns, vol_window)
        return str(labels.iloc[-1]) if len(labels) > 0 else "Unknown"

    def regime_stats(self, returns: pd.Series, vol_window: int = 21) -> pd.DataFrame:
        """
        Summary statistics per regime.

        Returns:
            DataFrame with mean return, volatility, frequency per regime.
        """
        labels = self.predict(returns, vol_window)
        r      = returns.reindex(labels.index)
        df     = pd.DataFrame({"return": r, "regime": labels})

        rows = []
        for regime, grp in df.groupby("regime"):
            rows.append({
                "regime":       regime,
                "mean_ret_ann": round(grp["return"].mean() * 252, 4),
                "vol_ann":      round(grp["return"].std() * np.sqrt(252), 4),
                "n_days":       len(grp),
                "frequency":    round(len(grp) / len(df), 4),
            })
        return pd.DataFrame(rows).set_index("regime")


# ---------------------------------------------------------------------------
# Simple rule-based regime (fallback when hmmlearn unavailable)
# ---------------------------------------------------------------------------

class SimpleRegimeDetector:
    """
    Rule-based regime detection using rolling returns and vol z-scores.

    Bull:  200-day SMA trend up AND realized vol below 1-year median.
    Bear:  200-day SMA trend down OR realized vol above 90th percentile.
    """

    def __init__(self, trend_window: int = 200, vol_window: int = 63) -> None:
        self.trend_window = trend_window
        self.vol_window   = vol_window

    def fit(self, returns: pd.Series, **kwargs) -> "SimpleRegimeDetector":
        return self

    def predict(self, returns: pd.Series, **kwargs) -> pd.Series:
        """Classify each day as 'Bull' or 'Bear'."""
        prices = (1 + returns).cumprod()
        sma    = prices.rolling(self.trend_window, min_periods=50).mean()
        trend  = prices > sma

        vol    = returns.rolling(self.vol_window, min_periods=10).std()
        vol_90 = vol.expanding(min_periods=50).quantile(0.9)
        high_vol = vol > vol_90

        regime = pd.Series("Bull", index=returns.index, name="regime")
        regime[~trend | high_vol] = "Bear"
        return regime

    def predict_proba(self, returns: pd.Series, **kwargs) -> pd.DataFrame:
        labels = self.predict(returns)
        bull   = (labels == "Bull").astype(float)
        bear   = 1.0 - bull
        return pd.DataFrame({"Bull": bull, "Bear": bear})

    def current_regime(self, returns: pd.Series, **kwargs) -> str:
        return str(self.predict(returns).iloc[-1])

    def regime_stats(self, returns: pd.Series, **kwargs) -> pd.DataFrame:
        labels = self.predict(returns)
        df     = pd.DataFrame({"return": returns, "regime": labels})
        rows   = []
        for regime, grp in df.groupby("regime"):
            rows.append({
                "regime":       regime,
                "mean_ret_ann": round(grp["return"].mean() * 252, 4),
                "vol_ann":      round(grp["return"].std() * np.sqrt(252), 4),
                "n_days":       len(grp),
                "frequency":    round(len(grp) / len(df), 4),
            })
        return pd.DataFrame(rows).set_index("regime")


def build_regime_detector(use_hmm: bool = True, **kwargs):
    """Factory: returns HMM or simple rule-based detector."""
    if use_hmm:
        try:
            import hmmlearn  # noqa
            return RegimeDetector(**kwargs)
        except ImportError:
            pass
    return SimpleRegimeDetector(**kwargs)
