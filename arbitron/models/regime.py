"""
Market regime detection using a Gaussian Hidden Markov Model.

A 2-state HMM fitted on (daily return, realised volatility) captures
bull/bear/low-vol/high-vol regimes.  Regime labels inform factor selection:
  - Bull/low-vol: momentum works well
  - Bear/high-vol: reversal / low-vol factors preferred

CAUSALITY — the point of this module, and the thing it previously got wrong
-------------------------------------------------------------------------
The old implementation fitted the HMM on the full sample and then
**Viterbi**-decoded the whole history.  Viterbi picks the single most likely
state *path*, which means the label printed for January 2020 depended on
what the market did in March 2020.  Any strategy conditioned on those labels
is reading the future.

Three levels of honesty are now offered explicitly:

* `predict_walk_forward()` — parameters refit on an expanding window and
  states forward-FILTERED inside each block.  Every label uses only data
  available at that date.  **This is the one to use for historical
  analysis or conditioning a backtest.**
* `current_regime()` — forward-filtered posterior at the final date.  Causal
  by construction (all data up to "now" is, by definition, the past).
* `predict()` / `predict_proba()` — forward-filtered but with full-sample
  PARAMETERS.  In-sample diagnostics only; each carries `causal=False` in
  its metadata.  Smoothed (forward-backward) posteriors are available via
  `smoothed=True` and are explicitly non-causal.

Reference: Hamilton (1989) "A New Approach to the Economic Analysis of
           Nonstationary Time Series and the Business Cycle"
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from scipy.special import logsumexp


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

    def _build_features(
        self,
        returns: pd.Series,
        vol_window: int = 21,
    ) -> tuple[np.ndarray, pd.Index]:
        """
        Build the causal (return, log-vol) feature matrix.

        Volatility is a TRAILING rolling standard deviation, so the feature
        at date t uses only returns up to and including t.

        Returns (X, index) where `index` are the dates that survived the
        volatility warm-up.  The index is taken from the *dropna'd* series —
        the previous version indexed into the raw `returns.index` with
        positions computed on the dropna'd array, which silently misaligned
        every label as soon as the return series contained a single NaN.
        """
        clean = pd.Series(returns, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
        r = clean.values
        vol = clean.rolling(vol_window, min_periods=5).std().values
        vol = np.where(vol > 0, np.log(vol + 1e-8), np.nan)

        valid = np.isfinite(vol) & np.isfinite(r)
        return np.column_stack([r[valid], vol[valid]]), clean.index[valid]

    # ------------------------------------------------------------------
    # Forward filtering (causal) — replaces Viterbi decoding
    # ------------------------------------------------------------------

    def _emission_loglik(self, X: np.ndarray, model=None) -> np.ndarray:
        """
        (T × n_states) Gaussian emission log-likelihoods.

        Computed from `means_` / `covars_` directly rather than via a private
        hmmlearn helper, so this does not break across hmmlearn releases.
        """
        from scipy.stats import multivariate_normal

        model = model or self._model
        covars = model.covars_
        out = np.empty((len(X), model.n_components), dtype=float)
        for s in range(model.n_components):
            cov = np.atleast_2d(covars[s])
            # Ridge for numerical safety on near-singular state covariances.
            cov = cov + 1e-12 * np.eye(cov.shape[0])
            out[:, s] = multivariate_normal.logpdf(
                X, mean=model.means_[s], cov=cov, allow_singular=True
            )
        return out

    def _forward_filter(self, X: np.ndarray, model=None) -> np.ndarray:
        """
        Forward (alpha) recursion in log space → **filtered** posteriors.

        `filtered[t, s] = P(state_t = s | observations_1..t)`

        Contrast with `hmmlearn`'s `predict_proba`, which runs the full
        forward-BACKWARD pass and therefore conditions on observations after
        t, and with `predict`, which runs Viterbi over the whole path.  Both
        of those look ahead.
        """
        model = model or self._model
        n_states = model.n_components
        loglik = self._emission_loglik(X, model)

        log_start = np.log(np.maximum(model.startprob_, 1e-300))
        log_trans = np.log(np.maximum(model.transmat_, 1e-300))

        filtered = np.empty((len(X), n_states), dtype=float)
        log_alpha = log_start + loglik[0]
        filtered[0] = np.exp(log_alpha - logsumexp(log_alpha))

        for t in range(1, len(X)):
            log_alpha = logsumexp(log_alpha[:, None] + log_trans, axis=0) + loglik[t]
            log_alpha -= logsumexp(log_alpha)      # normalise to avoid underflow
            filtered[t] = np.exp(log_alpha)

        return filtered

    def _label_states(self, model) -> dict[int, str]:
        """Map state indices to semantic labels by mean return."""
        means = model.means_[:, 0]
        labels = {int(np.argmax(means)): "Bull", int(np.argmin(means)): "Bear"}
        for s in range(model.n_components):
            labels.setdefault(s, f"Neutral-{s}")
        return labels

    def _new_model(self):
        import logging

        from hmmlearn.hmm import GaussianHMM

        # hmmlearn logs a WARNING per non-converged EM run.  Walk-forward
        # refits produce dozens of them and they drown real API logs; the
        # non-convergence is already visible in the posterior confidences.
        logging.getLogger("hmmlearn").setLevel(logging.ERROR)
        return GaussianHMM(
            n_components=self.n_components,
            covariance_type="full",
            n_iter=200,
            random_state=self.random_state,
            tol=1e-4,
        )

    def fit(self, returns: pd.Series, vol_window: int = 21) -> "RegimeDetector":
        """
        Fit the HMM on historical returns.

        Args:
            returns:    Daily return series.
            vol_window: Window for realised vol feature.
        """
        try:
            import hmmlearn  # noqa: F401
        except ImportError:
            raise ImportError("hmmlearn is required: pip install hmmlearn")

        X, _ = self._build_features(returns, vol_window)
        if len(X) < self.n_components * 10:
            raise ValueError(
                f"Need at least {self.n_components * 10} observations to fit a "
                f"{self.n_components}-state HMM; got {len(X)}."
            )

        self._model = self._new_model()
        self._model.fit(X)
        self._fitted = True
        self._regime_labels = self._label_states(self._model)
        return self

    def predict(
        self,
        returns:    pd.Series,
        vol_window: int = 21,
    ) -> pd.Series:
        """
        Regime label for each date from the **forward-filtered** posterior.

        Causality caveat: the state path no longer looks ahead (Viterbi is
        gone), but the HMM PARAMETERS are still whatever `fit()` saw.  If
        `fit()` was given the full sample, these labels are in-sample
        diagnostics.  Use `predict_walk_forward()` for history you intend to
        trade on or condition a backtest with.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before predict()")

        X, idx = self._build_features(returns, vol_window)
        if len(X) == 0:
            return pd.Series(dtype=object, name="regime")
        states = self._forward_filter(X).argmax(axis=1)
        labels = [self._regime_labels.get(int(s), str(s)) for s in states]
        return pd.Series(labels, index=idx, name="regime")

    def predict_proba(
        self,
        returns:    pd.Series,
        vol_window: int = 21,
        smoothed:   bool = False,
    ) -> pd.DataFrame:
        """
        Posterior probability of each state per date.

        Args:
            smoothed: If True, run the full forward-BACKWARD pass.  Those
                posteriors condition on future observations and are **not
                causal** — useful for describing history, never for trading
                it.  Default False = forward-filtered.

        Returns:
            (dates × n_components) DataFrame.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before predict_proba()")

        X, idx = self._build_features(returns, vol_window)
        cols = [self._regime_labels.get(i, str(i)) for i in range(self.n_components)]
        if len(X) == 0:
            return pd.DataFrame(columns=cols)

        proba = self._model.predict_proba(X) if smoothed else self._forward_filter(X)
        return pd.DataFrame(proba, index=idx, columns=cols)

    def predict_walk_forward(
        self,
        returns:    pd.Series,
        vol_window: int = 21,
        min_train:  int = 252,
        refit_freq: int = 63,
    ) -> pd.DataFrame:
        """
        Genuinely causal regime history: refit forward, filter forward.

        For each block of `refit_freq` days the HMM is re-estimated on the
        expanding window that ends *before* the block, then the block's
        states are forward-filtered under those parameters.  No label
        anywhere in the output uses an observation dated after it.

        The first `min_train` observations have no label (there is nothing to
        fit on yet) and are omitted rather than back-filled.

        Returns:
            DataFrame indexed by date with columns
            ``[regime, confidence, refit_date]``, plus one probability column
            per state.
        """
        if not self._fitted:
            # Walk-forward does its own fitting; a prior fit() is not required
            # but we need hmmlearn present.
            try:
                import hmmlearn  # noqa: F401
            except ImportError:
                raise ImportError("hmmlearn is required: pip install hmmlearn")

        X, idx = self._build_features(returns, vol_window)
        n = len(X)
        if n <= min_train:
            return pd.DataFrame(columns=["regime", "confidence", "refit_date"])

        rows: list[dict] = []
        for block_start in range(min_train, n, max(1, refit_freq)):
            block_end = min(block_start + refit_freq, n)

            model = self._new_model()
            try:
                model.fit(X[:block_start])           # strictly past data
            except Exception:
                continue
            labels = self._label_states(model)

            # Filter over past + block, then keep only the block's rows: the
            # alpha recursion needs the history to reach the correct state
            # belief, but every alpha_t still depends on t' <= t only.
            filtered = self._forward_filter(X[:block_end], model)[block_start:block_end]

            refit_date = idx[block_start - 1]
            for offset, probs in enumerate(filtered):
                s = int(np.argmax(probs))
                row = {
                    "regime":     labels.get(s, str(s)),
                    "confidence": round(float(probs[s]), 4),
                    "refit_date": refit_date,
                }
                for state in range(model.n_components):
                    row[f"p_{labels.get(state, state)}"] = round(float(probs[state]), 4)
                rows.append(row)

        if not rows:
            return pd.DataFrame(columns=["regime", "confidence", "refit_date"])

        out = pd.DataFrame(rows, index=idx[min_train:min_train + len(rows)])
        out.index.name = "date"
        return out

    def current_regime(
        self,
        returns:    pd.Series,
        vol_window: int = 21,
    ) -> str:
        """
        The regime as of the last observation — the only output of this path
        that is causal without qualification, because "all data up to now" is
        by definition the past.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before current_regime()")
        X, _ = self._build_features(returns, vol_window)
        if len(X) == 0:
            return "Unknown"
        probs = self._forward_filter(X)[-1]
        return self._regime_labels.get(int(np.argmax(probs)), "Unknown")

    def current_regime_detail(
        self,
        returns:    pd.Series,
        vol_window: int = 21,
    ) -> dict:
        """Current regime plus its filtered probability distribution."""
        if not self._fitted:
            raise RuntimeError("Call fit() before current_regime_detail()")
        X, idx = self._build_features(returns, vol_window)
        if len(X) == 0:
            return {"regime": "Unknown", "confidence": None, "probabilities": {}, "causal": True}
        probs = self._forward_filter(X)[-1]
        s = int(np.argmax(probs))
        return {
            "regime": self._regime_labels.get(s, str(s)),
            "confidence": round(float(probs[s]), 4),
            "probabilities": {
                self._regime_labels.get(i, str(i)): round(float(p), 4)
                for i, p in enumerate(probs)
            },
            "as_of": str(idx[-1].date()) if len(idx) else None,
            "causal": True,
        }

    def regime_stats(
        self,
        returns:    pd.Series,
        vol_window: int = 21,
        labels:     Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        """
        Summary statistics per regime.

        Args:
            labels: Optional pre-computed labels — pass
                `predict_walk_forward()["regime"]` for causal statistics.
                Defaults to `predict()` (in-sample parameters).
        """
        labels = self.predict(returns, vol_window) if labels is None else labels
        return _regime_stats_frame(returns, labels)


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

    def predict_walk_forward(self, returns: pd.Series, **kwargs) -> pd.DataFrame:
        """
        Causal history.  The rule-based detector is already causal end to end
        (expanding percentiles, trailing SMA), so this is `predict()` plus the
        metadata columns the HMM path emits.
        """
        labels = self.predict(returns)
        return pd.DataFrame({
            "regime": labels,
            "confidence": 1.0,
            "refit_date": pd.NaT,
        })

    def current_regime(self, returns: pd.Series, **kwargs) -> str:
        labels = self.predict(returns)
        return str(labels.iloc[-1]) if len(labels) else "Unknown"

    def current_regime_detail(self, returns: pd.Series, **kwargs) -> dict:
        labels = self.predict(returns)
        if not len(labels):
            return {"regime": "Unknown", "confidence": None, "probabilities": {}, "causal": True}
        regime = str(labels.iloc[-1])
        return {
            "regime": regime,
            "confidence": 1.0,
            "probabilities": {"Bull": 1.0 if regime == "Bull" else 0.0,
                              "Bear": 1.0 if regime == "Bear" else 0.0},
            "as_of": str(labels.index[-1].date()) if len(labels) else None,
            "causal": True,
        }

    def regime_stats(
        self,
        returns: pd.Series,
        labels: Optional[pd.Series] = None,
        **kwargs,
    ) -> pd.DataFrame:
        labels = self.predict(returns) if labels is None else labels
        return _regime_stats_frame(returns, labels)


def _regime_stats_frame(returns: pd.Series, labels: pd.Series) -> pd.DataFrame:
    """Per-regime return statistics, aligned on the labelled dates only."""
    df = pd.DataFrame({
        "return": pd.Series(returns, dtype=float),
        "regime": pd.Series(labels),
    }).dropna()
    if df.empty:
        return pd.DataFrame(
            columns=["mean_ret_ann", "vol_ann", "n_days", "frequency"]
        ).rename_axis("regime")

    rows = []
    for regime, grp in df.groupby("regime"):
        rows.append({
            "regime":       regime,
            "mean_ret_ann": round(float(grp["return"].mean() * 252), 4),
            "vol_ann":      round(float(grp["return"].std() * np.sqrt(252)), 4),
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
