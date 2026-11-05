"""
Market foundation modeling.

The implementation is intentionally local and dependency-light: it learns
self-supervised market representations from causal, multi-scale price features,
then trains a walk-forward adapter to turn those representations into a
cross-sectional return signal.  The architecture leaves room for external
foundation model adapters while making the current repo runnable with sklearn.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Any

import numpy as np
import pandas as pd

from arbitron.factors.evaluation import forward_returns
from arbitron.factors.library import cross_section_rank, cross_section_zscore


@dataclass(frozen=True)
class MarketFoundationConfig:
    horizon: int = 21
    train_window: int = 504
    refit_freq: int = 63
    embargo_days: int = 21
    n_components: int = 6
    min_train_samples: int = 250
    random_state: int = 42


@dataclass
class FoundationSignalResult:
    signal: pd.DataFrame
    embeddings: pd.DataFrame
    reconstruction_error: pd.Series
    explained_variance: list[float]
    diagnostics: dict[str, Any]


def build_asset_feature_panel(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Build causal asset-date features for representation learning.

    Index is (date, ticker), columns are feature names.  All features use data
    available at or before the date, matching the backtester convention.
    """
    prices = prices.sort_index().astype(float)
    returns = prices.pct_change(fill_method=None)
    market_ret = returns.mean(axis=1)
    vol_21 = returns.rolling(21, min_periods=10).std() * sqrt(252)
    vol_63 = returns.rolling(63, min_periods=21).std() * sqrt(252)

    features: dict[str, pd.DataFrame] = {
        "ret_1": returns,
        "ret_5": prices.pct_change(5, fill_method=None),
        "ret_21": prices.pct_change(21, fill_method=None),
        "ret_63": prices.pct_change(63, fill_method=None),
        "ret_126": prices.pct_change(126, fill_method=None),
        "vol_21": vol_21,
        "vol_63": vol_63,
        "vol_ratio": vol_21 / vol_63.replace(0, np.nan) - 1.0,
        "drawdown_63": prices / prices.rolling(63, min_periods=21).max() - 1.0,
        "drawdown_252": prices / prices.rolling(252, min_periods=63).max() - 1.0,
        "rank_mom_21": cross_section_rank(prices.pct_change(21, fill_method=None)),
        "rank_mom_126": cross_section_rank(prices.pct_change(126, fill_method=None)),
        "rank_low_vol_63": cross_section_rank(-vol_63),
        "z_mom_21": cross_section_zscore(prices.pct_change(21, fill_method=None)),
        "z_low_vol_63": cross_section_zscore(-vol_63),
    }

    beta = pd.DataFrame(index=prices.index, columns=prices.columns, dtype=float)
    corr = pd.DataFrame(index=prices.index, columns=prices.columns, dtype=float)
    market_var = market_ret.rolling(63, min_periods=21).var()
    for col in prices.columns:
        cov = returns[col].rolling(63, min_periods=21).cov(market_ret)
        beta[col] = cov / market_var.replace(0, np.nan)
        corr[col] = returns[col].rolling(63, min_periods=21).corr(market_ret)
    features["beta_63"] = beta
    features["corr_market_63"] = corr

    stacked = {
        name: _stack_frame(frame.replace([np.inf, -np.inf], np.nan))
        for name, frame in features.items()
    }
    panel = pd.concat(stacked, axis=1)
    panel.index = panel.index.set_names(["date", "ticker"])
    panel = panel.replace([np.inf, -np.inf], np.nan)
    panel = panel.groupby(level=0).transform(lambda x: x.fillna(x.median()))
    return panel.fillna(0.0)


def build_market_state_features(prices: pd.DataFrame) -> pd.DataFrame:
    """Build date-level market state features for the lab dashboard."""
    returns = prices.pct_change(fill_method=None)
    market = returns.mean(axis=1)
    cum = (1.0 + market.fillna(0.0)).cumprod()
    state = pd.DataFrame(index=prices.index)
    state["market_ret_1"] = market
    state["market_ret_21"] = cum.pct_change(21, fill_method=None)
    state["market_ret_63"] = cum.pct_change(63, fill_method=None)
    state["realized_vol_21"] = market.rolling(21, min_periods=10).std() * sqrt(252)
    state["realized_vol_63"] = market.rolling(63, min_periods=21).std() * sqrt(252)
    state["cross_section_dispersion"] = returns.std(axis=1)
    state["breadth"] = (returns > 0).mean(axis=1)
    state["market_drawdown"] = cum / cum.cummax() - 1.0
    state["xs_skew"] = returns.skew(axis=1)
    state["xs_kurtosis"] = returns.kurtosis(axis=1)
    state = state.replace([np.inf, -np.inf], np.nan)
    return state.ffill().fillna(0.0)


def market_state_embeddings(
    prices: pd.DataFrame,
    n_components: int = 6,
    random_state: int = 42,
) -> tuple[pd.DataFrame, list[float]]:
    """Fit a PCA state encoder and return date-level latent embeddings."""
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    state = build_market_state_features(prices)
    n_comp = max(1, min(n_components, state.shape[1], len(state) - 1))
    scaler = StandardScaler()
    X = scaler.fit_transform(state.values)
    pca = PCA(n_components=n_comp, random_state=random_state)
    Z = pca.fit_transform(X)
    cols = [f"state_z{i + 1}" for i in range(n_comp)]
    embeddings = pd.DataFrame(Z, index=state.index, columns=cols)
    return embeddings, [float(x) for x in pca.explained_variance_ratio_]


def build_foundation_signal(
    prices: pd.DataFrame,
    cfg: MarketFoundationConfig | None = None,
) -> FoundationSignalResult:
    """
    Build a walk-forward market-foundation signal.

    Each refit:
      1. Fit scaler + PCA on past asset features.
      2. Train RidgeCV on original features + latent representation.
      3. Predict forward returns for the next refit window.
      4. Track reconstruction error as a market novelty score.
    """
    from sklearn.decomposition import PCA
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler

    cfg = cfg or MarketFoundationConfig()
    prices = prices.sort_index()
    panel = build_asset_feature_panel(prices)
    y = _stack_frame(forward_returns(prices, cfg.horizon)).rename("target")
    all_dates = list(prices.index)

    signal_store: dict[pd.Timestamp, pd.Series] = {}
    recon_store: list[pd.Series] = []
    explained: list[float] = []
    coef_accumulator = pd.Series(0.0, index=list(panel.columns) + [f"z{i + 1}" for i in range(cfg.n_components)])
    refits = 0

    first_test_pos = cfg.train_window + cfg.horizon + cfg.embargo_days
    last_test_pos = max(first_test_pos, len(all_dates) - cfg.horizon)

    for test_start_pos in range(first_test_pos, last_test_pos, cfg.refit_freq):
        test_end_pos = min(test_start_pos + cfg.refit_freq, len(all_dates) - cfg.horizon)
        train_end_pos = test_start_pos - cfg.horizon - cfg.embargo_days
        train_start_pos = max(0, train_end_pos - cfg.train_window)
        if train_end_pos <= train_start_pos:
            continue

        train_dates = all_dates[train_start_pos:train_end_pos]
        test_dates = all_dates[test_start_pos:test_end_pos]
        if not train_dates or not test_dates:
            continue

        train_mask = panel.index.get_level_values("date").isin(train_dates)
        test_mask = panel.index.get_level_values("date").isin(test_dates)
        X_train = panel.loc[train_mask]
        X_test = panel.loc[test_mask]
        y_train = y.reindex(X_train.index)
        valid = y_train.notna() & X_train.notna().all(axis=1)
        if valid.sum() < cfg.min_train_samples or X_test.empty:
            continue

        X_train = X_train.loc[valid]
        y_train = y_train.loc[valid]
        n_comp = max(1, min(cfg.n_components, X_train.shape[1], len(X_train) - 1))

        scaler = StandardScaler()
        Xs_train = scaler.fit_transform(X_train.values)
        Xs_test = scaler.transform(X_test.values)

        pca = PCA(n_components=n_comp, random_state=cfg.random_state)
        Z_train = pca.fit_transform(Xs_train)
        Z_test = pca.transform(Xs_test)

        model = RidgeCV(alphas=(0.01, 0.1, 1.0, 10.0, 100.0))
        model.fit(np.column_stack([Xs_train, Z_train]), y_train.values)
        preds = model.predict(np.column_stack([Xs_test, Z_test]))
        pred_s = pd.Series(preds, index=X_test.index)

        for date, group in pred_s.groupby(level="date"):
            signal_store[pd.Timestamp(date)] = group.droplevel("date")

        reconstructed = pca.inverse_transform(Z_test)
        mse = pd.Series(np.mean((Xs_test - reconstructed) ** 2, axis=1), index=X_test.index)
        recon_store.append(mse.groupby(level="date").mean())
        explained.append(float(np.sum(pca.explained_variance_ratio_)))

        coef_names = list(panel.columns) + [f"z{i + 1}" for i in range(n_comp)]
        coef_abs = pd.Series(np.abs(model.coef_), index=coef_names)
        coef_accumulator = coef_accumulator.add(coef_abs, fill_value=0.0)
        refits += 1

    signal = pd.DataFrame(signal_store).T.sort_index()
    if not signal.empty:
        signal = cross_section_zscore(signal.reindex(columns=prices.columns))

    if recon_store:
        reconstruction_error = pd.concat(recon_store).sort_index()
        reconstruction_error = reconstruction_error[~reconstruction_error.index.duplicated(keep="last")]
    else:
        reconstruction_error = pd.Series(dtype=float, name="reconstruction_error")
    reconstruction_error.name = "reconstruction_error"

    embeddings, state_explained = market_state_embeddings(
        prices, n_components=cfg.n_components, random_state=cfg.random_state
    )

    top_features = (
        coef_accumulator.sort_values(ascending=False)
        .head(8)
        .round(6)
        .to_dict()
    )
    diagnostics = {
        "refits": refits,
        "coverage_days": int(signal.dropna(how="all").shape[0]) if not signal.empty else 0,
        "coverage_ratio": (
            round(float(signal.dropna(how="all").shape[0] / len(prices)), 4)
            if len(prices) and not signal.empty else 0.0
        ),
        "asset_features": list(panel.columns),
        "top_adapter_features": {str(k): float(v) for k, v in top_features.items()},
        "state_explained_variance": state_explained,
        "mean_window_explained_variance": round(float(np.mean(explained)), 4) if explained else 0.0,
    }

    return FoundationSignalResult(
        signal=signal,
        embeddings=embeddings,
        reconstruction_error=reconstruction_error,
        explained_variance=explained,
        diagnostics=diagnostics,
    )


def _stack_frame(frame: pd.DataFrame) -> pd.Series:
    """Stack while preserving missing ticker/date pairs across pandas versions."""
    try:
        return frame.stack(future_stack=True)
    except TypeError:
        return frame.stack(dropna=False)
