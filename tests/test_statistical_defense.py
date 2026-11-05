"""
Numerical tests for the statistical-defense layer.

These are the three tests the audit asked for before any new feature, plus
the regression tests for the Deflated Sharpe bug:

  1. PSR against a value computed by hand from the Bailey & Lopez de Prado
     formula, independently of the implementation.
  2. PBO ~= 0.5 on iid strategies (no one is really better) and PBO -> 1 on a
     deliberately overfit set (the in-sample winner is engineered to be the
     out-of-sample loser).
  3. Look-ahead detection: shifting a factor forward in time must destroy its
     apparent predictive power. If it does not, the pipeline is peeking.

Test 3 lives in `test_lookahead.py`; 1 and 2 are here.
"""

from __future__ import annotations

from math import e, sqrt

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from arbitron.lab.statistical_defense import (
    EULER_MASCHERONI,
    audit_strategy,
    deflated_sharpe_ratio,
    estimate_backtest_overfitting,
    expected_maximum_sharpe,
    probabilistic_sharpe_ratio,
)
from arbitron.trials import TrialRegistry


# ---------------------------------------------------------------------------
# 1. PSR against a hand-computed value
# ---------------------------------------------------------------------------

class TestProbabilisticSharpeRatio:
    """
    PSR = Phi( (SR - SR*) * sqrt(n-1) / sqrt(1 - g1*SR + (g2-1)/4 * SR^2) )

    where SR is the NON-annualised (per-period) Sharpe, g1 is sample skew and
    g2 is sample kurtosis on the NON-excess convention (fisher=False).  The
    reference values below are computed from that formula directly, using
    scipy only for skew/kurtosis/Phi — never by calling the implementation.
    """

    @staticmethod
    def _reference_psr(r: np.ndarray, benchmark_ann: float = 0.0, ann: int = 252) -> float:
        sr = r.mean() / r.std(ddof=1)
        # The implementation uses pandas' default ddof=1 std.
        sr_star = benchmark_ann / sqrt(ann)
        g1 = stats.skew(r, bias=False)
        g2 = stats.kurtosis(r, fisher=False, bias=False)
        denom = sqrt(1.0 - g1 * sr + ((g2 - 1.0) / 4.0) * sr ** 2)
        z = (sr - sr_star) * sqrt(len(r) - 1) / denom
        return float(stats.norm.cdf(z))

    def test_matches_hand_computed_formula(self):
        rng = np.random.default_rng(20240101)
        r = rng.normal(0.0006, 0.011, 750)
        expected = self._reference_psr(r)
        actual = probabilistic_sharpe_ratio(pd.Series(r))
        assert actual == pytest.approx(expected, abs=1e-12)

    def test_matches_hand_computed_with_benchmark(self):
        rng = np.random.default_rng(20240102)
        r = rng.normal(0.0008, 0.010, 900)
        expected = self._reference_psr(r, benchmark_ann=0.5)
        actual = probabilistic_sharpe_ratio(pd.Series(r), benchmark_sharpe=0.5)
        assert actual == pytest.approx(expected, abs=1e-12)

    def test_exact_value_on_a_fixed_symmetric_sample(self):
        """
        A fully deterministic case with no RNG at all.

        Returns alternate +0.02 / -0.01 for 400 observations:
          mean = 0.005, and both skew and excess kurtosis are exactly
          determined by the two-point distribution.  PSR must be ~1.0 because
          a 0.005/0.015 per-period Sharpe over 400 observations is
          overwhelming evidence against SR* = 0.
        """
        r = np.array([0.02, -0.01] * 200)
        expected = self._reference_psr(r)
        actual = probabilistic_sharpe_ratio(pd.Series(r))
        assert actual == pytest.approx(expected, abs=1e-12)
        assert actual > 0.999999

    def test_psr_is_bounded_and_monotone_in_sample_size(self):
        rng = np.random.default_rng(7)
        base = rng.normal(0.0005, 0.01, 4000)
        psrs = [probabilistic_sharpe_ratio(pd.Series(base[:n])) for n in (250, 500, 1000, 4000)]
        assert all(0.0 <= p <= 1.0 for p in psrs)
        # More evidence for the same edge → more confidence.
        assert psrs == sorted(psrs)

    def test_returns_nan_when_not_computable(self):
        assert np.isnan(probabilistic_sharpe_ratio(pd.Series([0.01, 0.02])))
        assert np.isnan(probabilistic_sharpe_ratio(pd.Series([0.01] * 50)))  # zero variance


# ---------------------------------------------------------------------------
# 2. Deflated Sharpe — the bug this suite exists to prevent coming back
# ---------------------------------------------------------------------------

class TestDeflatedSharpeRatio:
    """
    The old implementation returned a CONSTANT: with one trial the hurdle
    short-circuited to the strategy's own Sharpe, so z = 0 and DSR = 0.5
    exactly, regardless of the returns.  A losing strategy got hurdle 0 and
    DSR degraded silently to the un-deflated PSR.
    """

    @staticmethod
    def _returns(seed: int = 3, n: int = 1200) -> pd.Series:
        rng = np.random.default_rng(seed)
        return pd.Series(rng.normal(0.0007, 0.01, n))

    def test_single_trial_is_nan_not_a_number(self):
        r = self._returns()
        assert np.isnan(deflated_sharpe_ratio(r, [2.03], n_trials=1))
        assert np.isnan(expected_maximum_sharpe([2.03], n_trials=1))

    def test_single_trial_does_not_return_one_half(self):
        """The specific old failure: DSR was exactly 0.5 for every strategy."""
        for seed in (1, 2, 3, 4, 5):
            dsr = deflated_sharpe_ratio(self._returns(seed), [1.5], n_trials=1)
            assert np.isnan(dsr), f"seed {seed} produced a number, not NaN"

    def test_dsr_strictly_decreases_as_trials_grow(self):
        r = self._returns()
        rng = np.random.default_rng(99)
        trial_sharpes = list(rng.normal(0.4, 0.6, 30))

        values = [
            deflated_sharpe_ratio(r, trial_sharpes, n_trials=n)
            for n in (2, 5, 10, 25, 50, 100, 250, 1000)
        ]
        assert all(np.isfinite(v) for v in values)
        for earlier, later in zip(values, values[1:]):
            assert later < earlier, f"DSR did not decrease: {values}"

    def test_hurdle_grows_with_trials_and_ignores_trial_mean(self):
        """
        Under the null E[SR] = 0, so the hurdle must depend only on the
        DISPERSION of the trial Sharpes, never on their mean.  Adding a
        constant to every trial Sharpe must leave the hurdle unchanged.
        """
        base = [0.1, 0.5, 0.9, 1.3, 1.7]
        shifted = [s + 10.0 for s in base]
        assert expected_maximum_sharpe(base, n_trials=20) == pytest.approx(
            expected_maximum_sharpe(shifted, n_trials=20), abs=1e-12
        )

    def test_hurdle_matches_bailey_lopez_de_prado_formula(self):
        trials = [0.2, 0.6, 1.1, 1.4, 2.0, 0.3]
        n = 40
        sigma = float(pd.Series(trials).std(ddof=1))
        expected = sigma * (
            (1.0 - EULER_MASCHERONI) * stats.norm.ppf(1.0 - 1.0 / n)
            + EULER_MASCHERONI * stats.norm.ppf(1.0 - 1.0 / (n * e))
        )
        assert expected_maximum_sharpe(trials, n_trials=n) == pytest.approx(expected, abs=1e-12)

    def test_zero_dispersion_is_not_evaluable(self):
        assert np.isnan(expected_maximum_sharpe([1.0, 1.0, 1.0], n_trials=3))
        assert np.isnan(deflated_sharpe_ratio(self._returns(), [1.0, 1.0, 1.0], n_trials=3))

    def test_dsr_never_silently_equals_psr(self):
        """
        A losing strategy used to get hurdle 0, making DSR identical to the
        un-deflated PSR while still being labelled "deflated".
        """
        rng = np.random.default_rng(11)
        losing = pd.Series(rng.normal(-0.0004, 0.01, 900))
        trials = list(rng.normal(0.0, 0.8, 12))

        psr = probabilistic_sharpe_ratio(losing)
        dsr = deflated_sharpe_ratio(losing, trials, n_trials=12)
        assert np.isfinite(dsr)
        assert dsr < psr, "DSR must be strictly below the un-deflated PSR"

    def test_audit_marks_non_evaluable_gate_and_refuses_deployable(self):
        rng = np.random.default_rng(21)
        idx = pd.bdate_range("2019-01-01", periods=800)
        strategy = pd.Series(rng.normal(0.0012, 0.008, len(idx)), index=idx)

        report = audit_strategy(strategy, None, "solo", n_trials=1)

        assert report["deflated_sharpe"] is None       # NaN → null after JSON clean
        assert report["deflated_sharpe_evaluable"] is False
        assert "Not evaluable" in report["deflated_sharpe_reason"]

        dsr_gate = next(g for g in report["gates"] if g["name"] == "Deflated Sharpe")
        assert dsr_gate["pass"] is None
        assert dsr_gate["evaluable"] is False
        # A gate that could not be evaluated must not be read as a pass.
        assert report["verdict"] != "deployable"

    def test_audit_reports_the_trial_count_it_deflated_against(self):
        rng = np.random.default_rng(31)
        idx = pd.bdate_range("2019-01-01", periods=600)
        peers = pd.DataFrame(
            {f"s{i}": rng.normal(0.0003, 0.01, len(idx)) for i in range(5)}, index=idx
        )
        report = audit_strategy(peers["s0"], peers, "s0", n_trials=250)
        assert report["n_trials"] == 250
        assert report["n_peer_strategies"] == 5
        assert report["expected_max_sharpe_hurdle"] is not None


# ---------------------------------------------------------------------------
# 3. PBO — 0.5 on iid strategies, -> 1 on a deliberately overfit set
# ---------------------------------------------------------------------------

class TestProbabilityOfBacktestOverfitting:

    def test_pbo_is_about_half_for_iid_strategies(self):
        """
        When every strategy is drawn from the same distribution, the
        in-sample winner is a coin flip out of sample, so PBO ~= 0.5.

        Averaged over several seeds to keep the assertion tight without being
        flaky on any single draw.
        """
        pbos = []
        for seed in range(8):
            rng = np.random.default_rng(1000 + seed)
            idx = pd.bdate_range("2018-01-01", periods=1000)
            matrix = pd.DataFrame(
                rng.normal(0.0, 0.01, size=(len(idx), 10)),
                index=idx,
                columns=[f"s{i}" for i in range(10)],
            )
            result = estimate_backtest_overfitting(matrix, n_splits=10)
            assert result["pbo"] is not None
            pbos.append(result["pbo"])

        mean_pbo = float(np.mean(pbos))
        assert 0.35 <= mean_pbo <= 0.65, f"iid PBO should sit near 0.5, got {mean_pbo:.3f} from {pbos}"

    def test_pbo_approaches_one_for_a_deliberately_overfit_set(self):
        """
        Engineer a set where the in-sample winner is necessarily the
        out-of-sample loser, for EVERY train/test partition CSCV can form.

        Construction: give each strategy a per-block mean drawn at random,
        then subtract that strategy's own average so its block means sum to
        zero.  Because the total is pinned at zero, whatever a strategy earns
        in the training blocks it must give back in the complement.  Picking
        the best strategy in sample therefore picks the worst out of sample —
        the definition of a backtest-overfit selection, and PBO must be ~1.

        This is stronger than a first-half/second-half construction, which
        only reverses when the train blocks happen to be the early ones.
        """
        rng = np.random.default_rng(4242)
        n_splits = 8
        n = 1200
        idx = pd.bdate_range("2018-01-01", periods=n)

        # Mirror the engine's own partitioning so the block means line up.
        partitions = np.array_split(np.arange(n), n_splits)

        n_strategies = 10
        block_means = rng.normal(0.0, 0.004, size=(n_strategies, n_splits))
        block_means -= block_means.mean(axis=1, keepdims=True)   # each row sums to 0

        columns = {}
        for i in range(n_strategies):
            series = rng.normal(0.0, 0.0015, n)                  # small noise
            for b, block in enumerate(partitions):
                series[block] += block_means[i, b]
            columns[f"overfit_{i}"] = series

        matrix = pd.DataFrame(columns, index=idx)
        result = estimate_backtest_overfitting(matrix, n_splits=n_splits)

        assert result["pbo"] is not None
        assert result["pbo"] > 0.9, (
            f"engineered overfit set should have PBO near 1, got {result['pbo']}"
        )
        # The selected strategy should also lose money out of sample.
        assert result["loss_probability"] > 0.8

    def test_pbo_reports_not_evaluable_rather_than_guessing(self):
        tiny = pd.DataFrame({"a": [0.01, -0.01, 0.02], "b": [0.0, 0.01, -0.01]})
        result = estimate_backtest_overfitting(tiny)
        assert result["pbo"] is None
        assert result["reason"]


# ---------------------------------------------------------------------------
# Trial registry
# ---------------------------------------------------------------------------

class TestTrialRegistry:

    def test_counts_every_recorded_trial(self):
        reg = TrialRegistry()
        for i in range(5):
            reg.record(f"f{i}", 0.1 * i)
        assert reg.count() == 5
        assert len(reg.sharpes()) == 5

    def test_count_survives_retention_cap(self):
        reg = TrialRegistry()
        reg.MAX_RETAINED = 10
        for i in range(50):
            reg.record(f"f{i}", 1.0)
        assert reg.count() == 50          # the burden is not forgotten
        assert len(reg.sharpes()) == 10   # only the detail is capped

    def test_reset_clears_everything(self):
        reg = TrialRegistry()
        reg.record("a", 1.0)
        reg.reset()
        assert reg.count() == 0
        assert reg.sharpes() == []

    def test_persists_to_disk_when_configured(self, tmp_path):
        path = tmp_path / "trials.json"
        reg = TrialRegistry(persist_path=str(path))
        reg.record("a", 1.5)
        reg.record("b", 0.5)

        # A restart must not reset the multiple-testing burden.
        reloaded = TrialRegistry(persist_path=str(path))
        assert reloaded.count() == 2
        assert reloaded.sharpes() == [1.5, 0.5]

    def test_corrupt_log_does_not_break_a_research_run(self, tmp_path):
        path = tmp_path / "trials.json"
        path.write_text("{not valid json", encoding="utf-8")
        reg = TrialRegistry(persist_path=str(path))
        assert reg.count() == 0
