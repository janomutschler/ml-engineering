"""Tests for the champion/challenger promotion gate."""

from types import SimpleNamespace

import pytest
from mlflow.exceptions import MlflowException

from bike_rental.defs.assets.promotion import (
    CHALLENGER_ALIAS,
    CHAMPION_ALIAS,
    PromotionConfig,
    PromotionDecision,
    evaluate_promotion,
    run_promotion,
)

NAME = "bike_rental_forecaster"


# --------------------------------------------------------------------------- #
# Pure policy: evaluate_promotion                                             #
# --------------------------------------------------------------------------- #


def test_bootstrap_promotes_when_no_champion():
    """With no incumbent, the candidate becomes champion regardless of metric."""
    decision = evaluate_promotion(0.90, None, higher_is_better=True, min_improvement=0.0)
    assert decision == PromotionDecision(True, CHAMPION_ALIAS, None)


def test_strict_improvement_promotes():
    """A higher metric than the champion is promoted (higher-is-better)."""
    decision = evaluate_promotion(0.90, 0.85, higher_is_better=True, min_improvement=0.0)
    assert decision.promote and decision.alias == CHAMPION_ALIAS
    assert decision.improvement == pytest.approx(0.05)


def test_equal_metric_is_not_an_improvement():
    """An equal metric is not a strict improvement, so it becomes challenger."""
    decision = evaluate_promotion(0.90, 0.90, higher_is_better=True, min_improvement=0.0)
    assert not decision.promote and decision.alias == CHALLENGER_ALIAS


def test_worse_metric_becomes_challenger():
    """A worse candidate becomes challenger."""
    decision = evaluate_promotion(0.80, 0.90, higher_is_better=True, min_improvement=0.0)
    assert not decision.promote and decision.alias == CHALLENGER_ALIAS


def test_min_improvement_gate_is_respected():
    """Improvement must exceed min_improvement, not merely be positive."""
    below = evaluate_promotion(0.905, 0.90, higher_is_better=True, min_improvement=0.01)
    above = evaluate_promotion(0.920, 0.90, higher_is_better=True, min_improvement=0.01)
    assert not below.promote
    assert above.promote


def test_lower_is_better_inverts_the_comparison():
    """For an error metric (e.g. RMSE) a smaller candidate value wins."""
    better = evaluate_promotion(50.0, 60.0, higher_is_better=False, min_improvement=0.0)
    worse = evaluate_promotion(70.0, 60.0, higher_is_better=False, min_improvement=0.0)
    assert better.promote and better.improvement == 10.0
    assert not worse.promote


# --------------------------------------------------------------------------- #
# Orchestration: run_promotion with a fake registry client                   #
# --------------------------------------------------------------------------- #


class _FakeClient:
    """Minimal stand-in for ``MlflowClient`` covering the methods used."""

    def __init__(self, metrics: dict[str, dict[str, float]], champion_version: str | None = None):
        self._metrics = metrics
        self._champion = champion_version
        self.alias_calls: list[tuple[str, str]] = []

    def get_model_version(self, name, version):
        return SimpleNamespace(run_id=version)

    def get_run(self, run_id):
        return SimpleNamespace(data=SimpleNamespace(metrics=self._metrics[run_id]))

    def get_model_version_by_alias(self, name, alias):
        if alias == CHAMPION_ALIAS and self._champion is None:
            raise MlflowException("no champion set")
        return SimpleNamespace(version=self._champion)

    def set_registered_model_alias(self, name, alias, version):
        self.alias_calls.append((alias, version))


def test_run_promotion_bootstraps_first_champion():
    """The first version is set as champion and metadata records the bootstrap."""
    client = _FakeClient({"1": {"mean_r2": 0.85}})

    promoted, summary, metadata = run_promotion(client, NAME, "1", PromotionConfig())

    assert promoted
    assert (CHAMPION_ALIAS, "1") in client.alias_calls
    assert "bootstrap" in summary
    assert metadata["champion_version"] == "none"
    assert metadata["improvement"] == "n/a (bootstrap)"


def test_run_promotion_moves_champion_on_improvement():
    """A strictly better version takes the @champion alias."""
    client = _FakeClient({"1": {"mean_r2": 0.85}, "2": {"mean_r2": 0.90}}, champion_version="1")

    promoted, _, _ = run_promotion(client, NAME, "2", PromotionConfig())

    assert promoted
    assert (CHAMPION_ALIAS, "2") in client.alias_calls
    assert all(alias != CHALLENGER_ALIAS for alias, _ in client.alias_calls)


def test_run_promotion_keeps_champion_on_no_improvement():
    """A non-improving version becomes @challenger and the champion is untouched."""
    client = _FakeClient({"1": {"mean_r2": 0.90}, "2": {"mean_r2": 0.90}}, champion_version="1")

    promoted, summary, _ = run_promotion(client, NAME, "2", PromotionConfig())

    assert not promoted
    assert client.alias_calls == [(CHALLENGER_ALIAS, "2")]
    assert "champion remains v1" in summary


def test_run_promotion_honours_lower_is_better_metric():
    """An RMSE-style metric promotes the version with the smaller value."""
    client = _FakeClient({"1": {"rmse": 60.0}, "2": {"rmse": 50.0}}, champion_version="1")

    promoted, _, _ = run_promotion(
        client, NAME, "2", PromotionConfig(metric="rmse", higher_is_better=False)
    )

    assert promoted
    assert (CHAMPION_ALIAS, "2") in client.alias_calls
