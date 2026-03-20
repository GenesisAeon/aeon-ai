"""Tests for aeon_ai.crep_evaluator — CREPScore + CREPEvaluator."""

from __future__ import annotations

import pytest

from aeon_ai.crep_evaluator import CREPEvaluator, CREPScore

# ---------------------------------------------------------------------------
# CREPScore
# ---------------------------------------------------------------------------


class TestCREPScore:
    def test_defaults_valid(self) -> None:
        score = CREPScore(coherence=0.8, resonance=0.7, emergence=0.6, poetics=0.9)
        assert 0.0 <= score.score <= 1.0

    def test_clamping_above_one(self) -> None:
        score = CREPScore(coherence=1.5, resonance=0.5, emergence=0.5, poetics=0.5)
        assert score.coherence == 1.0

    def test_clamping_below_zero(self) -> None:
        score = CREPScore(coherence=-0.2, resonance=0.5, emergence=0.5, poetics=0.5)
        assert score.coherence == 0.0

    def test_harmonic_mean_computation(self) -> None:
        score = CREPScore(coherence=0.5, resonance=0.5, emergence=0.5, poetics=0.5)
        assert abs(score.harmonic_mean - 0.5) < 1e-9

    def test_harmonic_mean_zero_when_any_dim_zero(self) -> None:
        score = CREPScore(coherence=0.0, resonance=0.8, emergence=0.8, poetics=0.8)
        assert score.harmonic_mean == 0.0

    def test_weighted_mean_equal_weights(self) -> None:
        score = CREPScore(coherence=0.6, resonance=0.4, emergence=0.8, poetics=0.2)
        expected = (0.6 + 0.4 + 0.8 + 0.2) / 4
        assert abs(score.weighted_mean - expected) < 1e-9

    def test_custom_weights(self) -> None:
        score = CREPScore(
            coherence=1.0, resonance=0.0, emergence=0.0, poetics=0.0,
            weights=(1.0, 0.0, 0.0, 0.0),
        )
        assert abs(score.weighted_mean - 1.0) < 1e-9

    def test_invalid_weights_length(self) -> None:
        with pytest.raises(ValueError, match="exactly 4"):
            CREPScore(0.5, 0.5, 0.5, 0.5, weights=(0.5, 0.5))  # type: ignore[arg-type]

    def test_invalid_weights_sum(self) -> None:
        with pytest.raises(ValueError, match="sum to 1.0"):
            CREPScore(0.5, 0.5, 0.5, 0.5, weights=(0.3, 0.3, 0.3, 0.3))

    def test_as_dict_keys(self) -> None:
        score = CREPScore(0.7, 0.6, 0.5, 0.8)
        d = score.as_dict()
        assert set(d.keys()) == {
            "coherence", "resonance", "emergence", "poetics",
            "harmonic_mean", "weighted_mean",
        }

    def test_repr_contains_crep(self) -> None:
        score = CREPScore(0.5, 0.5, 0.5, 0.5)
        assert "CREPScore" in repr(score)

    def test_score_property_equals_harmonic_mean(self) -> None:
        score = CREPScore(0.8, 0.7, 0.6, 0.9)
        assert score.score == score.harmonic_mean


# ---------------------------------------------------------------------------
# CREPEvaluator
# ---------------------------------------------------------------------------


class TestCREPEvaluator:
    def test_evaluate_returns_crep_score(self, crep_evaluator: CREPEvaluator) -> None:
        score = crep_evaluator.evaluate(signal=[0.3, 0.7, 0.5])
        assert isinstance(score, CREPScore)

    def test_evaluate_score_in_range(self, crep_evaluator: CREPEvaluator) -> None:
        score = crep_evaluator.evaluate(signal=[0.1, 0.9, 0.5, 0.3])
        assert 0.0 <= score.score <= 1.0

    def test_evaluate_empty_signal(self, crep_evaluator: CREPEvaluator) -> None:
        score = crep_evaluator.evaluate(signal=[])
        assert isinstance(score, CREPScore)

    def test_evaluate_with_text(self, crep_evaluator: CREPEvaluator) -> None:
        score = crep_evaluator.evaluate(text="the mirror reflects the aeon")
        assert score.poetics > 0.0

    def test_evaluate_repeated_words_lower_ttr(self, crep_evaluator: CREPEvaluator) -> None:
        score_unique = crep_evaluator.evaluate(text="alpha beta gamma delta epsilon")
        score_repeat = crep_evaluator.evaluate(text="the the the the the")
        assert score_unique.poetics > score_repeat.poetics

    def test_evaluate_external_overrides(self, crep_evaluator: CREPEvaluator) -> None:
        score = crep_evaluator.evaluate(
            external={"coherence": 0.9, "resonance": 0.8, "emergence": 0.7, "poetics": 0.6}
        )
        assert abs(score.coherence - 0.9) < 1e-9
        assert abs(score.resonance - 0.8) < 1e-9

    def test_history_accumulates(self, crep_evaluator: CREPEvaluator) -> None:
        crep_evaluator.evaluate(signal=[0.5])
        crep_evaluator.evaluate(signal=[0.3])
        assert len(crep_evaluator.history) == 2

    def test_reset_clears_history(self, crep_evaluator: CREPEvaluator) -> None:
        crep_evaluator.evaluate(signal=[0.5])
        crep_evaluator.reset()
        assert len(crep_evaluator.history) == 0

    def test_evaluate_batch(self, crep_evaluator: CREPEvaluator) -> None:
        items = [{"signal": [0.5]}, {"text": "aeon mirror"}]
        results = crep_evaluator.evaluate_batch(items)
        assert len(results) == 2
        assert all(isinstance(r, CREPScore) for r in results)

    def test_invalid_weights(self) -> None:
        with pytest.raises(ValueError, match="sum to 1.0"):
            CREPEvaluator(weights=(0.5, 0.5, 0.5, 0.5))

    def test_coherence_single_element_is_one(self, crep_evaluator: CREPEvaluator) -> None:
        # Single element has no variance → coherence = 1
        score = crep_evaluator.evaluate(signal=[0.5])
        assert abs(score.coherence - 1.0) < 1e-9

    def test_resonance_constant_signal(self, crep_evaluator: CREPEvaluator) -> None:
        """Constant signal has perfect auto-correlation → resonance = 1.0."""
        score = crep_evaluator.evaluate(signal=[0.5, 0.5, 0.5, 0.5])
        assert abs(score.resonance - 1.0) < 1e-9

    def test_repr(self, crep_evaluator: CREPEvaluator) -> None:
        assert "CREPEvaluator" in repr(crep_evaluator)

    def test_entropy_scale_affects_emergence(self) -> None:
        ev1 = CREPEvaluator(entropy_scale=1.0)
        ev2 = CREPEvaluator(entropy_scale=0.1)
        s1 = ev1.evaluate(signal=[0.1, 0.9])
        s2 = ev2.evaluate(signal=[0.1, 0.9])
        assert s1.emergence != s2.emergence
