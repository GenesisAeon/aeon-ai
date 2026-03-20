"""Tests for aeon_ai.agents — Orchestrator + OrchestratorResult."""

from __future__ import annotations

import pytest

from aeon_ai.agents import Orchestrator, OrchestratorResult
from aeon_ai.crep_evaluator import CREPScore
from aeon_ai.mirror_core import MirrorPhase

# ---------------------------------------------------------------------------
# OrchestratorResult
# ---------------------------------------------------------------------------


class TestOrchestratorResult:
    @pytest.fixture
    def result(self, orchestrator: Orchestrator) -> OrchestratorResult:
        return orchestrator.run(s_a=0.7, s_v=0.5, sigil_text="mirror aeon")

    def test_final_output_is_modulated(self, result: OrchestratorResult) -> None:
        expected = result.reflection.output_val * result.modulation
        assert abs(result.final_output - expected) < 1e-9

    def test_as_dict_has_expected_keys(self, result: OrchestratorResult) -> None:
        d = result.as_dict()
        assert "input" in d
        assert "lagrangian_out" in d
        assert "final_output" in d
        assert "reflection" in d
        assert "crep" in d
        assert "sigil_activations" in d
        assert "cosmic_moment" in d

    def test_as_dict_reflection_phase_name(self, result: OrchestratorResult) -> None:
        assert result.as_dict()["reflection"]["phase"] == "EMIT"

    def test_crep_score_type(self, result: OrchestratorResult) -> None:
        assert isinstance(result.crep_score, CREPScore)

    def test_emission_phase(self, result: OrchestratorResult) -> None:
        assert result.reflection.phase == MirrorPhase.EMIT


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


class TestOrchestrator:
    def test_run_returns_result(self, orchestrator: Orchestrator) -> None:
        result = orchestrator.run(s_a=0.7, s_v=0.6)
        assert isinstance(result, OrchestratorResult)

    def test_run_with_sigil_text(self, orchestrator: Orchestrator) -> None:
        result = orchestrator.run(s_a=0.5, s_v=0.5, sigil_text="genesis origin seed")
        assert isinstance(result.sigil_activations, dict)

    def test_run_with_sigil_activates_genesis(self, orchestrator: Orchestrator) -> None:
        result = orchestrator.run(s_a=0.5, s_v=0.5, sigil_text="genesis origin")
        assert "GENESIS" in result.sigil_activations

    def test_run_empty_sigil_text_no_activations(self, orchestrator: Orchestrator) -> None:
        result = orchestrator.run(s_a=0.5, s_v=0.5, sigil_text="")
        assert result.sigil_activations == {}

    def test_run_accumulates_results(self, orchestrator: Orchestrator) -> None:
        orchestrator.run(s_a=0.3, s_v=0.7)
        orchestrator.run(s_a=0.6, s_v=0.4)
        assert len(orchestrator.results) == 2

    def test_reset_clears_results(self, orchestrator: Orchestrator) -> None:
        orchestrator.run(s_a=0.5, s_v=0.5)
        orchestrator.reset()
        assert len(orchestrator.results) == 0

    def test_run_batch(self, orchestrator: Orchestrator) -> None:
        inputs = [
            {"s_a": 0.5, "s_v": 0.4},
            {"s_a": 0.6, "s_v": 0.3, "sigil_text": "mirror"},
        ]
        results = orchestrator.run_batch(inputs)
        assert len(results) == 2

    def test_state_dict_keys(self, orchestrator: Orchestrator) -> None:
        sd = orchestrator.state_dict()
        assert "aeon_layer" in sd
        assert "mirror_core" in sd
        assert "result_count" in sd

    def test_invalid_t_raises(self, orchestrator: Orchestrator) -> None:
        with pytest.raises(ValueError, match="strictly positive"):
            orchestrator.run(s_a=0.5, s_v=0.5, t=0.0)

    def test_custom_signal_passed_to_crep(self, orchestrator: Orchestrator) -> None:
        result = orchestrator.run(
            s_a=0.5, s_v=0.5, signal=[0.1, 0.2, 0.3, 0.4, 0.5]
        )
        assert 0.0 <= result.crep_score.score <= 1.0

    def test_crep_external_override(self, orchestrator: Orchestrator) -> None:
        result = orchestrator.run(
            s_a=0.5, s_v=0.5,
            crep_external={"coherence": 0.99, "resonance": 0.99, "emergence": 0.99, "poetics": 0.99},  # noqa: E501
        )
        assert result.crep_score.coherence == 0.99

    def test_metadata_stored_in_result(self, orchestrator: Orchestrator) -> None:
        result = orchestrator.run(s_a=0.5, s_v=0.5, metadata={"test_key": "test_val"})
        assert result.metadata.get("test_key") == "test_val"

    def test_repr(self, orchestrator: Orchestrator) -> None:
        assert "Orchestrator" in repr(orchestrator)

    def test_results_property_is_copy(self, orchestrator: Orchestrator) -> None:
        orchestrator.run(s_a=0.5, s_v=0.5)
        results = orchestrator.results
        results.clear()
        assert len(orchestrator.results) == 1

    def test_modulation_factor_positive(self, orchestrator: Orchestrator) -> None:
        result = orchestrator.run(s_a=0.7, s_v=0.3)
        assert result.modulation > 0.0

    def test_lagrangian_out_is_float(self, orchestrator: Orchestrator) -> None:
        result = orchestrator.run(s_a=0.7, s_v=0.3)
        assert isinstance(result.lagrangian_out, float)

    def test_no_sigils_loaded(self) -> None:
        orch = Orchestrator(load_sigils=False)
        result = orch.run(s_a=0.5, s_v=0.5, sigil_text="mirror genesis")
        assert result.sigil_activations == {}
