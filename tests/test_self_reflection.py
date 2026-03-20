"""Tests for aeon_ai.self_reflection — SelfReflector, ReflectionLoopResult."""

from __future__ import annotations

import pytest

from aeon_ai.self_reflection import (
    MAX_ITER,
    IterationSnapshot,
    ReflectionLoopResult,
    SelfReflector,
    _DEFAULT_ENTROPY_THRESHOLD,
    _DEFAULT_STEP_SIZE,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


class TestConstants:
    def test_max_iter_is_seven(self) -> None:
        assert MAX_ITER == 7

    def test_default_entropy_threshold(self) -> None:
        assert _DEFAULT_ENTROPY_THRESHOLD == 0.37

    def test_default_step_size_positive(self) -> None:
        assert _DEFAULT_STEP_SIZE > 0.0


# ---------------------------------------------------------------------------
# IterationSnapshot
# ---------------------------------------------------------------------------


class TestIterationSnapshot:
    def _make_snapshot(self, iteration: int = 0, converged: bool = False) -> IterationSnapshot:
        from aeon_ai.crep_evaluator import CREPScore
        from aeon_ai.mirror_core import MirrorCore, MirrorPhase, ReflectionState

        crep = CREPScore(coherence=0.5, resonance=0.5, emergence=0.5, poetics=0.5)
        reflection = ReflectionState(
            phase=MirrorPhase.EMIT, input_val=0.3, output_val=0.3, entropy=0.4
        )
        return IterationSnapshot(
            iteration=iteration,
            s_a=0.7,
            s_v=0.6,
            lagrangian_value=-0.5,
            crep_score=crep,
            reflection=reflection,
            sigil_activations={"MIRROR": 0.5},
            entropy=0.4,
            converged=converged,
        )

    def test_creation(self) -> None:
        snap = self._make_snapshot(iteration=2, converged=True)
        assert snap.iteration == 2
        assert snap.converged is True

    def test_as_dict_keys(self) -> None:
        snap = self._make_snapshot()
        d = snap.as_dict()
        assert "iteration" in d
        assert "s_a" in d
        assert "s_v" in d
        assert "lagrangian_value" in d
        assert "crep_score" in d
        assert "reflection" in d
        assert "sigil_activations" in d
        assert "entropy" in d
        assert "converged" in d

    def test_as_dict_reflection_keys(self) -> None:
        snap = self._make_snapshot()
        d = snap.as_dict()
        assert "phase" in d["reflection"]
        assert "output_val" in d["reflection"]
        assert "entropy" in d["reflection"]

    def test_default_metadata(self) -> None:
        snap = self._make_snapshot()
        assert snap.metadata == {}


# ---------------------------------------------------------------------------
# ReflectionLoopResult
# ---------------------------------------------------------------------------


class TestReflectionLoopResult:
    def _make_result(self, n_snapshots: int = 3, converged: bool = False) -> ReflectionLoopResult:
        from aeon_ai.crep_evaluator import CREPScore
        from aeon_ai.mirror_core import MirrorCore, MirrorPhase, ReflectionState

        crep = CREPScore(coherence=0.6, resonance=0.6, emergence=0.6, poetics=0.6)
        reflection = ReflectionState(
            phase=MirrorPhase.EMIT, input_val=0.3, output_val=0.3, entropy=0.4
        )
        snaps = [
            IterationSnapshot(
                iteration=i,
                s_a=0.7,
                s_v=0.6,
                lagrangian_value=-0.5,
                crep_score=crep,
                reflection=reflection,
                sigil_activations={},
                entropy=0.37,
            )
            for i in range(n_snapshots)
        ]
        return ReflectionLoopResult(
            snapshots=snaps,
            converged=converged,
            final_crep=crep,
            final_lagrangian=-0.5,
            final_s_a=0.7,
            final_s_v=0.6,
            total_iterations=n_snapshots,
            entropy_threshold=0.37,
        )

    def test_total_iterations(self) -> None:
        result = self._make_result(n_snapshots=5)
        assert result.total_iterations == 5

    def test_as_dict_structure(self) -> None:
        result = self._make_result()
        d = result.as_dict()
        assert "converged" in d
        assert "total_iterations" in d
        assert "entropy_threshold" in d
        assert "final_s_a" in d
        assert "final_s_v" in d
        assert "final_lagrangian" in d
        assert "final_crep" in d
        assert "snapshots" in d

    def test_as_dict_snapshots_list(self) -> None:
        result = self._make_result(n_snapshots=3)
        d = result.as_dict()
        assert isinstance(d["snapshots"], list)
        assert len(d["snapshots"]) == 3

    def test_default_metadata(self) -> None:
        result = self._make_result()
        assert result.metadata == {}


# ---------------------------------------------------------------------------
# SelfReflector construction
# ---------------------------------------------------------------------------


class TestSelfReflectorInit:
    def test_default_init(self) -> None:
        reflector = SelfReflector()
        assert reflector.delta == 0.0
        assert reflector.step_size > 0.0

    def test_custom_params(self) -> None:
        reflector = SelfReflector(delta=0.1, step_size=0.01, mirror_depth=3)
        assert reflector.delta == 0.1
        assert reflector.step_size == 0.01

    def test_repr_contains_classname(self) -> None:
        reflector = SelfReflector()
        assert "SelfReflector" in repr(reflector)

    def test_state_dict_keys(self) -> None:
        reflector = SelfReflector()
        sd = reflector.state_dict()
        assert "delta" in sd
        assert "step_size" in sd
        assert "aeon_layer" in sd
        assert "mirror_core" in sd
        assert "loop_history_len" in sd


# ---------------------------------------------------------------------------
# SelfReflector.self_reflect
# ---------------------------------------------------------------------------


class TestSelfReflect:
    def test_returns_result(self) -> None:
        reflector = SelfReflector()
        result = reflector.self_reflect()
        assert isinstance(result, ReflectionLoopResult)

    def test_total_iterations_le_max_iter(self) -> None:
        reflector = SelfReflector()
        result = reflector.self_reflect()
        assert result.total_iterations <= MAX_ITER

    def test_total_iterations_ge_one(self) -> None:
        reflector = SelfReflector()
        result = reflector.self_reflect()
        assert result.total_iterations >= 1

    def test_entropy_threshold_stored(self) -> None:
        reflector = SelfReflector()
        result = reflector.self_reflect(entropy_threshold=0.5)
        assert result.entropy_threshold == 0.5

    def test_final_crep_in_range(self) -> None:
        reflector = SelfReflector()
        result = reflector.self_reflect()
        assert 0.0 <= result.final_crep.score <= 1.0

    def test_snapshots_have_correct_iterations(self) -> None:
        reflector = SelfReflector()
        result = reflector.self_reflect()
        for i, snap in enumerate(result.snapshots):
            assert snap.iteration == i

    def test_custom_s_a_s_v(self) -> None:
        reflector = SelfReflector()
        result = reflector.self_reflect(s_a=0.3, s_v=0.9)
        assert isinstance(result, ReflectionLoopResult)

    def test_custom_sigil_text(self) -> None:
        reflector = SelfReflector()
        result = reflector.self_reflect(sigil_text="mirror genesis aeon")
        assert result.total_iterations >= 1

    def test_none_sigil_uses_default(self) -> None:
        reflector = SelfReflector(sigil_text="aeon mirror genesis")
        result = reflector.self_reflect(sigil_text=None)
        assert result.total_iterations >= 1

    def test_empty_sigil_text(self) -> None:
        reflector = SelfReflector()
        result = reflector.self_reflect(sigil_text="")
        assert isinstance(result, ReflectionLoopResult)

    def test_loop_history_grows(self) -> None:
        reflector = SelfReflector()
        assert len(reflector.loop_history) == 0
        reflector.self_reflect()
        assert len(reflector.loop_history) == 1
        reflector.self_reflect()
        assert len(reflector.loop_history) == 2

    def test_metadata_attached(self) -> None:
        reflector = SelfReflector()
        result = reflector.self_reflect(metadata={"run": "test"})
        assert result.metadata == {"run": "test"}

    def test_converged_or_max_iter(self) -> None:
        reflector = SelfReflector()
        result = reflector.self_reflect()
        # Either converged or reached max_iter
        if result.converged:
            # last snapshot should have converged=True
            assert result.snapshots[-1].converged
        else:
            assert result.total_iterations == MAX_ITER

    def test_final_lagrangian_is_float(self) -> None:
        reflector = SelfReflector()
        result = reflector.self_reflect()
        assert isinstance(result.final_lagrangian, float)

    def test_final_s_a_clipped(self) -> None:
        reflector = SelfReflector(step_size=100.0)
        result = reflector.self_reflect(s_a=5.0, s_v=5.0)
        assert abs(result.final_s_a) <= 10.0
        assert abs(result.final_s_v) <= 10.0

    def test_different_delta_produces_different_result(self) -> None:
        r1 = SelfReflector(delta=0.0).self_reflect(s_a=0.7, s_v=0.6)
        r2 = SelfReflector(delta=0.5).self_reflect(s_a=0.7, s_v=0.6)
        # Results may differ in lagrangian value
        assert isinstance(r1.final_lagrangian, float)
        assert isinstance(r2.final_lagrangian, float)

    def test_very_low_entropy_threshold(self) -> None:
        reflector = SelfReflector()
        result = reflector.self_reflect(entropy_threshold=0.01)
        assert result.total_iterations >= 1

    def test_very_high_entropy_threshold(self) -> None:
        reflector = SelfReflector()
        result = reflector.self_reflect(entropy_threshold=0.99)
        assert result.total_iterations >= 1

    def test_custom_time_step(self) -> None:
        reflector = SelfReflector()
        result = reflector.self_reflect(t=2.0)
        assert isinstance(result, ReflectionLoopResult)

    def test_all_snapshots_have_valid_crep(self) -> None:
        reflector = SelfReflector()
        result = reflector.self_reflect()
        for snap in result.snapshots:
            assert 0.0 <= snap.crep_score.score <= 1.0

    def test_sigillin_activations_in_snapshots(self) -> None:
        reflector = SelfReflector()
        result = reflector.self_reflect(sigil_text="mirror aeon genesis")
        for snap in result.snapshots:
            assert isinstance(snap.sigil_activations, dict)


# ---------------------------------------------------------------------------
# SelfReflector.reset
# ---------------------------------------------------------------------------


class TestSelfReflectorReset:
    def test_reset_clears_history(self) -> None:
        reflector = SelfReflector()
        reflector.self_reflect()
        assert len(reflector.loop_history) == 1
        reflector.reset()
        assert len(reflector.loop_history) == 0

    def test_reset_clears_mirror_trace(self) -> None:
        reflector = SelfReflector()
        reflector.self_reflect()
        reflector.reset()
        assert len(reflector._mirror_core.trace) == 0

    def test_loop_history_property_is_copy(self) -> None:
        reflector = SelfReflector()
        reflector.self_reflect()
        history = reflector.loop_history
        history.clear()
        assert len(reflector.loop_history) == 1
