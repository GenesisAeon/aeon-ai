"""Tests for aeon_ai.phase_detector — PhaseDetector, PhaseTransitionEvent, helpers."""

from __future__ import annotations

import pytest

from aeon_ai.mirror_core import MirrorCore, MirrorPhase, ReflectionState
from aeon_ai.phase_detector import (
    PhaseDetector,
    PhaseTransitionEvent,
    TransitionType,
    _PHASE_ORDER,
    _PHASE_SUCCESSOR,
    _entropy_to_phase_label,
    detect_phases_from_core,
    entropy_phase_label,
)


# ---------------------------------------------------------------------------
# TransitionType
# ---------------------------------------------------------------------------


class TestTransitionType:
    def test_all_types_exist(self) -> None:
        assert TransitionType.FORWARD
        assert TransitionType.COLLAPSE
        assert TransitionType.UTAC_TRIGGER
        assert TransitionType.FORCED

    def test_enum_values_are_unique(self) -> None:
        vals = [t.value for t in TransitionType]
        assert len(vals) == len(set(vals))


# ---------------------------------------------------------------------------
# PhaseTransitionEvent
# ---------------------------------------------------------------------------


class TestPhaseTransitionEvent:
    def test_creation(self) -> None:
        ev = PhaseTransitionEvent(
            source_phase=MirrorPhase.INIT,
            target_phase=MirrorPhase.REFLECT,
            transition_type=TransitionType.FORWARD,
            entropy=0.4,
            utac_value=0.6,
        )
        assert ev.source_phase == MirrorPhase.INIT
        assert ev.target_phase == MirrorPhase.REFLECT
        assert ev.entropy == 0.4

    def test_as_dict_keys(self) -> None:
        ev = PhaseTransitionEvent(
            source_phase=MirrorPhase.REFLECT,
            target_phase=MirrorPhase.INTEGRATE,
            transition_type=TransitionType.COLLAPSE,
            entropy=0.3,
            utac_value=0.2,
        )
        d = ev.as_dict()
        assert "source_phase" in d
        assert "target_phase" in d
        assert "transition_type" in d
        assert "entropy" in d
        assert "utac_value" in d
        assert "timestamp" in d

    def test_as_dict_values(self) -> None:
        ev = PhaseTransitionEvent(
            source_phase=MirrorPhase.INTEGRATE,
            target_phase=MirrorPhase.EMIT,
            transition_type=TransitionType.UTAC_TRIGGER,
            entropy=0.8,
            utac_value=0.9,
            metadata={"test": True},
        )
        d = ev.as_dict()
        assert d["source_phase"] == "INTEGRATE"
        assert d["target_phase"] == "EMIT"
        assert d["transition_type"] == "UTAC_TRIGGER"
        assert d["entropy"] == 0.8

    def test_default_metadata(self) -> None:
        ev = PhaseTransitionEvent(
            source_phase=MirrorPhase.EMIT,
            target_phase=MirrorPhase.INIT,
            transition_type=TransitionType.FORCED,
            entropy=0.5,
            utac_value=0.5,
        )
        assert ev.metadata == {}

    def test_timestamp_is_positive(self) -> None:
        ev = PhaseTransitionEvent(
            source_phase=MirrorPhase.INIT,
            target_phase=MirrorPhase.REFLECT,
            transition_type=TransitionType.FORWARD,
            entropy=0.3,
            utac_value=0.4,
        )
        assert ev.timestamp > 0.0


# ---------------------------------------------------------------------------
# PhaseDetector construction
# ---------------------------------------------------------------------------


class TestPhaseDetectorInit:
    def test_default_init(self) -> None:
        detector = PhaseDetector()
        assert detector.entropy_threshold == 0.37
        assert detector.stability_floor > 0.0
        assert detector.utac_trigger_ceil > 0.0

    def test_custom_params(self) -> None:
        detector = PhaseDetector(
            entropy_threshold=0.5,
            stability_floor=1e-3,
            utac_trigger_ceil=0.9,
        )
        assert detector.entropy_threshold == 0.5
        assert detector.stability_floor == 1e-3

    def test_invalid_entropy_threshold_zero(self) -> None:
        with pytest.raises(ValueError, match="entropy_threshold"):
            PhaseDetector(entropy_threshold=0.0)

    def test_invalid_entropy_threshold_one(self) -> None:
        with pytest.raises(ValueError, match="entropy_threshold"):
            PhaseDetector(entropy_threshold=1.0)

    def test_invalid_stability_floor(self) -> None:
        with pytest.raises(ValueError, match="stability_floor"):
            PhaseDetector(stability_floor=0.0)

    def test_invalid_utac_trigger_ceil_zero(self) -> None:
        with pytest.raises(ValueError, match="utac_trigger_ceil"):
            PhaseDetector(utac_trigger_ceil=0.0)

    def test_invalid_utac_trigger_ceil_negative(self) -> None:
        with pytest.raises(ValueError, match="utac_trigger_ceil"):
            PhaseDetector(utac_trigger_ceil=-0.1)

    def test_repr_contains_classname(self) -> None:
        detector = PhaseDetector()
        assert "PhaseDetector" in repr(detector)


# ---------------------------------------------------------------------------
# PhaseDetector.utac_value_at / utac_trigger_check
# ---------------------------------------------------------------------------


class TestUtacMethods:
    def test_utac_value_at_midpoint(self) -> None:
        detector = PhaseDetector(entropy_threshold=0.5)
        val = detector.utac_value_at(0.5)
        # At midpoint, UTAC = L/2 = 0.5 (capacity=1)
        assert abs(val - 0.5) < 0.01

    def test_utac_value_at_high_entropy(self) -> None:
        detector = PhaseDetector(entropy_threshold=0.1)
        val = detector.utac_value_at(0.9)
        assert val > 0.9

    def test_utac_value_at_low_entropy(self) -> None:
        detector = PhaseDetector(entropy_threshold=0.9)
        val = detector.utac_value_at(0.1)
        assert val < 0.1

    def test_utac_trigger_check_triggered(self) -> None:
        # Very low threshold so high entropy triggers
        detector = PhaseDetector(entropy_threshold=0.1, utac_trigger_ceil=0.5)
        assert detector.utac_trigger_check(0.9)

    def test_utac_trigger_check_not_triggered(self) -> None:
        # Very high threshold so low entropy does not trigger
        detector = PhaseDetector(entropy_threshold=0.9, utac_trigger_ceil=0.99)
        assert not detector.utac_trigger_check(0.1)


# ---------------------------------------------------------------------------
# PhaseDetector.detect_collapse / detect_collapse_pair
# ---------------------------------------------------------------------------


class TestCollapseDetection:
    def _make_state(self, val: float, entropy: float, phase: MirrorPhase) -> ReflectionState:
        return ReflectionState(
            phase=phase, input_val=val, output_val=val, entropy=entropy
        )

    def test_collapse_pair_small_delta(self) -> None:
        detector = PhaseDetector(stability_floor=0.01)
        assert detector.detect_collapse_pair(0.5, 0.5001)

    def test_collapse_pair_large_delta(self) -> None:
        detector = PhaseDetector(stability_floor=0.01)
        assert not detector.detect_collapse_pair(0.5, 0.6)

    def test_detect_collapse_true(self) -> None:
        detector = PhaseDetector(entropy_threshold=0.9, stability_floor=0.01)
        trace = [
            self._make_state(0.5, 0.1, MirrorPhase.INIT),
            self._make_state(0.5001, 0.1, MirrorPhase.REFLECT),
        ]
        assert detector.detect_collapse(trace)

    def test_detect_collapse_false_large_delta(self) -> None:
        detector = PhaseDetector(stability_floor=0.001)
        trace = [
            self._make_state(0.0, 0.5, MirrorPhase.INIT),
            self._make_state(0.5, 0.5, MirrorPhase.REFLECT),
        ]
        assert not detector.detect_collapse(trace)

    def test_detect_collapse_entropy_above_threshold(self) -> None:
        # High entropy → no collapse even if delta is small
        detector = PhaseDetector(entropy_threshold=0.2, stability_floor=0.01)
        trace = [
            self._make_state(0.5, 0.9, MirrorPhase.INIT),
            self._make_state(0.5001, 0.9, MirrorPhase.REFLECT),
        ]
        assert not detector.detect_collapse(trace)

    def test_detect_collapse_empty_trace(self) -> None:
        detector = PhaseDetector()
        assert not detector.detect_collapse([])

    def test_detect_collapse_single_state(self) -> None:
        detector = PhaseDetector()
        trace = [self._make_state(0.5, 0.1, MirrorPhase.INIT)]
        assert not detector.detect_collapse(trace)

    def test_detect_collapse_custom_threshold_override(self) -> None:
        detector = PhaseDetector(entropy_threshold=0.5, stability_floor=0.001)
        trace = [
            self._make_state(0.5, 0.1, MirrorPhase.INIT),
            self._make_state(0.5001, 0.1, MirrorPhase.REFLECT),
        ]
        # Override threshold to 0.9 → entropy 0.1 < 0.9 → still collapse
        assert detector.detect_collapse(trace, entropy_threshold=0.9)


# ---------------------------------------------------------------------------
# PhaseDetector.detect_transition
# ---------------------------------------------------------------------------


class TestDetectTransition:
    def test_forward_transition_detected(self) -> None:
        detector = PhaseDetector(entropy_threshold=0.37, utac_trigger_ceil=0.99)
        states = [
            ReflectionState(MirrorPhase.INIT, 0.5, 0.5, 0.3),
            ReflectionState(MirrorPhase.REFLECT, 0.5, 0.45, 0.3),
        ]
        detector.detect_transition(states[0])
        ev = detector.detect_transition(states[1])
        assert ev is not None
        assert ev.transition_type == TransitionType.FORWARD
        assert ev.source_phase == MirrorPhase.INIT
        assert ev.target_phase == MirrorPhase.REFLECT

    def test_no_event_on_same_phase_different_output(self) -> None:
        # When phase does not advance and no collapse/UTAC → no event
        detector = PhaseDetector(entropy_threshold=0.37, utac_trigger_ceil=0.99)
        s1 = ReflectionState(MirrorPhase.INIT, 0.5, 0.5, 0.3)
        s2 = ReflectionState(MirrorPhase.INIT, 0.5, 0.7, 0.3)  # different output → no collapse
        detector.detect_transition(s1)
        ev = detector.detect_transition(s2)
        assert ev is None

    def test_utac_trigger_fires(self) -> None:
        # Use low threshold and high entropy to force UTAC trigger
        detector = PhaseDetector(entropy_threshold=0.1, utac_trigger_ceil=0.5)
        state = ReflectionState(MirrorPhase.INIT, 0.5, 0.5, 0.9)
        ev = detector.detect_transition(state)
        assert ev is not None
        assert ev.transition_type == TransitionType.UTAC_TRIGGER

    def test_collapse_detected(self) -> None:
        detector = PhaseDetector(
            entropy_threshold=0.99, stability_floor=0.01, utac_trigger_ceil=0.99
        )
        s1 = ReflectionState(MirrorPhase.INIT, 0.5, 0.5, 0.1)
        s2 = ReflectionState(MirrorPhase.REFLECT, 0.5, 0.5001, 0.1)
        detector.detect_transition(s1)
        ev = detector.detect_transition(s2)
        assert ev is not None
        assert ev.transition_type == TransitionType.COLLAPSE

    def test_first_state_no_event(self) -> None:
        detector = PhaseDetector(utac_trigger_ceil=0.99)
        state = ReflectionState(MirrorPhase.INIT, 0.5, 0.5, 0.3)
        ev = detector.detect_transition(state)
        assert ev is None


# ---------------------------------------------------------------------------
# PhaseDetector.process_trace
# ---------------------------------------------------------------------------


class TestProcessTrace:
    def test_process_full_mirror_trace(self) -> None:
        core = MirrorCore(depth=2)
        core.reflect(0.6, entropy=0.4)
        detector = PhaseDetector(entropy_threshold=0.37)
        events = detector.process_trace(core.trace)
        assert isinstance(events, list)
        assert all(isinstance(e, PhaseTransitionEvent) for e in events)

    def test_process_empty_trace(self) -> None:
        detector = PhaseDetector()
        events = detector.process_trace([])
        assert events == []

    def test_history_populated(self) -> None:
        core = MirrorCore(depth=1)
        core.reflect(0.5)
        detector = PhaseDetector()
        detector.process_trace(core.trace)
        assert len(detector.transition_history) == len(
            [e for e in detector.transition_history if e is not None]
        )

    def test_events_have_valid_phases(self) -> None:
        core = MirrorCore(depth=1)
        core.reflect(0.5, entropy=0.3)
        detector = PhaseDetector(utac_trigger_ceil=0.99)
        events = detector.process_trace(core.trace)
        for ev in events:
            assert isinstance(ev.source_phase, MirrorPhase)
            assert isinstance(ev.target_phase, MirrorPhase)


# ---------------------------------------------------------------------------
# PhaseDetector.force_transition
# ---------------------------------------------------------------------------


class TestForceTransition:
    def test_forced_event_type(self) -> None:
        detector = PhaseDetector()
        ev = detector.force_transition(MirrorPhase.INIT, MirrorPhase.EMIT, entropy=0.5)
        assert ev.transition_type == TransitionType.FORCED

    def test_forced_event_in_history(self) -> None:
        detector = PhaseDetector()
        detector.force_transition(MirrorPhase.REFLECT, MirrorPhase.INTEGRATE, entropy=0.4)
        assert len(detector.transition_history) == 1

    def test_forced_event_has_utac_value(self) -> None:
        detector = PhaseDetector()
        ev = detector.force_transition(MirrorPhase.INIT, MirrorPhase.REFLECT, entropy=0.37)
        assert 0.0 < ev.utac_value < 1.0


# ---------------------------------------------------------------------------
# PhaseDetector.reset / state_dict
# ---------------------------------------------------------------------------


class TestResetAndStateDict:
    def test_reset_clears_history(self) -> None:
        detector = PhaseDetector()
        detector.force_transition(MirrorPhase.INIT, MirrorPhase.REFLECT, entropy=0.5)
        assert len(detector.transition_history) == 1
        detector.reset()
        assert len(detector.transition_history) == 0

    def test_reset_clears_last_phase(self) -> None:
        detector = PhaseDetector(utac_trigger_ceil=0.99)
        state = ReflectionState(MirrorPhase.INIT, 0.5, 0.5, 0.3)
        detector.detect_transition(state)
        detector.reset()
        assert detector._last_phase is None
        assert detector._last_output is None

    def test_state_dict_keys(self) -> None:
        detector = PhaseDetector()
        sd = detector.state_dict()
        assert "entropy_threshold" in sd
        assert "stability_floor" in sd
        assert "utac_trigger_ceil" in sd
        assert "history_len" in sd

    def test_state_dict_history_len(self) -> None:
        detector = PhaseDetector()
        detector.force_transition(MirrorPhase.INIT, MirrorPhase.REFLECT, entropy=0.5)
        sd = detector.state_dict()
        assert sd["history_len"] == 1


# ---------------------------------------------------------------------------
# detect_phases_from_core convenience function
# ---------------------------------------------------------------------------


class TestDetectPhasesFromCore:
    def test_returns_list(self) -> None:
        core = MirrorCore(depth=1)
        core.reflect(0.5)
        events = detect_phases_from_core(core.trace)
        assert isinstance(events, list)

    def test_empty_trace_returns_empty(self) -> None:
        events = detect_phases_from_core([])
        assert events == []

    def test_custom_threshold(self) -> None:
        core = MirrorCore(depth=2)
        core.reflect(0.8, entropy=0.4)
        events = detect_phases_from_core(core.trace, entropy_threshold=0.2, utac_trigger_ceil=0.5)
        assert isinstance(events, list)


# ---------------------------------------------------------------------------
# entropy_phase_label / _entropy_to_phase_label
# ---------------------------------------------------------------------------


class TestEntropyPhaseLabel:
    def test_low_entropy_stable(self) -> None:
        label = entropy_phase_label(0.01, threshold=0.37)
        assert label == "STABLE"

    def test_high_entropy_collapse_risk(self) -> None:
        label = entropy_phase_label(0.99, threshold=0.37)
        assert label == "COLLAPSE_RISK"

    def test_mid_entropy_transitioning(self) -> None:
        # entropy at threshold gives UTAC ~0.5 → TRANSITIONING
        label = entropy_phase_label(0.37, threshold=0.37)
        assert label == "TRANSITIONING"

    def test_internal_function_matches(self) -> None:
        assert _entropy_to_phase_label(0.01) == entropy_phase_label(0.01)

    def test_returns_string(self) -> None:
        assert isinstance(entropy_phase_label(0.5), str)


# ---------------------------------------------------------------------------
# Phase ordering and successor maps
# ---------------------------------------------------------------------------


class TestPhaseOrderMaps:
    def test_all_phases_in_order(self) -> None:
        for phase in MirrorPhase:
            assert phase in _PHASE_ORDER

    def test_all_phases_in_successor(self) -> None:
        for phase in MirrorPhase:
            assert phase in _PHASE_SUCCESSOR

    def test_forward_order(self) -> None:
        assert _PHASE_ORDER[MirrorPhase.INIT] < _PHASE_ORDER[MirrorPhase.REFLECT]
        assert _PHASE_ORDER[MirrorPhase.REFLECT] < _PHASE_ORDER[MirrorPhase.INTEGRATE]
        assert _PHASE_ORDER[MirrorPhase.INTEGRATE] < _PHASE_ORDER[MirrorPhase.EMIT]

    def test_emit_successor_cycles_to_init(self) -> None:
        assert _PHASE_SUCCESSOR[MirrorPhase.EMIT] == MirrorPhase.INIT
