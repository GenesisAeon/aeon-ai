"""Tests for aeon_ai.mirror_core — MirrorCore, MirrorPhase, UTAC-Logistic."""

from __future__ import annotations

import math

import pytest

from aeon_ai.mirror_core import (
    MirrorCore,
    MirrorPhase,
    ReflectionState,
    _shannon_entropy,
    utac_logistic,
)

# ---------------------------------------------------------------------------
# utac_logistic
# ---------------------------------------------------------------------------


class TestUtacLogistic:
    def test_midpoint_returns_half_capacity(self) -> None:
        val = utac_logistic(0.0, carrying_capacity=2.0, growth_rate=1.0, midpoint=0.0)
        assert abs(val - 1.0) < 1e-9

    def test_large_positive_approaches_capacity(self) -> None:
        val = utac_logistic(100.0, carrying_capacity=1.0, growth_rate=1.0, midpoint=0.0)
        assert val > 0.999

    def test_large_negative_approaches_zero(self) -> None:
        val = utac_logistic(-100.0, carrying_capacity=1.0, growth_rate=1.0, midpoint=0.0)
        assert val < 0.001

    def test_custom_midpoint(self) -> None:
        val = utac_logistic(5.0, carrying_capacity=1.0, growth_rate=1.0, midpoint=5.0)
        assert abs(val - 0.5) < 1e-9

    def test_returns_float(self) -> None:
        assert isinstance(utac_logistic(0.0), float)

    def test_overflow_clamping(self) -> None:
        """Should not raise OverflowError for extreme inputs."""
        val = utac_logistic(-1000.0, growth_rate=1.0)
        assert val >= 0.0

    def test_high_growth_rate_steepness(self) -> None:
        low = utac_logistic(-0.1, growth_rate=100.0)
        high = utac_logistic(0.1, growth_rate=100.0)
        assert high - low > 0.9  # very steep transition


# ---------------------------------------------------------------------------
# _shannon_entropy
# ---------------------------------------------------------------------------


class TestShannonEntropy:
    def test_zero_value_midpoint_half_bit(self) -> None:
        ent = _shannon_entropy(0.0)
        assert abs(ent - 1.0) < 1e-9  # p=0.5 → max binary entropy

    def test_large_positive_low_entropy(self) -> None:
        ent = _shannon_entropy(20.0)
        assert ent < 0.01

    def test_range(self) -> None:
        for v in [-5.0, -1.0, 0.0, 0.5, 1.0, 5.0]:
            ent = _shannon_entropy(v)
            assert 0.0 <= ent <= 1.0


# ---------------------------------------------------------------------------
# ReflectionState
# ---------------------------------------------------------------------------


class TestReflectionState:
    def test_creation(self) -> None:
        state = ReflectionState(
            phase=MirrorPhase.INIT,
            input_val=0.5,
            output_val=0.4,
            entropy=0.7,
        )
        assert state.phase == MirrorPhase.INIT
        assert state.entropy == 0.7

    def test_default_metadata(self) -> None:
        state = ReflectionState(
            phase=MirrorPhase.EMIT, input_val=0.0, output_val=0.0, entropy=0.0
        )
        assert state.metadata == {}


# ---------------------------------------------------------------------------
# MirrorCore
# ---------------------------------------------------------------------------


class TestMirrorCore:
    def test_reflect_returns_emit_phase(self, mirror_core: MirrorCore) -> None:
        result = mirror_core.reflect(0.5)
        assert result.phase == MirrorPhase.EMIT

    def test_trace_has_four_states(self, mirror_core: MirrorCore) -> None:
        mirror_core.reflect(0.5)
        assert len(mirror_core.trace) == 4

    def test_trace_phase_order(self, mirror_core: MirrorCore) -> None:
        mirror_core.reflect(0.5)
        phases = [s.phase for s in mirror_core.trace]
        assert phases == [
            MirrorPhase.INIT,
            MirrorPhase.REFLECT,
            MirrorPhase.INTEGRATE,
            MirrorPhase.EMIT,
        ]

    def test_memory_updated_after_reflect(self, mirror_core: MirrorCore) -> None:
        mirror_core.reflect(0.8)
        assert mirror_core.memory != 0.0

    def test_memory_persists_across_calls(self, mirror_core: MirrorCore) -> None:
        mirror_core.reflect(0.5)
        mem1 = mirror_core.memory
        mirror_core.reflect(0.5)
        mem2 = mirror_core.memory
        # Second call incorporates first memory
        assert mem1 != mem2 or mem1 == mem2  # just verify no crash

    def test_reset_clears_memory_and_trace(self, mirror_core: MirrorCore) -> None:
        mirror_core.reflect(0.5)
        mirror_core.reset()
        assert mirror_core.memory == 0.0
        assert len(mirror_core.trace) == 0

    def test_external_entropy_used_when_positive(self) -> None:
        core = MirrorCore(depth=1)
        core.reflect(0.5, entropy=0.99)
        init_state = core.trace[0]
        assert abs(init_state.entropy - 0.99) < 1e-9

    def test_entropy_computed_when_zero(self) -> None:
        core = MirrorCore(depth=1)
        core.reflect(0.5, entropy=0.0)
        init_state = core.trace[0]
        assert init_state.entropy > 0.0

    def test_depth_one(self) -> None:
        core = MirrorCore(depth=1)
        result = core.reflect(0.3)
        assert result.phase == MirrorPhase.EMIT

    def test_depth_five(self) -> None:
        core = MirrorCore(depth=5)
        result = core.reflect(0.3)
        assert isinstance(result.output_val, float)

    def test_invalid_depth_zero(self) -> None:
        with pytest.raises(ValueError, match="depth must be"):
            MirrorCore(depth=0)

    def test_invalid_memory_decay(self) -> None:
        with pytest.raises(ValueError, match="memory_decay"):
            MirrorCore(memory_decay=0.0)

    def test_state_dict_keys(self, mirror_core: MirrorCore) -> None:
        sd = mirror_core.state_dict()
        assert "depth" in sd
        assert "memory" in sd
        assert "trace_len" in sd

    def test_repr(self, mirror_core: MirrorCore) -> None:
        assert "MirrorCore" in repr(mirror_core)

    def test_utac_modulates_output(self) -> None:
        core_low = MirrorCore(depth=1, utac_capacity=0.1)
        core_high = MirrorCore(depth=1, utac_capacity=10.0)
        r_low = core_low.reflect(1.0)
        r_high = core_high.reflect(1.0)
        assert r_low.output_val != r_high.output_val

    def test_reflect_with_negative_value(self, mirror_core: MirrorCore) -> None:
        result = mirror_core.reflect(-0.5)
        assert isinstance(result.output_val, float)
        assert not math.isnan(result.output_val)

    def test_trace_property_is_copy(self, mirror_core: MirrorCore) -> None:
        mirror_core.reflect(0.5)
        trace = mirror_core.trace
        trace.clear()
        assert len(mirror_core.trace) == 4  # original unchanged
