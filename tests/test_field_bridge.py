"""Tests for aeon_ai.field_bridge — CosmicMoment, MediumMode, FieldBridge."""

from __future__ import annotations

import pytest

from aeon_ai.field_bridge import CosmicMoment, FieldBridge, MediumMode, _medium_from_entropy

# ---------------------------------------------------------------------------
# MediumMode
# ---------------------------------------------------------------------------


class TestMediumMode:
    def test_all_modes_exist(self) -> None:
        modes = [m.value for m in MediumMode]
        assert "vacuum" in modes
        assert "aetheric" in modes
        assert "resonant" in modes
        assert "dense" in modes


# ---------------------------------------------------------------------------
# _medium_from_entropy
# ---------------------------------------------------------------------------


class TestMediumFromEntropy:
    def test_very_low_entropy_vacuum(self) -> None:
        assert _medium_from_entropy(0.1) == MediumMode.VACUUM

    def test_low_entropy_aetheric(self) -> None:
        assert _medium_from_entropy(0.3) == MediumMode.AETHERIC

    def test_mid_entropy_resonant(self) -> None:
        assert _medium_from_entropy(0.55) == MediumMode.RESONANT

    def test_high_entropy_dense(self) -> None:
        assert _medium_from_entropy(0.9) == MediumMode.DENSE


# ---------------------------------------------------------------------------
# CosmicMoment
# ---------------------------------------------------------------------------


class TestCosmicMoment:
    def _make(self, **kwargs) -> CosmicMoment:
        defaults = {
            "timestamp": 1.0,
            "entropy": 0.3,
            "tension": 0.2,
            "coherence": 0.8,
            "resonance": 0.7,
            "medium": MediumMode.AETHERIC,
            "metadata": {},
        }
        defaults.update(kwargs)
        return CosmicMoment(**defaults)

    def test_field_strength_positive(self) -> None:
        moment = self._make()
        assert moment.field_strength > 0.0

    def test_field_strength_formula(self) -> None:
        moment = self._make(coherence=0.8, resonance=0.7, tension=0.2)
        expected = (0.8 * 0.7) / (1.0 + 0.2)
        assert abs(moment.field_strength - expected) < 1e-9

    def test_zero_tension_max_strength(self) -> None:
        full = self._make(coherence=1.0, resonance=1.0, tension=0.0)
        assert abs(full.field_strength - 1.0) < 1e-9

    def test_as_dict_keys(self) -> None:
        moment = self._make()
        d = moment.as_dict()
        assert "timestamp" in d
        assert "field_strength" in d
        assert "medium" in d

    def test_as_dict_medium_is_string(self) -> None:
        moment = self._make()
        assert isinstance(moment.as_dict()["medium"], str)


# ---------------------------------------------------------------------------
# FieldBridge
# ---------------------------------------------------------------------------


class TestFieldBridge:
    def test_sample_moment_returns_cosmic_moment(self, field_bridge: FieldBridge) -> None:
        moment = field_bridge.sample_moment()
        assert isinstance(moment, CosmicMoment)

    def test_sample_moment_records_history(self, field_bridge: FieldBridge) -> None:
        field_bridge.sample_moment()
        field_bridge.sample_moment()
        assert len(field_bridge.moment_history) == 2

    def test_reset_clears_history(self, field_bridge: FieldBridge) -> None:
        field_bridge.sample_moment()
        field_bridge.reset()
        assert len(field_bridge.moment_history) == 0

    def test_modulation_factor_positive(self, field_bridge: FieldBridge) -> None:
        moment = field_bridge.sample_moment()
        factor = field_bridge.modulation_factor(moment)
        assert factor > 0.0

    def test_modulation_factor_bounded(self, field_bridge: FieldBridge) -> None:
        moment = field_bridge.sample_moment()
        factor = field_bridge.modulation_factor(moment)
        assert factor <= 2.0

    def test_adjust_delta_increases_with_tension(self) -> None:
        bridge = FieldBridge(base_entropy=0.5)
        from aeon_ai.field_bridge import CosmicMoment, MediumMode

        moment = CosmicMoment(
            timestamp=1.0, entropy=0.8, tension=1.0,
            coherence=0.5, resonance=0.5,
            medium=MediumMode.DENSE, metadata={},
        )
        adjusted = bridge.adjust_delta(0.0, moment)
        assert adjusted > 0.0

    def test_inject_entropy_equals_moment_entropy(self, field_bridge: FieldBridge) -> None:
        moment = field_bridge.sample_moment()
        assert field_bridge.inject_entropy(moment) == moment.entropy

    def test_invalid_base_entropy(self) -> None:
        with pytest.raises(ValueError, match="base_entropy"):
            FieldBridge(base_entropy=1.5)

    def test_invalid_coherence_base(self) -> None:
        with pytest.raises(ValueError, match="coherence_base"):
            FieldBridge(coherence_base=-0.1)

    def test_invalid_resonance_base(self) -> None:
        with pytest.raises(ValueError, match="resonance_base"):
            FieldBridge(resonance_base=2.0)

    def test_repr(self, field_bridge: FieldBridge) -> None:
        assert "FieldBridge" in repr(field_bridge)

    def test_moment_history_is_copy(self, field_bridge: FieldBridge) -> None:
        field_bridge.sample_moment()
        history = field_bridge.moment_history
        history.clear()
        assert len(field_bridge.moment_history) == 1

    def test_custom_time_coordinate(self, field_bridge: FieldBridge) -> None:
        moment = field_bridge.sample_moment(t=42.0)
        assert moment.timestamp == 42.0

    def test_oscillator_entropy_within_range(self, field_bridge: FieldBridge) -> None:
        for i in range(5):
            moment = field_bridge.sample_moment(t=float(i * 100))
            assert 0.0 <= moment.entropy <= 1.0
