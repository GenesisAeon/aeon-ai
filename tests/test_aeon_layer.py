"""Tests for aeon_ai.aeon_layer — AeonLayer + Lagrangian."""

from __future__ import annotations

import pytest

from aeon_ai.aeon_layer import AeonLayer, LagrangianConfig, lagrangian, lagrangian_gradient

# ---------------------------------------------------------------------------
# lagrangian function
# ---------------------------------------------------------------------------


class TestLagrangianFunction:
    def test_basic_computation(self) -> None:
        """L = S_A*S_V/(S_A+S_V) - (1+δ)/t²."""
        val = lagrangian(s_a=1.0, s_v=1.0, delta=0.0, t=1.0)
        expected = 0.5 - 1.0  # 1*1/(1+1) - (1+0)/1²
        assert abs(val - expected) < 1e-9

    def test_with_nonzero_delta(self) -> None:
        val = lagrangian(s_a=2.0, s_v=2.0, delta=1.0, t=2.0)
        # harmonic = 4/4 = 1; temporal = 2/4 = 0.5
        assert abs(val - 0.5) < 1e-9

    def test_zero_denominator_guard(self) -> None:
        """When s_a + s_v ≈ 0 the harmonic term is 0."""
        val = lagrangian(s_a=0.0, s_v=0.0, delta=0.0, t=1.0, epsilon=1e-8)
        assert val < 0  # 0 - (1+0)/1 = -1

    def test_negative_signals_allowed(self) -> None:
        """Negative signals should not raise."""
        val = lagrangian(s_a=-0.5, s_v=0.5, delta=0.0, t=1.0)
        assert isinstance(val, float)

    def test_invalid_t_zero(self) -> None:
        with pytest.raises(ValueError, match="strictly positive"):
            lagrangian(1.0, 1.0, 0.0, t=0.0)

    def test_invalid_t_negative(self) -> None:
        with pytest.raises(ValueError, match="strictly positive"):
            lagrangian(1.0, 1.0, 0.0, t=-1.0)

    def test_large_t_penalty_approaches_zero(self) -> None:
        """For large t the temporal penalty → 0."""
        val = lagrangian(s_a=1.0, s_v=1.0, delta=0.0, t=1e6)
        assert abs(val - 0.5) < 1e-5  # approaches harmonic term


# ---------------------------------------------------------------------------
# lagrangian_gradient
# ---------------------------------------------------------------------------


class TestLagrangianGradient:
    def test_keys_present(self) -> None:
        grad = lagrangian_gradient(1.0, 1.0, 0.0, 1.0)
        assert set(grad.keys()) == {"dL/ds_a", "dL/ds_v", "dL/dt"}

    def test_symmetric_signals(self) -> None:
        """Symmetric signals yield equal dL/ds_a and dL/ds_v."""
        grad = lagrangian_gradient(1.0, 1.0, 0.0, 1.0)
        assert abs(grad["dL/ds_a"] - grad["dL/ds_v"]) < 1e-9

    def test_dt_positive_for_positive_delta(self) -> None:
        """dL/dt = 2(1+δ)/t³ > 0 for δ ≥ 0."""
        grad = lagrangian_gradient(1.0, 1.0, 0.5, 1.0)
        assert grad["dL/dt"] > 0

    def test_zero_denominator_zero_signal_grads(self) -> None:
        """Near-zero denominator yields zero signal gradients."""
        grad = lagrangian_gradient(0.0, 0.0, 0.0, 1.0, epsilon=1e-8)
        assert abs(grad["dL/ds_a"]) < 1e-9
        assert abs(grad["dL/ds_v"]) < 1e-9


# ---------------------------------------------------------------------------
# LagrangianConfig
# ---------------------------------------------------------------------------


class TestLagrangianConfig:
    def test_defaults(self) -> None:
        cfg = LagrangianConfig()
        assert cfg.delta == 0.0
        assert cfg.epsilon == 1e-8

    def test_custom(self) -> None:
        cfg = LagrangianConfig(delta=0.5, epsilon=1e-6)
        assert cfg.delta == 0.5


# ---------------------------------------------------------------------------
# AeonLayer
# ---------------------------------------------------------------------------


class TestAeonLayer:
    def test_forward_returns_float(self, aeon_layer: AeonLayer) -> None:
        result = aeon_layer.forward(0.7, 0.6, t=1.0)
        assert isinstance(result, float)

    def test_forward_matches_lagrangian(self) -> None:
        layer = AeonLayer(delta=0.2)
        expected = lagrangian(0.8, 0.4, delta=0.2, t=2.0)
        assert abs(layer.forward(0.8, 0.4, t=2.0) - expected) < 1e-9

    def test_history_records_entries(self, aeon_layer: AeonLayer) -> None:
        aeon_layer.forward(0.5, 0.5)
        aeon_layer.forward(0.3, 0.7)
        assert len(aeon_layer._history) == 2

    def test_reset_history(self, aeon_layer: AeonLayer) -> None:
        aeon_layer.forward(0.5, 0.5)
        aeon_layer.reset_history()
        assert len(aeon_layer._history) == 0

    def test_state_dict_keys(self, aeon_layer: AeonLayer) -> None:
        sd = aeon_layer.state_dict()
        assert "delta" in sd
        assert "epsilon" in sd
        assert "history_len" in sd

    def test_load_state_dict(self, aeon_layer: AeonLayer) -> None:
        aeon_layer.load_state_dict({"delta": 99.9, "epsilon": 1e-6})
        assert aeon_layer.config.delta == 99.9
        assert aeon_layer.config.epsilon == 1e-6

    def test_load_state_dict_partial(self, aeon_layer: AeonLayer) -> None:
        original_epsilon = aeon_layer.config.epsilon
        aeon_layer.load_state_dict({"delta": 5.0})
        assert aeon_layer.config.delta == 5.0
        assert aeon_layer.config.epsilon == original_epsilon

    def test_gradient_returns_expected_keys(self, aeon_layer: AeonLayer) -> None:
        grad = aeon_layer.gradient(0.5, 0.5, t=1.0)
        assert "dL/ds_a" in grad
        assert "dL/ds_v" in grad
        assert "dL/dt" in grad

    def test_invalid_t_raises(self, aeon_layer: AeonLayer) -> None:
        with pytest.raises(ValueError, match="strictly positive"):
            aeon_layer.forward(0.5, 0.5, t=0.0)

    def test_repr(self, aeon_layer: AeonLayer) -> None:
        assert "AeonLayer" in repr(aeon_layer)

    def test_from_advanced_weighting_systems_fallback(self) -> None:
        """Should return native AeonLayer when package absent."""
        layer = AeonLayer.from_advanced_weighting_systems(delta=0.3)
        assert isinstance(layer, AeonLayer)
        assert layer.config.delta == 0.3

    def test_with_base_layer(self) -> None:
        """Base layer should modulate s_a in forward pass."""

        class DoubleLayer:
            def forward(self, s_a: float, s_v: float) -> float:
                return s_a * 2.0

            def state_dict(self) -> dict:
                return {}

        layer = AeonLayer(delta=0.0, _base_layer=DoubleLayer())
        # s_a will be doubled → different result
        result_with_base = layer.forward(0.5, 0.4, t=1.0)
        result_without = lagrangian(0.5, 0.4, delta=0.0, t=1.0)
        assert result_with_base != result_without

    def test_zero_delta_default(self) -> None:
        layer = AeonLayer()
        assert layer.config.delta == 0.0
