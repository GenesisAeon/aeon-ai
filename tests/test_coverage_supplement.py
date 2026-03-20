"""Supplementary tests targeting specific uncovered lines for 99%+ coverage."""

from __future__ import annotations

import sys
from typing import Any
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from aeon_ai.aeon_layer import AeonLayer, WeightingLayerProtocol
from aeon_ai.cli import _try_visualize, app, main

runner = CliRunner()


# ---------------------------------------------------------------------------
# WeightingLayerProtocol stubs (Protocol ...  methods)
# ---------------------------------------------------------------------------


class TestWeightingLayerProtocol:
    """Ensure the Protocol is usable as a runtime-checkable interface."""

    def test_concrete_satisfies_protocol(self) -> None:
        class ConcreteLayer:
            def forward(self, s_a: float, s_v: float) -> float:
                return s_a + s_v

            def state_dict(self) -> dict[str, Any]:
                return {}

        assert isinstance(ConcreteLayer(), WeightingLayerProtocol)

    def test_incomplete_class_fails_check(self) -> None:
        class Incomplete:
            def forward(self, s_a: float, s_v: float) -> float:
                return s_a

        # Missing state_dict → not a full implementation
        assert not isinstance(Incomplete(), WeightingLayerProtocol)


# ---------------------------------------------------------------------------
# AeonLayer — AWS happy path (mocked)
# ---------------------------------------------------------------------------


class TestAeonLayerAWSMock:
    def test_from_advanced_weighting_systems_uses_base(self) -> None:
        """Mock AWS package to exercise the successful import branch."""
        mock_base = MagicMock()
        mock_base.forward.return_value = 0.5

        mock_aws_module = MagicMock()
        mock_aws_layer_cls = MagicMock(return_value=mock_base)
        mock_aws_module.AeonLayer = mock_aws_layer_cls

        with patch.dict(sys.modules, {"advanced_weighting_systems": mock_aws_module}):
            layer = AeonLayer.from_advanced_weighting_systems(delta=0.1)

        assert isinstance(layer, AeonLayer)
        assert layer._base is mock_base


# ---------------------------------------------------------------------------
# CLI — ValueError branch in reflect
# ---------------------------------------------------------------------------


class TestCliValueErrorBranch:
    def test_reflect_pipeline_error_on_bad_time_step(self) -> None:
        """Trigger the ValueError handler via zero time-step."""
        result = runner.invoke(app, ["reflect", "--time-step", "0.0"])
        assert result.exit_code != 0

    def test_reflect_pipeline_error_mocked_orchestrator(self) -> None:
        """Trigger lines 170-172: ValueError inside the orchestrator run."""
        mock_orch = MagicMock()
        mock_orch.run.side_effect = ValueError("forced error")

        with patch("aeon_ai.cli._build_orchestrator", return_value=mock_orch):
            result = runner.invoke(app, ["reflect", "--entropy", "0.3"])

        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# CLI — visualize with mocked packages
# ---------------------------------------------------------------------------


class TestCliVisualizeMocked:
    def test_try_visualize_mandala_called(self) -> None:
        mock_mv = MagicMock()
        mock_cw = MagicMock()
        mock_result = MagicMock()
        mock_result.as_dict.return_value = {}
        mock_result.cosmic_moment.as_dict.return_value = {}

        with (
            patch.dict(sys.modules, {"mandala_visualizer": mock_mv, "cosmic_web": mock_cw}),
        ):
            _try_visualize(mock_result)

        mock_mv.render.assert_called_once()
        mock_cw.sonify.assert_called_once()

    def test_try_visualize_cosmic_web_only(self) -> None:
        mock_cw = MagicMock()
        mock_result = MagicMock()
        mock_result.as_dict.return_value = {}
        mock_result.cosmic_moment.as_dict.return_value = {}

        # mandala_visualizer raises ImportError, cosmic_web works
        with patch.dict(sys.modules, {"cosmic_web": mock_cw}):
            # Remove mandala_visualizer from sys.modules if present
            sys.modules.pop("mandala_visualizer", None)
            _try_visualize(mock_result)

        mock_cw.sonify.assert_called_once()


# ---------------------------------------------------------------------------
# CLI — info command with a module that fails to import
# ---------------------------------------------------------------------------


class TestCliInfoImportError:
    def test_info_handles_broken_module(self) -> None:
        """Cause one component import to fail to exercise the except branch."""
        original = sys.modules.get("aeon_ai.crep_evaluator")
        sys.modules["aeon_ai.crep_evaluator"] = None  # type: ignore[assignment]
        try:
            result = runner.invoke(app, ["info"])
            assert result.exit_code == 0
        finally:
            if original is not None:
                sys.modules["aeon_ai.crep_evaluator"] = original
            else:
                sys.modules.pop("aeon_ai.crep_evaluator", None)


# ---------------------------------------------------------------------------
# CLI — main() entry point
# ---------------------------------------------------------------------------


class TestCliMain:
    def test_main_runs_help(self) -> None:
        """Call main() to cover the entry-point lines."""
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0

    def test_main_function_directly(self) -> None:
        """Exercise the main() function and __main__ guard (lines 302, 306)."""
        import contextlib

        with patch.object(sys, "argv", ["aeon", "--help"]), contextlib.suppress(SystemExit):
            main()

    def test_info_with_stack_package_present(self) -> None:
        """Exercise line 266: avail = green tick when a stack pkg is imported."""
        mock_pkg = MagicMock()
        pkg_name = "advanced_weighting_systems"
        with patch.dict(sys.modules, {pkg_name: mock_pkg}):
            result = runner.invoke(app, ["info"])
        assert result.exit_code == 0


# ---------------------------------------------------------------------------
# FieldBridge — _try_fieldtheory mocked
# ---------------------------------------------------------------------------


class TestFieldBridgeMocked:
    def test_try_fieldtheory_success(self) -> None:
        from aeon_ai.field_bridge import FieldBridge, MediumMode

        mock_ft = MagicMock()
        mock_ft.cosmic_moment.return_value = {
            "timestamp": 42.0,
            "entropy": 0.3,
            "tension": 0.1,
            "coherence": 0.9,
            "resonance": 0.8,
            "medium": MediumMode.RESONANT.value,
            "metadata": {"source": "mock"},
        }

        with patch.dict(sys.modules, {"fieldtheory": mock_ft}):
            bridge = FieldBridge(base_entropy=0.3)
            moment = bridge.sample_moment(t=42.0)

        assert moment.timestamp == 42.0
        assert moment.entropy == 0.3

    def test_try_fieldtheory_exception_fallback(self) -> None:
        """If fieldtheory raises a non-ImportError, fall back to oscillator."""
        from aeon_ai.field_bridge import FieldBridge

        mock_ft = MagicMock()
        mock_ft.cosmic_moment.side_effect = RuntimeError("bad")

        with patch.dict(sys.modules, {"fieldtheory": mock_ft}):
            bridge = FieldBridge(base_entropy=0.3)
            moment = bridge.sample_moment(t=1.0)

        assert moment is not None


# ---------------------------------------------------------------------------
# SigillinBridge — _try_load_external mocked
# ---------------------------------------------------------------------------


class TestSigillinBridgeMocked:
    def test_try_load_external_success(self) -> None:
        from aeon_ai.sigillin_bridge import SigillinBridge

        mock_sigillin = MagicMock()
        mock_sigillin.registry.return_value = [
            {
                "id": "EXTERNAL_TEST",
                "name": "External",
                "description": "Loaded from sigillin",
                "triggers": [r"\bexternal\b"],
                "weight": 0.7,
                "phase": "R",
            }
        ]

        with patch.dict(sys.modules, {"sigillin": mock_sigillin}):
            bridge = SigillinBridge(load_defaults=False)

        assert "EXTERNAL_TEST" in bridge.sigils

    def test_try_load_external_general_exception(self) -> None:
        """If sigillin.registry() raises, bridge proceeds with defaults."""
        from aeon_ai.sigillin_bridge import SigillinBridge

        mock_sigillin = MagicMock()
        mock_sigillin.registry.side_effect = RuntimeError("bad registry")

        with patch.dict(sys.modules, {"sigillin": mock_sigillin}):
            bridge = SigillinBridge(load_defaults=True)

        assert "GENESIS" in bridge.sigils  # defaults still loaded


# ---------------------------------------------------------------------------
# Orchestrator — _try_load_external mocked
# ---------------------------------------------------------------------------


class TestOrchestratorExternalMock:
    def test_external_integration_called(self) -> None:
        from aeon_ai.agents.orchestrator import Orchestrator

        mock_umn = MagicMock()

        modules = {
            "unified_mandala": mock_umn,
            "unified_mandala.neural": mock_umn,
            "unified_mandala.neural.orchestrator": mock_umn,
        }
        with patch.dict(sys.modules, modules):
            orch = Orchestrator()

        assert orch is not None
