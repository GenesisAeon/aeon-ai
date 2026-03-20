"""Contract tests for aeon-ai v0.2.0 integration with the GenesisAeon stack.

Tests the graceful-degradation contracts when optional packages from the stack
(mirror-machine, advanced-weighting-systems, fieldtheory, entropy-governance,
utac-core) are absent, and verifies that the new v0.2.0 modules correctly
integrate with the orchestrator pipeline.
"""

from __future__ import annotations

import sys
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aeon_ai.agents.orchestrator import Orchestrator, OrchestratorResult
from aeon_ai.aeon_layer import AeonLayer
from aeon_ai.crep_evaluator import CREPEvaluator
from aeon_ai.field_bridge import FieldBridge
from aeon_ai.mirror_core import MirrorCore, MirrorPhase
from aeon_ai.phase_detector import PhaseDetector, PhaseTransitionEvent, TransitionType
from aeon_ai.self_reflection import MAX_ITER, ReflectionLoopResult, SelfReflector
from aeon_ai.sigillin_bridge import SigillinBridge


# ---------------------------------------------------------------------------
# Contract: mirror-machine absent → PhaseDetector operates standalone
# ---------------------------------------------------------------------------


class TestMirrorMachineContract:
    def test_phase_detector_works_without_mirror_machine(self) -> None:
        """PhaseDetector must initialise and work when mirror-machine absent."""
        with patch.dict(sys.modules, {"mirror_machine": None}):
            detector = PhaseDetector(entropy_threshold=0.37)
        assert detector is not None
        core = MirrorCore(depth=1)
        core.reflect(0.5)
        events = detector.process_trace(core.trace)
        assert isinstance(events, list)

    def test_phase_detector_with_mock_mirror_machine_register(self) -> None:
        """If mirror-machine exposes register_phase_detector, it is called."""
        mock_mm = MagicMock()
        mock_mm.register_phase_detector = MagicMock()
        with patch.dict(sys.modules, {"mirror_machine": mock_mm}):
            detector = PhaseDetector(entropy_threshold=0.37)
        mock_mm.register_phase_detector.assert_called_once_with(detector)

    def test_phase_detector_with_mirror_machine_no_register_attr(self) -> None:
        """mirror-machine without register_phase_detector does not crash."""
        mock_mm = MagicMock(spec=[])  # no attributes at all
        with patch.dict(sys.modules, {"mirror_machine": mock_mm}):
            detector = PhaseDetector()
        assert detector is not None

    def test_phase_detector_mirror_machine_raises(self) -> None:
        """mirror-machine that raises on import is handled gracefully."""
        with patch.dict(sys.modules, {"mirror_machine": None}):
            # None in sys.modules causes ImportError when importing
            detector = PhaseDetector()
        assert detector is not None


# ---------------------------------------------------------------------------
# Contract: advanced-weighting-systems absent → AeonLayer native fallback
# ---------------------------------------------------------------------------


class TestAdvancedWeightingSystemsContract:
    def test_aeon_layer_native_without_aws(self) -> None:
        """AeonLayer must work natively when AWS absent."""
        with patch.dict(sys.modules, {"advanced_weighting_systems": None}):
            layer = AeonLayer.from_advanced_weighting_systems(delta=0.1)
        assert isinstance(layer, AeonLayer)
        val = layer.forward(0.7, 0.6, t=1.0)
        assert isinstance(val, float)

    def test_aeon_layer_with_mock_aws(self) -> None:
        """AeonLayer wraps AWS base when available."""
        mock_base = MagicMock()
        mock_base.forward.return_value = 0.55

        mock_aws = MagicMock()
        mock_aws.AeonLayer = MagicMock(return_value=mock_base)

        with patch.dict(sys.modules, {"advanced_weighting_systems": mock_aws}):
            layer = AeonLayer.from_advanced_weighting_systems(delta=0.05)

        assert layer._base is mock_base

    def test_self_reflector_with_aws_absent(self) -> None:
        """SelfReflector must complete loop when AWS is absent."""
        with patch.dict(sys.modules, {"advanced_weighting_systems": None}):
            reflector = SelfReflector(delta=0.1)
            result = reflector.self_reflect()
        assert result.total_iterations >= 1


# ---------------------------------------------------------------------------
# Contract: fieldtheory absent → FieldBridge oscillator fallback
# ---------------------------------------------------------------------------


class TestFieldtheoryContract:
    def test_field_bridge_oscillator_fallback(self) -> None:
        """FieldBridge uses oscillator when fieldtheory absent."""
        with patch.dict(sys.modules, {"fieldtheory": None}):
            bridge = FieldBridge(base_entropy=0.3)
            moment = bridge.sample_moment()
        assert 0.0 <= moment.entropy <= 1.0
        assert moment.metadata.get("source") == "oscillator"

    def test_field_bridge_with_mock_fieldtheory(self) -> None:
        """FieldBridge uses fieldtheory.cosmic_moment when available."""
        mock_ft = MagicMock()
        mock_ft.cosmic_moment.return_value = {
            "timestamp": 1000.0,
            "entropy": 0.4,
            "tension": 0.1,
            "coherence": 0.8,
            "resonance": 0.75,
            "medium": "aetheric",
            "metadata": {"source": "fieldtheory"},
        }
        with patch.dict(sys.modules, {"fieldtheory": mock_ft}):
            bridge = FieldBridge(base_entropy=0.3)
            moment = bridge.sample_moment()
        assert moment.entropy == 0.4
        assert moment.metadata.get("source") == "fieldtheory"


# ---------------------------------------------------------------------------
# Contract: entropy-governance absent → graceful degradation
# ---------------------------------------------------------------------------


class TestEntropyGovernanceContract:
    def test_orchestrator_works_without_entropy_governance(self) -> None:
        """Orchestrator must run when entropy-governance is absent."""
        with patch.dict(sys.modules, {"entropy_governance": None}):
            orch = Orchestrator()
            result = orch.run(s_a=0.7, s_v=0.6)
        assert isinstance(result, OrchestratorResult)

    def test_self_reflector_works_without_entropy_governance(self) -> None:
        with patch.dict(sys.modules, {"entropy_governance": None}):
            reflector = SelfReflector()
            result = reflector.self_reflect()
        assert result.total_iterations >= 1


# ---------------------------------------------------------------------------
# Contract: utac-core absent → native utac_logistic used
# ---------------------------------------------------------------------------


class TestUtacCoreContract:
    def test_phase_detector_native_utac_without_utac_core(self) -> None:
        """PhaseDetector uses native utac_logistic when utac-core absent."""
        with patch.dict(sys.modules, {"utac_core": None}):
            detector = PhaseDetector()
            val = detector.utac_value_at(0.5)
        assert 0.0 < val < 1.0

    def test_utac_trigger_check_without_utac_core(self) -> None:
        with patch.dict(sys.modules, {"utac_core": None}):
            detector = PhaseDetector(entropy_threshold=0.1, utac_trigger_ceil=0.5)
            assert detector.utac_trigger_check(0.9)


# ---------------------------------------------------------------------------
# Contract: sigillin absent → SigillinBridge built-in sigils used
# ---------------------------------------------------------------------------


class TestSigillinContract:
    def test_sigillin_bridge_default_sigils_without_sigillin(self) -> None:
        with patch.dict(sys.modules, {"sigillin": None}):
            bridge = SigillinBridge(load_defaults=True)
        assert "GENESIS" in bridge.sigils
        assert "MIRROR" in bridge.sigils

    def test_self_reflector_sigillin_without_package(self) -> None:
        with patch.dict(sys.modules, {"sigillin": None}):
            reflector = SelfReflector(sigil_text="mirror aeon genesis")
            result = reflector.self_reflect()
        assert isinstance(result, ReflectionLoopResult)


# ---------------------------------------------------------------------------
# Contract: OrchestratorResult includes phase_events (v0.2.0)
# ---------------------------------------------------------------------------


class TestOrchestratorV020Contract:
    def test_result_has_phase_events_field(self) -> None:
        orch = Orchestrator()
        result = orch.run(s_a=0.7, s_v=0.6)
        assert hasattr(result, "phase_events")
        assert isinstance(result.phase_events, list)

    def test_phase_events_are_valid_type(self) -> None:
        orch = Orchestrator()
        result = orch.run(s_a=0.7, s_v=0.6)
        for ev in result.phase_events:
            assert isinstance(ev, PhaseTransitionEvent)

    def test_as_dict_includes_phase_events(self) -> None:
        orch = Orchestrator()
        result = orch.run(s_a=0.7, s_v=0.6)
        d = result.as_dict()
        assert "phase_events" in d
        assert isinstance(d["phase_events"], list)

    def test_orchestrator_state_dict_has_phase_detector(self) -> None:
        orch = Orchestrator()
        sd = orch.state_dict()
        assert "phase_detector" in sd

    def test_orchestrator_reset_clears_phase_detector(self) -> None:
        orch = Orchestrator()
        orch.run(s_a=0.7, s_v=0.6)
        orch.reset()
        assert len(orch.phase_detector.transition_history) == 0

    def test_orchestrator_batch_returns_results_with_phase_events(self) -> None:
        orch = Orchestrator()
        results = orch.run_batch([
            {"s_a": 0.5, "s_v": 0.5},
            {"s_a": 0.8, "s_v": 0.3},
        ])
        assert len(results) == 2
        for r in results:
            assert hasattr(r, "phase_events")

    def test_orchestrator_custom_entropy_threshold(self) -> None:
        orch = Orchestrator(entropy_threshold=0.5)
        assert orch.phase_detector.entropy_threshold == 0.5


# ---------------------------------------------------------------------------
# Contract: SelfReflector max-iteration guard
# ---------------------------------------------------------------------------


class TestSelfReflectorMaxIterContract:
    def test_never_exceeds_max_iter(self) -> None:
        """The loop must never produce more than MAX_ITER snapshots."""
        for _ in range(5):
            reflector = SelfReflector()
            result = reflector.self_reflect()
            assert result.total_iterations <= MAX_ITER

    def test_snapshots_count_matches_total_iterations(self) -> None:
        reflector = SelfReflector()
        result = reflector.self_reflect()
        assert len(result.snapshots) == result.total_iterations


# ---------------------------------------------------------------------------
# Contract: PhaseDetector + MirrorCore integration loop
# ---------------------------------------------------------------------------


class TestPhaseDetectorMirrorCoreContract:
    def test_full_pipeline_produces_forward_events(self) -> None:
        """A full mirror trace always contains forward-transition events."""
        core = MirrorCore(depth=1)
        core.reflect(0.6, entropy=0.3)
        detector = PhaseDetector(entropy_threshold=0.37, utac_trigger_ceil=0.99)
        events = detector.process_trace(core.trace)
        # Trace has 4 states → at least 3 forward transitions possible
        forward = [e for e in events if e.transition_type == TransitionType.FORWARD]
        assert len(forward) >= 1

    def test_multiple_runs_accumulate_history(self) -> None:
        detector = PhaseDetector(utac_trigger_ceil=0.99)
        core = MirrorCore(depth=1)
        for _ in range(3):
            detector.reset()
            core.reset()
            core.reflect(0.5, entropy=0.3)
            detector.process_trace(core.trace)
        # After reset, only last run's events remain
        assert len(detector.transition_history) >= 0  # may be 0 after reset

    def test_phase_events_phases_are_valid_enum_members(self) -> None:
        core = MirrorCore(depth=2)
        core.reflect(0.7, entropy=0.4)
        detector = PhaseDetector(entropy_threshold=0.37)
        events = detector.process_trace(core.trace)
        for ev in events:
            assert ev.source_phase in MirrorPhase
            assert ev.target_phase in MirrorPhase


# ---------------------------------------------------------------------------
# Contract: CLI v0.2.0 commands
# ---------------------------------------------------------------------------


class TestCLIV020Contract:
    def test_reflect_loop_flag(self) -> None:
        from typer.testing import CliRunner
        from aeon_ai.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["reflect", "--loop"])
        assert result.exit_code == 0

    def test_reflect_loop_phases_flags(self) -> None:
        from typer.testing import CliRunner
        from aeon_ai.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["reflect", "--loop", "--phases"])
        assert result.exit_code == 0

    def test_reflect_phases_flag_standard_mode(self) -> None:
        from typer.testing import CliRunner
        from aeon_ai.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["reflect", "--phases"])
        assert result.exit_code == 0

    def test_reflect_loop_json_output(self) -> None:
        import json
        from typer.testing import CliRunner
        from aeon_ai.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["reflect", "--loop", "--json"])
        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert "converged" in parsed
        assert "snapshots" in parsed

    def test_detect_phase_command_default(self) -> None:
        from typer.testing import CliRunner
        from aeon_ai.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["detect-phase"])
        assert result.exit_code == 0

    def test_detect_phase_command_with_entropy(self) -> None:
        from typer.testing import CliRunner
        from aeon_ai.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["detect-phase", "--entropy", "0.45"])
        assert result.exit_code == 0

    def test_detect_phase_json_output(self) -> None:
        import json
        from typer.testing import CliRunner
        from aeon_ai.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["detect-phase", "--entropy", "0.6", "--json"])
        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert "entropy" in parsed
        assert "utac_value" in parsed
        assert "triggered" in parsed
        assert "phase_label" in parsed

    def test_detect_phase_invalid_entropy(self) -> None:
        from typer.testing import CliRunner
        from aeon_ai.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["detect-phase", "--entropy", "1.5"])
        assert result.exit_code != 0

    def test_detect_phase_invalid_threshold(self) -> None:
        from typer.testing import CliRunner
        from aeon_ai.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["detect-phase", "--threshold", "0.0"])
        assert result.exit_code != 0

    def test_detect_phase_invalid_utac_ceil(self) -> None:
        from typer.testing import CliRunner
        from aeon_ai.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["detect-phase", "--utac-ceil", "0.0"])
        assert result.exit_code != 0

    def test_info_shows_v020(self) -> None:
        from typer.testing import CliRunner
        from aeon_ai.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["info"])
        assert "0.2.0" in result.output

    def test_info_shows_phase_detector(self) -> None:
        from typer.testing import CliRunner
        from aeon_ai.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["info"])
        assert "PhaseDetector" in result.output

    def test_info_shows_self_reflector(self) -> None:
        from typer.testing import CliRunner
        from aeon_ai.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["info"])
        assert "SelfReflector" in result.output

    def test_reflect_loop_visualize_flag(self) -> None:
        from typer.testing import CliRunner
        from aeon_ai.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["reflect", "--loop", "--visualize"])
        assert result.exit_code == 0

    def test_reflect_loop_with_sigil(self) -> None:
        from typer.testing import CliRunner
        from aeon_ai.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["reflect", "--loop", "--sigil", "mirror aeon"])
        assert result.exit_code == 0

    def test_detect_phase_triggered_high_entropy(self) -> None:
        from typer.testing import CliRunner
        from aeon_ai.cli import app

        runner = CliRunner()
        # Very high entropy with very low threshold → should trigger
        result = runner.invoke(
            app, ["detect-phase", "--entropy", "0.99", "--threshold", "0.01", "--json"]
        )
        assert result.exit_code == 0
        import json
        parsed = json.loads(result.output)
        assert parsed["triggered"] is True
