"""Orchestrator: Unified-mandala neural adapter and pipeline coordinator.

The Orchestrator wires together all aeon-ai components into a single coherent
pipeline that mirrors the ``unified-mandala-neural/orchestrator`` architecture:

    FieldBridge   -> sample cosmic moment
    AeonLayer     -> compute Lagrangian-weighted output
    MirrorCore    -> self-reflective transformation
    CREPEvaluator -> quality scoring
    SigillinBridge -> symbolic activation
    PhaseDetector -> real-time phase-transition detection (v0.2.0)

Each ``run`` call returns an :class:`OrchestratorResult` with all intermediate
and final values for full traceability.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from aeon_ai.aeon_layer import AeonLayer
from aeon_ai.crep_evaluator import CREPEvaluator, CREPScore
from aeon_ai.field_bridge import CosmicMoment, FieldBridge
from aeon_ai.mirror_core import MirrorCore, ReflectionState
from aeon_ai.phase_detector import PhaseDetector, PhaseTransitionEvent
from aeon_ai.sigillin_bridge import SigillinBridge


@dataclass
class OrchestratorResult:
    """Full output record from a single Orchestrator pipeline run.

    Attributes:
        input_s_a:         Original auditory signal amplitude.
        input_s_v:         Original visual signal amplitude.
        lagrangian_out:    Raw AeonLayer Lagrangian output.
        reflection:        Final :class:`~aeon_ai.mirror_core.ReflectionState`.
        crep_score:        :class:`~aeon_ai.crep_evaluator.CREPScore` of the run.
        sigil_activations: Dict mapping sigil-id to activation score.
        cosmic_moment:     :class:`~aeon_ai.field_bridge.CosmicMoment` snapshot.
        modulation:        Cosmic modulation factor applied.
        phase_events:      Phase-transition events detected during this run (v0.2.0).
        metadata:          Arbitrary tracing key-value pairs.
    """

    input_s_a: float
    input_s_v: float
    lagrangian_out: float
    reflection: ReflectionState
    crep_score: CREPScore
    sigil_activations: dict[str, float]
    cosmic_moment: CosmicMoment
    modulation: float
    phase_events: list[PhaseTransitionEvent] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def final_output(self) -> float:
        """Modulated reflection output: ``reflection.output_val * modulation``."""
        return self.reflection.output_val * self.modulation

    def as_dict(self) -> dict[str, Any]:
        """Export the full result as a nested dictionary.

        Returns:
            Flat-ish dict suitable for JSON serialisation.
        """
        return {
            "input": {"s_a": self.input_s_a, "s_v": self.input_s_v},
            "lagrangian_out": self.lagrangian_out,
            "final_output": self.final_output,
            "reflection": {
                "phase": self.reflection.phase.name,
                "output_val": self.reflection.output_val,
                "entropy": self.reflection.entropy,
            },
            "crep": self.crep_score.as_dict(),
            "sigil_activations": self.sigil_activations,
            "cosmic_moment": self.cosmic_moment.as_dict(),
            "modulation": self.modulation,
            "phase_events": [ev.as_dict() for ev in self.phase_events],
            "metadata": self.metadata,
        }


class Orchestrator:
    """Unified-mandala neural adapter coordinating all aeon-ai components.

    Instantiates and wires together:
    - :class:`~aeon_ai.aeon_layer.AeonLayer`
    - :class:`~aeon_ai.mirror_core.MirrorCore`
    - :class:`~aeon_ai.crep_evaluator.CREPEvaluator`
    - :class:`~aeon_ai.sigillin_bridge.SigillinBridge`
    - :class:`~aeon_ai.field_bridge.FieldBridge`
    - :class:`~aeon_ai.phase_detector.PhaseDetector` (v0.2.0)

    Integration with ``unified-mandala-neural`` is attempted at instantiation;
    native implementations are used as fallback.

    Example:
        >>> orch = Orchestrator()
        >>> result = orch.run(s_a=0.7, s_v=0.5, sigil_text="mirror aeon genesis")
        >>> result.crep_score.score > 0
        True
    """

    def __init__(
        self,
        delta: float = 0.0,
        mirror_depth: int = 2,
        crep_weights: tuple[float, float, float, float] = (0.25, 0.25, 0.25, 0.25),
        base_entropy: float = 0.3,
        load_sigils: bool = True,
        entropy_threshold: float = 0.37,
    ) -> None:
        """Initialise Orchestrator with default sub-components.

        Args:
            delta:             AeonLayer deformation parameter delta.
            mirror_depth:      Recursive reflection depth for MirrorCore.
            crep_weights:      Per-dimension CREP weights (C, R, E, P).
            base_entropy:      Baseline entropy for FieldBridge.
            load_sigils:       Pre-load built-in sigils in SigillinBridge.
            entropy_threshold: Entropy pivot for PhaseDetector (v0.2.0).
        """
        self.aeon_layer = AeonLayer(delta=delta)
        self.mirror_core = MirrorCore(depth=mirror_depth)
        self.crep_evaluator = CREPEvaluator(weights=crep_weights)
        self.sigillin = SigillinBridge(load_defaults=load_sigils)
        self.field_bridge = FieldBridge(base_entropy=base_entropy)
        self.phase_detector = PhaseDetector(entropy_threshold=entropy_threshold)
        self._results: list[OrchestratorResult] = []
        self._try_load_external()

    # ------------------------------------------------------------------
    # Primary API
    # ------------------------------------------------------------------

    def run(
        self,
        s_a: float,
        s_v: float,
        t: float = 1.0,
        sigil_text: str = "",
        signal: list[float] | None = None,
        crep_external: dict[str, float] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> OrchestratorResult:
        """Execute the full aeon-ai pipeline for one input vector.

        Pipeline order:
            1. Sample CosmicMoment from FieldBridge
            2. Adjust Lagrangian delta via cosmic tension
            3. Compute AeonLayer forward pass
            4. Run MirrorCore reflection with cosmic entropy
            5. Evaluate CREP quality score
            6. Compute Sigillin activations
            7. Apply cosmic modulation factor
            8. Detect phase-transition events (v0.2.0)
            9. Package and return :class:`OrchestratorResult`

        Args:
            s_a:           Auditory signal amplitude.
            s_v:           Visual signal amplitude.
            t:             Time step for AeonLayer (> 0).
            sigil_text:    Text string for Sigillin activation.
            signal:        Numeric signal for CREP emergence/coherence.
            crep_external: Pre-computed CREP dimension overrides.
            metadata:      Extra key-value pairs attached to the result.

        Returns:
            :class:`OrchestratorResult` with all intermediate values.
        """
        # 1. Cosmic moment
        moment = self.field_bridge.sample_moment()

        # 2. Adjust delta
        adjusted_delta = self.field_bridge.adjust_delta(self.aeon_layer.config.delta, moment)
        self.aeon_layer.config.delta = adjusted_delta

        # 3. AeonLayer
        l_out = self.aeon_layer.forward(s_a=s_a, s_v=s_v, t=t)

        # 4. MirrorCore
        cosmic_entropy = self.field_bridge.inject_entropy(moment)
        self.mirror_core.reset()
        reflection = self.mirror_core.reflect(l_out, entropy=cosmic_entropy)

        # 5. CREP
        sig = signal or [s_a, s_v, l_out, reflection.output_val]
        crep_score = self.crep_evaluator.evaluate(
            signal=sig,
            text=sigil_text,
            external=crep_external,
        )

        # 6. Sigillin
        activations = self.sigillin.activate(sigil_text) if sigil_text else {}

        # 7. Modulation
        mod = self.field_bridge.modulation_factor(moment)

        # 8. Phase-transition detection (v0.2.0)
        self.phase_detector.reset()
        phase_events = self.phase_detector.process_trace(self.mirror_core.trace)

        # 9. Package
        result = OrchestratorResult(
            input_s_a=s_a,
            input_s_v=s_v,
            lagrangian_out=l_out,
            reflection=reflection,
            crep_score=crep_score,
            sigil_activations=activations,
            cosmic_moment=moment,
            modulation=mod,
            phase_events=phase_events,
            metadata=metadata or {},
        )
        self._results.append(result)
        return result

    def run_batch(
        self,
        inputs: list[dict[str, Any]],
    ) -> list[OrchestratorResult]:
        """Run the pipeline for multiple input vectors.

        Args:
            inputs: List of keyword-argument dicts forwarded to :meth:`run`.

        Returns:
            List of :class:`OrchestratorResult` in input order.
        """
        return [self.run(**inp) for inp in inputs]

    def reset(self) -> None:
        """Reset all sub-components and result history."""
        self.mirror_core.reset()
        self.crep_evaluator.reset()
        self.field_bridge.reset()
        self.phase_detector.reset()
        self._results.clear()

    @property
    def results(self) -> list[OrchestratorResult]:
        """All results produced since last :meth:`reset`."""
        return list(self._results)

    def state_dict(self) -> dict[str, Any]:
        """Serialise orchestrator state.

        Returns:
            Nested dict with all sub-component states.
        """
        return {
            "aeon_layer": self.aeon_layer.state_dict(),
            "mirror_core": self.mirror_core.state_dict(),
            "phase_detector": self.phase_detector.state_dict(),
            "result_count": len(self._results),
        }

    # ------------------------------------------------------------------
    # External integration
    # ------------------------------------------------------------------

    def _try_load_external(self) -> None:
        """Attempt to integrate unified-mandala-neural orchestrator hooks."""
        try:
            import unified_mandala.neural.orchestrator as _umn  # type: ignore[import-untyped]

            _umn.register(self)
        except (ImportError, AttributeError, Exception):  # noqa: BLE001
            pass  # optional integration; fail silently

    def __repr__(self) -> str:
        """Return developer-friendly string representation."""
        return (
            f"Orchestrator(delta={self.aeon_layer.config.delta}, "
            f"mirror_depth={self.mirror_core.depth}, "
            f"results={len(self._results)})"
        )
