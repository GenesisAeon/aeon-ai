"""SelfReflector: Native AeonAI closed-loop self-reflection engine.

Implements the *self-reflection loop*: an iterative coupling between the
CREP quality score, the fieldtheory Lagrangian gradient, and the Sigillin
symbolic bridges.

Reflection Loop Dynamics
------------------------
Starting from an initial signal pair :math:`(S_A, S_V)`, the loop iterates
up to ``MAX_ITER = 7`` times, updating :math:`S_A` and :math:`S_V` at each
step via the Lagrangian gradient, then checking convergence against the
CREP harmonic-mean score.

.. math::

    S_A^{(i+1)} = S_A^{(i)} + \\eta \\cdot \\frac{\\partial L}{\\partial S_A}

    S_V^{(i+1)} = S_V^{(i)} + \\eta \\cdot \\frac{\\partial L}{\\partial S_V}

    \\text{convergence: } |\\text{CREP}^{(i)} - \\text{CREP}^{(i-1)}| < \\epsilon

Where :math:`\\eta` is a step-size controlled by the UTAC-modulated entropy,
and CREP is evaluated over the reflection signal at each iteration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from aeon_ai.aeon_layer import AeonLayer, lagrangian_gradient
from aeon_ai.crep_evaluator import CREPEvaluator, CREPScore
from aeon_ai.mirror_core import MirrorCore, ReflectionState
from aeon_ai.sigillin_bridge import SigillinBridge

MAX_ITER: int = 7
"""Maximum number of self-reflection iterations."""

_DEFAULT_ENTROPY_THRESHOLD: float = 0.37
_DEFAULT_STEP_SIZE: float = 0.05
_CONVERGENCE_EPSILON: float = 1e-4


@dataclass
class IterationSnapshot:
    """Snapshot of a single self-reflection iteration.

    Attributes:
        iteration:        Zero-based iteration index.
        s_a:              Auditory amplitude after this iteration.
        s_v:              Visual amplitude after this iteration.
        lagrangian_value: :math:`L(S_A, S_V, \\delta, t)` at this step.
        crep_score:       :class:`~aeon_ai.crep_evaluator.CREPScore` at this step.
        reflection:       :class:`~aeon_ai.mirror_core.ReflectionState` (EMIT).
        sigil_activations: Active sigil scores at this step.
        entropy:          Effective entropy used in this iteration.
        converged:        Whether convergence was reached at or before this step.
        metadata:         Arbitrary tracing key-value pairs.
    """

    iteration: int
    s_a: float
    s_v: float
    lagrangian_value: float
    crep_score: CREPScore
    reflection: ReflectionState
    sigil_activations: dict[str, float]
    entropy: float
    converged: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """Export the snapshot as a flat dictionary.

        Returns:
            JSON-serialisable representation.
        """
        return {
            "iteration": self.iteration,
            "s_a": self.s_a,
            "s_v": self.s_v,
            "lagrangian_value": self.lagrangian_value,
            "crep_score": self.crep_score.as_dict(),
            "reflection": {
                "phase": self.reflection.phase.name,
                "output_val": self.reflection.output_val,
                "entropy": self.reflection.entropy,
            },
            "sigil_activations": self.sigil_activations,
            "entropy": self.entropy,
            "converged": self.converged,
            "metadata": self.metadata,
        }


@dataclass
class ReflectionLoopResult:
    """Complete result of a :meth:`SelfReflector.self_reflect` call.

    Attributes:
        snapshots:         Ordered list of per-iteration snapshots.
        converged:         Whether the loop converged within ``MAX_ITER``.
        final_crep:        CREP score of the final iteration.
        final_lagrangian:  Lagrangian value at final iteration.
        final_s_a:         :math:`S_A` at final iteration.
        final_s_v:         :math:`S_V` at final iteration.
        total_iterations:  Number of iterations executed.
        entropy_threshold: Threshold that governed convergence.
        metadata:          Arbitrary tracing key-value pairs.
    """

    snapshots: list[IterationSnapshot]
    converged: bool
    final_crep: CREPScore
    final_lagrangian: float
    final_s_a: float
    final_s_v: float
    total_iterations: int
    entropy_threshold: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """Export the full result as a nested dictionary.

        Returns:
            JSON-serialisable representation.
        """
        return {
            "converged": self.converged,
            "total_iterations": self.total_iterations,
            "entropy_threshold": self.entropy_threshold,
            "final_s_a": self.final_s_a,
            "final_s_v": self.final_s_v,
            "final_lagrangian": self.final_lagrangian,
            "final_crep": self.final_crep.as_dict(),
            "snapshots": [s.as_dict() for s in self.snapshots],
            "metadata": self.metadata,
        }


class SelfReflector:
    """Native AeonAI self-reflection loop.

    Iteratively couples the CREP quality score, Lagrangian gradient updates,
    and Sigillin symbolic activations over at most :data:`MAX_ITER` iterations.

    The loop converges when the absolute change in the CREP harmonic-mean score
    between consecutive iterations falls below ``_CONVERGENCE_EPSILON``.

    Example:
        >>> reflector = SelfReflector()
        >>> result = reflector.self_reflect(entropy_threshold=0.37)
        >>> result.total_iterations <= 7
        True
        >>> 0.0 <= result.final_crep.score <= 1.0
        True
    """

    def __init__(
        self,
        delta: float = 0.0,
        mirror_depth: int = 2,
        step_size: float = _DEFAULT_STEP_SIZE,
        crep_weights: tuple[float, float, float, float] = (0.25, 0.25, 0.25, 0.25),
        sigil_text: str = "aeon mirror genesis",
    ) -> None:
        """Initialise SelfReflector.

        Args:
            delta:        AeonLayer deformation parameter δ.
            mirror_depth: Recursive reflection depth for MirrorCore.
            step_size:    η — Lagrangian gradient step size.
            crep_weights: Per-dimension CREP weights (C, R, E, P).
            sigil_text:   Default sigil trigger text for each iteration.
        """
        self.delta = delta
        self.step_size = step_size
        self.sigil_text = sigil_text

        self._aeon_layer = AeonLayer(delta=delta)
        self._mirror_core = MirrorCore(depth=mirror_depth)
        self._crep_evaluator = CREPEvaluator(weights=crep_weights)
        self._sigillin = SigillinBridge(load_defaults=True)
        self._loop_history: list[ReflectionLoopResult] = []

    # ------------------------------------------------------------------
    # Primary API
    # ------------------------------------------------------------------

    def self_reflect(
        self,
        s_a: float = 0.7,
        s_v: float = 0.6,
        t: float = 1.0,
        entropy_threshold: float = _DEFAULT_ENTROPY_THRESHOLD,
        sigil_text: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ReflectionLoopResult:
        """Run the closed-loop self-reflection iteration.

        Iterates at most :data:`MAX_ITER` (7) times, updating :math:`S_A`
        and :math:`S_V` via Lagrangian gradients and checking CREP convergence.

        At each iteration:

        1. Compute the AeonLayer Lagrangian :math:`L`.
        2. Run MirrorCore reflection with entropy := CREP score from prior step.
        3. Evaluate CREP from the reflection signal.
        4. Compute Sigillin activations.
        5. Update :math:`S_A` and :math:`S_V` via Lagrangian gradient.
        6. Check convergence: :math:`|\\text{CREP}_i - \\text{CREP}_{i-1}| < \\epsilon`.

        Args:
            s_a:               Initial auditory signal amplitude.
            s_v:               Initial visual signal amplitude.
            t:                 Time step for Lagrangian (> 0).
            entropy_threshold: Entropy pivot for UTAC trigger; also governs
                               the entropy injected on the first iteration.
            sigil_text:        Sigil trigger text (overrides constructor default).
            metadata:          Extra key-value pairs attached to the result.

        Returns:
            :class:`ReflectionLoopResult` with full iteration history.
        """
        text = sigil_text if sigil_text is not None else self.sigil_text
        snapshots: list[IterationSnapshot] = []
        prev_crep_score: float | None = None
        converged = False
        current_entropy = entropy_threshold  # warm-start entropy

        cur_s_a, cur_s_v = s_a, s_v

        for i in range(MAX_ITER):
            # Step 1: Lagrangian
            l_val = self._aeon_layer.forward(s_a=cur_s_a, s_v=cur_s_v, t=t)

            # Step 2: Mirror reflection (entropy driven by CREP from prior step)
            reflection = self._mirror_core.reflect(l_val, entropy=current_entropy)

            # Step 3: CREP evaluation
            signal = [cur_s_a, cur_s_v, l_val, reflection.output_val]
            crep = self._crep_evaluator.evaluate(signal=signal, text=text)

            # Step 4: Sigillin activations
            activations = self._sigillin.activate(text) if text else {}

            # Convergence check (requires at least one prior step)
            if prev_crep_score is not None:
                delta_crep = abs(crep.score - prev_crep_score)
                if delta_crep < _CONVERGENCE_EPSILON:
                    snap = IterationSnapshot(
                        iteration=i,
                        s_a=cur_s_a,
                        s_v=cur_s_v,
                        lagrangian_value=l_val,
                        crep_score=crep,
                        reflection=reflection,
                        sigil_activations=activations,
                        entropy=current_entropy,
                        converged=True,
                        metadata={"delta_crep": delta_crep},
                    )
                    snapshots.append(snap)
                    converged = True
                    break

            # Snapshot (non-converged)
            snap = IterationSnapshot(
                iteration=i,
                s_a=cur_s_a,
                s_v=cur_s_v,
                lagrangian_value=l_val,
                crep_score=crep,
                reflection=reflection,
                sigil_activations=activations,
                entropy=current_entropy,
                converged=False,
                metadata={},
            )
            snapshots.append(snap)

            # Step 5: Update S_A, S_V via Lagrangian gradient
            grad = lagrangian_gradient(
                s_a=cur_s_a,
                s_v=cur_s_v,
                delta=self._aeon_layer.config.delta,
                t=t,
            )
            sigillin_boost = max(activations.values()) if activations else 0.0
            effective_step = self.step_size * (1.0 + sigillin_boost * 0.1)

            cur_s_a = cur_s_a + effective_step * grad["dL/ds_a"]
            cur_s_v = cur_s_v + effective_step * grad["dL/ds_v"]

            # Clip to reasonable range
            cur_s_a = max(-10.0, min(10.0, cur_s_a))
            cur_s_v = max(-10.0, min(10.0, cur_s_v))

            # Step 6: Update entropy from reflection + CREP coupling
            current_entropy = max(
                0.01,
                min(0.99, crep.score * entropy_threshold + (1 - crep.score) * reflection.entropy),
            )

            prev_crep_score = crep.score

        last = snapshots[-1]
        result = ReflectionLoopResult(
            snapshots=snapshots,
            converged=converged,
            final_crep=last.crep_score,
            final_lagrangian=last.lagrangian_value,
            final_s_a=last.s_a,
            final_s_v=last.s_v,
            total_iterations=len(snapshots),
            entropy_threshold=entropy_threshold,
            metadata=metadata or {},
        )
        self._loop_history.append(result)
        return result

    # ------------------------------------------------------------------
    # Convenience / introspection
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset all sub-components and loop history."""
        self._mirror_core.reset()
        self._crep_evaluator.reset()
        self._aeon_layer.reset_history()
        self._loop_history.clear()

    @property
    def loop_history(self) -> list[ReflectionLoopResult]:
        """All :class:`ReflectionLoopResult` objects produced since last reset."""
        return list(self._loop_history)

    def state_dict(self) -> dict[str, Any]:
        """Serialise reflector configuration.

        Returns:
            Dictionary suitable for inspection or persistence.
        """
        return {
            "delta": self.delta,
            "step_size": self.step_size,
            "sigil_text": self.sigil_text,
            "aeon_layer": self._aeon_layer.state_dict(),
            "mirror_core": self._mirror_core.state_dict(),
            "loop_history_len": len(self._loop_history),
        }

    def __repr__(self) -> str:
        """Return developer-friendly string representation."""
        return (
            f"SelfReflector("
            f"delta={self.delta}, "
            f"step_size={self.step_size}, "
            f"loops={len(self._loop_history)})"
        )
