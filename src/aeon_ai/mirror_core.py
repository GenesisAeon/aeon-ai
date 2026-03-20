"""MirrorCore: Self-Reflection Loop, Mirror-Machine phases, and UTAC-Logistic.

The Mirror Machine processes a symbolic state through four canonical phases:

    INIT      → Receive raw signal / seed state
    REFLECT   → Apply self-referential transformation (mirror pass)
    INTEGRATE → Merge reflected output with prior context
    EMIT      → Produce observable output and update memory

UTAC-Logistic (Universal Transformation and Adaptation Coefficient):

    UTAC(x) = L / (1 + exp(-k · (x − x₀)))

Where L is the carrying capacity, k the growth rate, and x₀ the midpoint.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any


class MirrorPhase(Enum):
    """Canonical phases of the Mirror Machine pipeline."""

    INIT = auto()
    REFLECT = auto()
    INTEGRATE = auto()
    EMIT = auto()


@dataclass
class ReflectionState:
    """Immutable snapshot of a single mirror-loop iteration.

    Attributes:
        phase:      The :class:`MirrorPhase` in which this state was captured.
        input_val:  Scalar input to the current phase.
        output_val: Scalar output produced by the phase.
        entropy:    Shannon-like entropy of the state (bits).
        metadata:   Arbitrary key-value pairs for tracing.
    """

    phase: MirrorPhase
    input_val: float
    output_val: float
    entropy: float
    metadata: dict[str, Any] = field(default_factory=dict)


def utac_logistic(
    x: float,
    carrying_capacity: float = 1.0,
    growth_rate: float = 1.0,
    midpoint: float = 0.0,
) -> float:
    r"""Compute the UTAC-Logistic value.

    .. math::
        \\text{UTAC}(x) = \\frac{L}{1 + e^{-k(x - x_0)}}

    Args:
        x:                 Input value.
        carrying_capacity: Saturation ceiling :math:`L`.
        growth_rate:       Steepness of the sigmoid :math:`k`.
        midpoint:          Inflection point :math:`x_0`.

    Returns:
        UTAC-Logistic coefficient in ``(0, L)``.
    """
    exponent = -growth_rate * (x - midpoint)
    # clamp to avoid overflow in exp
    exponent = max(-500.0, min(500.0, exponent))
    return carrying_capacity / (1.0 + math.exp(exponent))


def _shannon_entropy(value: float, epsilon: float = 1e-9) -> float:
    """Estimate single-value Shannon entropy (bits).

    Maps ``value`` to a pseudo-probability *p = sigmoid(value)* and computes
    the binary entropy H(p) = -p log2(p) - (1-p) log2(1-p).

    Args:
        value:   Input scalar.
        epsilon: Guard against log(0).

    Returns:
        Entropy in bits, in [0, 1].
    """
    p = 1.0 / (1.0 + math.exp(-float(value)))
    p = max(epsilon, min(1.0 - epsilon, p))
    return -(p * math.log2(p) + (1.0 - p) * math.log2(1.0 - p))


class MirrorCore:
    """Self-Reflection Loop implementing the Mirror Machine pipeline.

    The loop processes an input scalar through four :class:`MirrorPhase` stages
    and records each :class:`ReflectionState` for introspection.

    UTAC-Logistic is applied during the INTEGRATE phase to adaptively modulate
    the merged signal.

    Example:
        >>> core = MirrorCore(depth=3)
        >>> result = core.reflect(0.7, entropy=0.5)
        >>> result.phase
        <MirrorPhase.EMIT: 4>
    """

    def __init__(
        self,
        depth: int = 1,
        utac_capacity: float = 1.0,
        utac_growth: float = 1.0,
        utac_midpoint: float = 0.0,
        memory_decay: float = 0.9,
    ) -> None:
        """Initialise MirrorCore.

        Args:
            depth:          Number of recursive mirror passes (REFLECT depth).
            utac_capacity:  Carrying capacity L for UTAC-Logistic.
            utac_growth:    Growth rate k for UTAC-Logistic.
            utac_midpoint:  Midpoint x₀ for UTAC-Logistic.
            memory_decay:   Exponential decay factor for context memory (0–1).
        """
        if depth < 1:
            raise ValueError(f"depth must be ≥ 1, got {depth!r}")
        if not 0.0 < memory_decay <= 1.0:
            raise ValueError(f"memory_decay must be in (0, 1], got {memory_decay!r}")

        self.depth = depth
        self.utac_capacity = utac_capacity
        self.utac_growth = utac_growth
        self.utac_midpoint = utac_midpoint
        self.memory_decay = memory_decay

        self._memory: float = 0.0
        self._trace: list[ReflectionState] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reflect(self, value: float, entropy: float = 0.0) -> ReflectionState:
        """Run the full four-phase Mirror Machine pipeline.

        Args:
            value:   Input scalar to the INIT phase.
            entropy: External entropy hint (overrides internal calculation
                     when > 0; otherwise computed from *value*).

        Returns:
            :class:`ReflectionState` of the final EMIT phase.
        """
        state = self._phase_init(value, entropy)
        state = self._phase_reflect(state)
        state = self._phase_integrate(state)
        state = self._phase_emit(state)
        return state

    def reset(self) -> None:
        """Reset memory and trace to initial state."""
        self._memory = 0.0
        self._trace.clear()

    @property
    def trace(self) -> list[ReflectionState]:
        """Read-only view of the full reflection trace."""
        return list(self._trace)

    @property
    def memory(self) -> float:
        """Current context memory value."""
        return self._memory

    def state_dict(self) -> dict[str, Any]:
        """Serialise core configuration and runtime state.

        Returns:
            Dictionary suitable for inspection or persistence.
        """
        return {
            "depth": self.depth,
            "utac_capacity": self.utac_capacity,
            "utac_growth": self.utac_growth,
            "utac_midpoint": self.utac_midpoint,
            "memory_decay": self.memory_decay,
            "memory": self._memory,
            "trace_len": len(self._trace),
        }

    # ------------------------------------------------------------------
    # Phase implementations
    # ------------------------------------------------------------------

    def _phase_init(self, value: float, entropy: float) -> ReflectionState:
        """INIT phase: receive and normalise input signal."""
        ent = entropy if entropy > 0.0 else _shannon_entropy(value)
        state = ReflectionState(
            phase=MirrorPhase.INIT,
            input_val=value,
            output_val=value,
            entropy=ent,
            metadata={"depth": self.depth},
        )
        self._trace.append(state)
        return state

    def _phase_reflect(self, state: ReflectionState) -> ReflectionState:
        """REFLECT phase: apply recursive self-referential transformation."""
        val = state.output_val
        for i in range(self.depth):
            # Mirror transform: tanh(val) weighted by iteration-scaled entropy
            val = math.tanh(val * (1.0 + state.entropy * (i + 1) * 0.1))

        reflected = ReflectionState(
            phase=MirrorPhase.REFLECT,
            input_val=state.output_val,
            output_val=val,
            entropy=_shannon_entropy(val),
            metadata={"passes": self.depth},
        )
        self._trace.append(reflected)
        return reflected

    def _phase_integrate(self, state: ReflectionState) -> ReflectionState:
        """INTEGRATE phase: merge reflected signal with context via UTAC-Logistic."""
        merged = state.output_val * (1.0 - self.memory_decay) + self._memory * self.memory_decay
        utac_coeff = utac_logistic(
            merged,
            carrying_capacity=self.utac_capacity,
            growth_rate=self.utac_growth,
            midpoint=self.utac_midpoint,
        )
        integrated_val = merged * utac_coeff
        integrated = ReflectionState(
            phase=MirrorPhase.INTEGRATE,
            input_val=state.output_val,
            output_val=integrated_val,
            entropy=_shannon_entropy(integrated_val),
            metadata={"utac_coeff": utac_coeff, "memory_before": self._memory},
        )
        self._memory = integrated_val
        self._trace.append(integrated)
        return integrated

    def _phase_emit(self, state: ReflectionState) -> ReflectionState:
        """EMIT phase: produce observable output."""
        emitted = ReflectionState(
            phase=MirrorPhase.EMIT,
            input_val=state.output_val,
            output_val=state.output_val,
            entropy=state.entropy,
            metadata={"memory_after": self._memory},
        )
        self._trace.append(emitted)
        return emitted

    def __repr__(self) -> str:
        """Return developer-friendly string representation."""
        return (
            f"MirrorCore(depth={self.depth}, memory={self._memory:.4f}, "
            f"trace_len={len(self._trace)})"
        )
