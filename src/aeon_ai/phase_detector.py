"""PhaseDetector: Real-time phase-transition detection for the Mirror Machine.

Implements real-time monitoring of :class:`~aeon_ai.mirror_core.MirrorPhase`
transitions, collapse-detection, and UTAC-Logistic trigger thresholds sourced
from the ``mirror-machine`` stack.

Phase Transition Model
----------------------
A *transition event* is emitted whenever the field entropy crosses a threshold
or the UTAC-Logistic value exceeds a trigger ceiling.  Collapse is detected when
successive reflection states converge below a stability floor.

.. math::

    \\Phi_{\\text{trigger}}(H) = \\frac{L}{1 + e^{-k(H - H_0)}}

    \\Delta_{\\text{collapse}} = \\left| x_n - x_{n-1} \\right| < \\epsilon_{\\text{stab}}

Where :math:`H` is the current Shannon entropy and :math:`H_0` is the pivot
entropy threshold.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

from aeon_ai.mirror_core import MirrorPhase, ReflectionState, utac_logistic


class TransitionType(Enum):
    """Classification of a phase-transition event.

    Attributes:
        FORWARD:      Normal sequential phase advancement.
        COLLAPSE:     Convergence below stability floor (signal collapse).
        UTAC_TRIGGER: UTAC-Logistic value exceeded the trigger ceiling.
        FORCED:       Externally imposed transition (entropy override).
    """

    FORWARD = auto()
    COLLAPSE = auto()
    UTAC_TRIGGER = auto()
    FORCED = auto()


@dataclass
class PhaseTransitionEvent:
    """Record of a single detected phase-transition event.

    Attributes:
        source_phase:   The :class:`~aeon_ai.mirror_core.MirrorPhase` before the event.
        target_phase:   The :class:`~aeon_ai.mirror_core.MirrorPhase` after the event.
        transition_type: Classification of the transition.
        entropy:        Entropy at the moment of detection.
        utac_value:     UTAC-Logistic coefficient that triggered or was present.
        timestamp:      Unix timestamp of detection.
        metadata:       Arbitrary key-value pairs for tracing.
    """

    source_phase: MirrorPhase
    target_phase: MirrorPhase
    transition_type: TransitionType
    entropy: float
    utac_value: float
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """Export the event as a flat dictionary.

        Returns:
            JSON-serialisable representation.
        """
        return {
            "source_phase": self.source_phase.name,
            "target_phase": self.target_phase.name,
            "transition_type": self.transition_type.name,
            "entropy": self.entropy,
            "utac_value": self.utac_value,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# Phase ordering for forward-transition detection
# ---------------------------------------------------------------------------

_PHASE_ORDER: dict[MirrorPhase, int] = {
    MirrorPhase.INIT: 0,
    MirrorPhase.REFLECT: 1,
    MirrorPhase.INTEGRATE: 2,
    MirrorPhase.EMIT: 3,
}

_PHASE_SUCCESSOR: dict[MirrorPhase, MirrorPhase] = {
    MirrorPhase.INIT: MirrorPhase.REFLECT,
    MirrorPhase.REFLECT: MirrorPhase.INTEGRATE,
    MirrorPhase.INTEGRATE: MirrorPhase.EMIT,
    MirrorPhase.EMIT: MirrorPhase.INIT,  # cycle back for loop mode
}


class PhaseDetector:
    """Real-time phase-transition detector for the Mirror Machine.

    Monitors a stream of :class:`~aeon_ai.mirror_core.ReflectionState` objects
    and emits :class:`PhaseTransitionEvent` records when:

    - A forward phase transition occurs (INIT → REFLECT → INTEGRATE → EMIT).
    - The signal collapses (consecutive output values converge below
      *stability_floor*).
    - The UTAC-Logistic value at the current entropy exceeds *utac_trigger_ceil*.

    Integration with the ``mirror-machine`` package is attempted at instantiation;
    the detector operates standalone when that package is absent.

    Example:
        >>> detector = PhaseDetector(entropy_threshold=0.37)
        >>> from aeon_ai.mirror_core import MirrorCore
        >>> core = MirrorCore(depth=2)
        >>> state = core.reflect(0.6, entropy=0.4)
        >>> events = detector.process_trace(core.trace)
        >>> all(isinstance(e, PhaseTransitionEvent) for e in events)
        True
    """

    def __init__(
        self,
        entropy_threshold: float = 0.37,
        stability_floor: float = 1e-4,
        utac_trigger_ceil: float = 0.85,
        utac_capacity: float = 1.0,
        utac_growth: float = 6.0,
    ) -> None:
        """Initialise PhaseDetector.

        Args:
            entropy_threshold: Pivot entropy :math:`H_0` for the UTAC trigger.
            stability_floor:   Δ below which consecutive outputs are "collapsed".
            utac_trigger_ceil: UTAC coefficient above which UTAC_TRIGGER fires.
            utac_capacity:     Carrying capacity :math:`L` of the UTAC sigmoid.
            utac_growth:       Growth rate :math:`k` of the UTAC sigmoid.
        """
        if not 0.0 < entropy_threshold < 1.0:
            raise ValueError(
                f"entropy_threshold must be in (0, 1), got {entropy_threshold!r}"
            )
        if stability_floor <= 0.0:
            raise ValueError(
                f"stability_floor must be > 0, got {stability_floor!r}"
            )
        if not 0.0 < utac_trigger_ceil <= 1.0:
            raise ValueError(
                f"utac_trigger_ceil must be in (0, 1], got {utac_trigger_ceil!r}"
            )

        self.entropy_threshold = entropy_threshold
        self.stability_floor = stability_floor
        self.utac_trigger_ceil = utac_trigger_ceil
        self.utac_capacity = utac_capacity
        self.utac_growth = utac_growth

        self._history: list[PhaseTransitionEvent] = []
        self._last_phase: MirrorPhase | None = None
        self._last_output: float | None = None
        self._try_load_external()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect_transition(
        self,
        state: ReflectionState,
    ) -> PhaseTransitionEvent | None:
        """Analyse a single :class:`~aeon_ai.mirror_core.ReflectionState`.

        Checks forward-transition, collapse, and UTAC trigger in priority order.

        Args:
            state: The reflection state to inspect.

        Returns:
            A :class:`PhaseTransitionEvent` if any condition is met, else ``None``.
        """
        utac_val = self._utac_at_entropy(state.entropy)
        event: PhaseTransitionEvent | None = None

        # Priority 1: UTAC trigger
        if utac_val >= self.utac_trigger_ceil:
            target = _PHASE_SUCCESSOR.get(state.phase, state.phase)
            event = PhaseTransitionEvent(
                source_phase=state.phase,
                target_phase=target,
                transition_type=TransitionType.UTAC_TRIGGER,
                entropy=state.entropy,
                utac_value=utac_val,
                metadata={"output_val": state.output_val},
            )

        # Priority 2: Collapse
        elif self._last_output is not None and self.detect_collapse_pair(
            self._last_output, state.output_val
        ):
            target = _PHASE_SUCCESSOR.get(state.phase, state.phase)
            event = PhaseTransitionEvent(
                source_phase=state.phase,
                target_phase=target,
                transition_type=TransitionType.COLLAPSE,
                entropy=state.entropy,
                utac_value=utac_val,
                metadata={
                    "delta": abs(state.output_val - self._last_output),
                    "stability_floor": self.stability_floor,
                },
            )

        # Priority 3: Forward phase transition
        elif (
            self._last_phase is not None
            and self._last_phase != state.phase
            and _PHASE_ORDER.get(state.phase, -1) == _PHASE_ORDER.get(self._last_phase, -1) + 1
        ):
            event = PhaseTransitionEvent(
                source_phase=self._last_phase,
                target_phase=state.phase,
                transition_type=TransitionType.FORWARD,
                entropy=state.entropy,
                utac_value=utac_val,
                metadata={"output_val": state.output_val},
            )

        self._last_phase = state.phase
        self._last_output = state.output_val

        if event is not None:
            self._history.append(event)

        return event

    def process_trace(
        self,
        trace: list[ReflectionState],
    ) -> list[PhaseTransitionEvent]:
        """Process a complete reflection trace and emit all detected events.

        Args:
            trace: Ordered list of :class:`~aeon_ai.mirror_core.ReflectionState`
                   from a :class:`~aeon_ai.mirror_core.MirrorCore` run.

        Returns:
            List of :class:`PhaseTransitionEvent` objects, in detection order.
        """
        events: list[PhaseTransitionEvent] = []
        for state in trace:
            ev = self.detect_transition(state)
            if ev is not None:
                events.append(ev)
        return events

    def detect_collapse(
        self,
        trace: list[ReflectionState],
        entropy_threshold: float | None = None,
    ) -> bool:
        """Determine whether the given trace exhibits signal collapse.

        A trace is collapsed if any two consecutive output values are within
        *stability_floor* of each other **and** the entropy is below
        *entropy_threshold*.

        Args:
            trace:             Reflection trace to inspect.
            entropy_threshold: Override entropy pivot (default: ``self.entropy_threshold``).

        Returns:
            ``True`` if collapse is detected.
        """
        threshold = entropy_threshold if entropy_threshold is not None else self.entropy_threshold
        for i in range(1, len(trace)):
            delta = abs(trace[i].output_val - trace[i - 1].output_val)
            if delta < self.stability_floor and trace[i].entropy < threshold:
                return True
        return False

    def detect_collapse_pair(self, val_a: float, val_b: float) -> bool:
        """Test a single (previous, current) output pair for collapse.

        Args:
            val_a: Previous output value.
            val_b: Current output value.

        Returns:
            ``True`` if |val_b - val_a| < stability_floor.
        """
        return abs(val_b - val_a) < self.stability_floor

    def utac_trigger_check(self, entropy: float) -> bool:
        """Check whether the UTAC-Logistic at *entropy* exceeds trigger ceiling.

        Args:
            entropy: Current field entropy ∈ [0, 1].

        Returns:
            ``True`` if the UTAC coefficient exceeds ``utac_trigger_ceil``.
        """
        return self._utac_at_entropy(entropy) >= self.utac_trigger_ceil

    def utac_value_at(self, entropy: float) -> float:
        """Return the UTAC-Logistic coefficient for a given entropy.

        .. math::
            \\Phi(H) = \\frac{L}{1 + e^{-k(H - H_0)}}

        Args:
            entropy: Field entropy ∈ [0, 1].

        Returns:
            UTAC coefficient ∈ (0, L).
        """
        return self._utac_at_entropy(entropy)

    def force_transition(
        self,
        source: MirrorPhase,
        target: MirrorPhase,
        entropy: float,
    ) -> PhaseTransitionEvent:
        """Emit a FORCED transition event (external override).

        Args:
            source:  Source phase.
            target:  Target phase.
            entropy: Entropy at override point.

        Returns:
            :class:`PhaseTransitionEvent` with type FORCED.
        """
        event = PhaseTransitionEvent(
            source_phase=source,
            target_phase=target,
            transition_type=TransitionType.FORCED,
            entropy=entropy,
            utac_value=self._utac_at_entropy(entropy),
            metadata={"forced": True},
        )
        self._history.append(event)
        return event

    @property
    def transition_history(self) -> list[PhaseTransitionEvent]:
        """All detected :class:`PhaseTransitionEvent` objects since last reset."""
        return list(self._history)

    def reset(self) -> None:
        """Clear internal state and history."""
        self._history.clear()
        self._last_phase = None
        self._last_output = None

    def state_dict(self) -> dict[str, Any]:
        """Serialise detector configuration.

        Returns:
            Dictionary suitable for inspection or persistence.
        """
        return {
            "entropy_threshold": self.entropy_threshold,
            "stability_floor": self.stability_floor,
            "utac_trigger_ceil": self.utac_trigger_ceil,
            "utac_capacity": self.utac_capacity,
            "utac_growth": self.utac_growth,
            "history_len": len(self._history),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _utac_at_entropy(self, entropy: float) -> float:
        """Compute UTAC-Logistic at the given entropy.

        Args:
            entropy: Input entropy value.

        Returns:
            UTAC coefficient.
        """
        return utac_logistic(
            entropy,
            carrying_capacity=self.utac_capacity,
            growth_rate=self.utac_growth,
            midpoint=self.entropy_threshold,
        )

    def _try_load_external(self) -> None:
        """Attempt to load phase-detection hooks from ``mirror-machine`` package."""
        try:
            import mirror_machine  # type: ignore[import-untyped]

            if hasattr(mirror_machine, "register_phase_detector"):
                mirror_machine.register_phase_detector(self)
        except (ImportError, Exception):  # noqa: BLE001
            pass  # optional — fail silently

    def __repr__(self) -> str:
        """Return developer-friendly string representation."""
        return (
            f"PhaseDetector("
            f"entropy_threshold={self.entropy_threshold}, "
            f"utac_trigger_ceil={self.utac_trigger_ceil}, "
            f"history_len={len(self._history)})"
        )


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------


def detect_phases_from_core(
    trace: list[ReflectionState],
    entropy_threshold: float = 0.37,
    utac_trigger_ceil: float = 0.85,
) -> list[PhaseTransitionEvent]:
    r"""One-shot phase detection from a completed mirror trace.

    Args:
        trace:             List of :class:`~aeon_ai.mirror_core.ReflectionState`.
        entropy_threshold: Entropy pivot :math:`H_0` for UTAC trigger.
        utac_trigger_ceil: UTAC ceiling for trigger detection.

    Returns:
        List of detected :class:`PhaseTransitionEvent` objects.
    """
    detector = PhaseDetector(
        entropy_threshold=entropy_threshold,
        utac_trigger_ceil=utac_trigger_ceil,
    )
    return detector.process_trace(trace)


def _entropy_to_phase_label(entropy: float, threshold: float = 0.37) -> str:
    """Map an entropy value to a human-readable phase-transition label.

    Args:
        entropy:   Input entropy ∈ [0, 1].
        threshold: Pivot threshold.

    Returns:
        Label string.
    """
    utac_val = utac_logistic(entropy, carrying_capacity=1.0, growth_rate=6.0, midpoint=threshold)
    if utac_val < 0.3:  # noqa: PLR2004
        return "STABLE"
    if utac_val < 0.7:  # noqa: PLR2004
        return "TRANSITIONING"
    return "COLLAPSE_RISK"


def entropy_phase_label(entropy: float, threshold: float = 0.37) -> str:
    """Public wrapper for entropy → phase label mapping.

    Args:
        entropy:   Field entropy ∈ [0, 1].
        threshold: Pivot threshold (default 0.37).

    Returns:
        Phase-label string: ``'STABLE'``, ``'TRANSITIONING'``, or
        ``'COLLAPSE_RISK'``.
    """
    return _entropy_to_phase_label(entropy, threshold)
