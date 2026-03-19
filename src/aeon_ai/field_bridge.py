"""FieldBridge: Cosmic-moment detection and medium modulation from fieldtheory.

The field bridge connects aeon-ai to the fieldtheory cosmological layer,
providing:

    CosmicMoment  — a snapshot of the current field state (entropy, tension,
                    coherence, resonance) at a given time coordinate.
    MediumMode    — enumeration of field propagation media.
    FieldBridge   — modulates AeonLayer and MirrorCore outputs by the cosmic
                    field state.

Integration with the ``fieldtheory`` and ``cosmic-web`` packages is attempted
at import time; a standalone implementation is used as fallback.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any


class MediumMode(Enum):
    """Field propagation medium modes.

    Attributes:
        VACUUM:    Free-field propagation (no medium).
        AETHERIC:  Low-density symbolic medium.
        RESONANT:  High-coherence harmonic medium.
        DENSE:     High-entropy saturated medium.
    """

    VACUUM = "vacuum"
    AETHERIC = "aetheric"
    RESONANT = "resonant"
    DENSE = "dense"


@dataclass
class CosmicMoment:
    """Snapshot of the fieldtheory cosmic field state.

    Attributes:
        timestamp:  Unix time of the snapshot.
        entropy:    Field entropy ∈ [0, 1].
        tension:    Topological tension (gradient magnitude proxy) ≥ 0.
        coherence:  Field coherence ∈ [0, 1].
        resonance:  Resonance amplitude ∈ [0, 1].
        medium:     Dominant :class:`MediumMode`.
        metadata:   Arbitrary extra information.
    """

    timestamp: float
    entropy: float
    tension: float
    coherence: float
    resonance: float
    medium: MediumMode
    metadata: dict[str, Any]

    @property
    def field_strength(self) -> float:
        """Composite field strength F = coherence · resonance / (1 + tension).

        Returns:
            Positive real-valued field strength.
        """
        return (self.coherence * self.resonance) / (1.0 + self.tension)

    def as_dict(self) -> dict[str, Any]:
        """Export the moment as a flat dictionary.

        Returns:
            Dictionary representation including ``field_strength``.
        """
        return {
            "timestamp": self.timestamp,
            "entropy": self.entropy,
            "tension": self.tension,
            "coherence": self.coherence,
            "resonance": self.resonance,
            "medium": self.medium.value,
            "field_strength": self.field_strength,
            "metadata": self.metadata,
        }


def _medium_from_entropy(entropy: float) -> MediumMode:
    """Infer :class:`MediumMode` from entropy value.

    Args:
        entropy: Field entropy ∈ [0, 1].

    Returns:
        Appropriate :class:`MediumMode` enum member.
    """
    if entropy < 0.2:  # noqa: PLR2004
        return MediumMode.VACUUM
    if entropy < 0.45:  # noqa: PLR2004
        return MediumMode.AETHERIC
    if entropy < 0.7:  # noqa: PLR2004
        return MediumMode.RESONANT
    return MediumMode.DENSE


class FieldBridge:
    """Modulate aeon-ai components using fieldtheory cosmic-moment state.

    The bridge samples a :class:`CosmicMoment` (from the ``fieldtheory``
    package or its own oscillator) and exposes modulation methods for:
    - Lagrangian delta adjustment
    - MirrorCore entropy injection
    - CREP dimension scaling

    Example:
        >>> bridge = FieldBridge(base_entropy=0.4)
        >>> moment = bridge.sample_moment()
        >>> isinstance(moment, CosmicMoment)
        True
        >>> factor = bridge.modulation_factor(moment)
        >>> 0.0 < factor <= 2.0
        True
    """

    def __init__(
        self,
        base_entropy: float = 0.3,
        tension_frequency: float = 0.1,
        coherence_base: float = 0.8,
        resonance_base: float = 0.75,
    ) -> None:
        """Initialise FieldBridge.

        Args:
            base_entropy:       Baseline field entropy ∈ [0, 1].
            tension_frequency:  Angular frequency of the tension oscillator (rad/s).
            coherence_base:     Static coherence level ∈ [0, 1].
            resonance_base:     Static resonance level ∈ [0, 1].
        """
        if not 0.0 <= base_entropy <= 1.0:
            raise ValueError(f"base_entropy must be in [0, 1], got {base_entropy!r}")
        if not 0.0 <= coherence_base <= 1.0:
            raise ValueError(f"coherence_base must be in [0, 1], got {coherence_base!r}")
        if not 0.0 <= resonance_base <= 1.0:
            raise ValueError(f"resonance_base must be in [0, 1], got {resonance_base!r}")

        self.base_entropy = base_entropy
        self.tension_frequency = tension_frequency
        self.coherence_base = coherence_base
        self.resonance_base = resonance_base
        self._moments: list[CosmicMoment] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def sample_moment(self, t: float | None = None) -> CosmicMoment:
        """Sample the current cosmic field moment.

        Attempts to delegate to the ``fieldtheory`` package; falls back to
        the internal oscillator.

        Args:
            t: Optional time coordinate (Unix seconds).  Defaults to ``time.time()``.

        Returns:
            :class:`CosmicMoment` snapshot.
        """
        if t is None:
            t = time.time()

        moment = self._try_fieldtheory(t)
        if moment is None:
            moment = self._oscillator_moment(t)

        self._moments.append(moment)
        return moment

    def modulation_factor(self, moment: CosmicMoment) -> float:
        r"""Compute a scalar modulation factor from a :class:`CosmicMoment`.

        The factor scales outputs of AeonLayer / MirrorCore:

        .. math::
            M = F \\cdot (1 - H/2)

        where *F* is the field strength and *H* is the entropy.

        Args:
            moment: Field snapshot.

        Returns:
            Modulation factor in (0, 2].
        """
        return moment.field_strength * (1.0 - moment.entropy / 2.0)

    def adjust_delta(self, base_delta: float, moment: CosmicMoment) -> float:
        """Adjust Lagrangian δ using the cosmic moment's tension.

        Args:
            base_delta: Base deformation parameter.
            moment:     Current cosmic moment.

        Returns:
            Adjusted δ = base_delta + tension · entropy.
        """
        return base_delta + moment.tension * moment.entropy

    def inject_entropy(self, moment: CosmicMoment) -> float:
        """Return an entropy value suitable for :meth:`MirrorCore.reflect`.

        Args:
            moment: Current cosmic moment.

        Returns:
            Entropy value derived from field state.
        """
        return moment.entropy

    @property
    def moment_history(self) -> list[CosmicMoment]:
        """All moments sampled since last :meth:`reset`."""
        return list(self._moments)

    def reset(self) -> None:
        """Clear moment history."""
        self._moments.clear()

    # ------------------------------------------------------------------
    # Internal oscillator (standalone fallback)
    # ------------------------------------------------------------------

    def _oscillator_moment(self, t: float) -> CosmicMoment:
        """Generate a synthetic CosmicMoment via a harmonic oscillator.

        Args:
            t: Time coordinate.

        Returns:
            Synthesised :class:`CosmicMoment`.
        """
        # Oscillating tension: T(t) = 0.5·|sin(ω·t)|
        tension = 0.5 * abs(math.sin(self.tension_frequency * t))
        # Entropy drifts slowly with time
        entropy = self.base_entropy + 0.1 * math.sin(0.01 * t)
        entropy = max(0.0, min(1.0, entropy))
        coherence = self.coherence_base * (1.0 - 0.2 * tension)
        resonance = self.resonance_base * (1.0 + 0.1 * math.cos(self.tension_frequency * t))
        resonance = max(0.0, min(1.0, resonance))

        return CosmicMoment(
            timestamp=t,
            entropy=entropy,
            tension=tension,
            coherence=coherence,
            resonance=resonance,
            medium=_medium_from_entropy(entropy),
            metadata={"source": "oscillator"},
        )

    def _try_fieldtheory(self, t: float) -> CosmicMoment | None:
        """Try to sample from the ``fieldtheory`` package.

        Args:
            t: Time coordinate.

        Returns:
            :class:`CosmicMoment` if ``fieldtheory`` is available, else ``None``.
        """
        try:
            import fieldtheory  # type: ignore[import-untyped]

            raw = fieldtheory.cosmic_moment(t=t)
            return CosmicMoment(
                timestamp=raw.get("timestamp", t),
                entropy=float(raw.get("entropy", self.base_entropy)),
                tension=float(raw.get("tension", 0.0)),
                coherence=float(raw.get("coherence", self.coherence_base)),
                resonance=float(raw.get("resonance", self.resonance_base)),
                medium=MediumMode(raw.get("medium", MediumMode.AETHERIC.value)),
                metadata=raw.get("metadata", {"source": "fieldtheory"}),
            )
        except (ImportError, Exception):  # noqa: BLE001
            return None

    def __repr__(self) -> str:
        """Return developer-friendly string representation."""
        return (
            f"FieldBridge(base_entropy={self.base_entropy}, "
            f"tension_freq={self.tension_frequency}, "
            f"moments_sampled={len(self._moments)})"
        )
