"""SigillinBridge: Symbolic sigil activation with poetic trigger semantics.

Sigils are named symbolic anchors that carry:
    - A compressed intention (``intent``)
    - A set of poetic trigger phrases (``triggers``)
    - An activation weight (``weight`` ∈ [0, 1])
    - Optional metadata (phase, version, context)

The bridge resolves plain text against the known sigil registry and fires
matching triggers, returning an activation vector.

Integration with the ``sigillin`` and ``aeon`` packages is attempted at
import time; the bridge operates standalone when those packages are absent.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Sigil:
    """A symbolic sigil with poetic triggers.

    Attributes:
        id:       Unique identifier (ASCII, uppercase).
        name:     Human-readable name.
        intent:   Compressed symbolic intention.
        triggers: List of trigger phrases or regex patterns.
        weight:   Activation weight ∈ [0, 1].
        phase:    Optional CREP phase association (C / R / E / P).
        metadata: Arbitrary context key-value pairs.
    """

    id: str
    name: str
    intent: str
    triggers: list[str] = field(default_factory=list)
    weight: float = 1.0
    phase: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate fields on construction."""
        if not self.id:
            raise ValueError("Sigil id must not be empty")
        if not 0.0 <= self.weight <= 1.0:
            raise ValueError(f"weight must be in [0, 1], got {self.weight!r}")

    def matches(self, text: str, case_sensitive: bool = False) -> bool:
        """Test whether *text* activates at least one trigger.

        Args:
            text:           Input string to test.
            case_sensitive: Use case-sensitive matching (default False).

        Returns:
            ``True`` if any trigger pattern matches *text*.
        """
        flags = 0 if case_sensitive else re.IGNORECASE
        return any(re.search(pattern, text, flags) for pattern in self.triggers)

    def activation_score(self, text: str) -> float:
        """Compute activation score for *text*.

        Counts the number of distinct triggers that match and normalises
        by the total trigger count, then scales by ``weight``.

        Args:
            text: Input string.

        Returns:
            Activation score ∈ [0, ``weight``].
        """
        if not self.triggers:
            return 0.0
        hits = sum(1 for p in self.triggers if re.search(p, text, re.IGNORECASE))
        return self.weight * (hits / len(self.triggers))


# ---------------------------------------------------------------------------
# Built-in genesis sigils (from the GenesisAeon / aeon repository)
# ---------------------------------------------------------------------------

_GENESIS_SIGILS: list[Sigil] = [
    Sigil(
        id="GENESIS",
        name="Genesis Anchor",
        intent="Origin and first-cause initiation",
        triggers=[r"\bgenesis\b", r"\borigin\b", r"\binitiat\w*\b", r"\bseed\b"],
        weight=1.0,
        phase="C",
    ),
    Sigil(
        id="HEIMKEHR",
        name="Homecoming",
        intent="Return, completion, and integration",
        triggers=[r"\bheiml\w*\b", r"\breturn\b", r"\bintegrat\w*\b", r"\bcomplet\w*\b"],
        weight=0.9,
        phase="E",
    ),
    Sigil(
        id="FRAKTALURS",
        name="Fractal Memory",
        intent="Recursive self-reference and memory persistence",
        triggers=[r"\bfraktal\w*\b", r"\brecursiv\w*\b", r"\bmemory\b", r"\bfractal\b"],
        weight=0.85,
        phase="R",
    ),
    Sigil(
        id="RESO_ECHO",
        name="Resonance Echo",
        intent="Harmonic persistence and symbolic reverberation",
        triggers=[r"\breso\w*\b", r"\becho\b", r"\bvibrat\w*\b", r"\bharmoni\w*\b"],
        weight=0.8,
        phase="R",
    ),
    Sigil(
        id="MIRROR",
        name="Mirror Gate",
        intent="Self-reflection and inversion",
        triggers=[r"\bmirror\b", r"\breflect\w*\b", r"\binvers\w*\b", r"\bself.refle\w*\b"],
        weight=1.0,
        phase="P",
    ),
    Sigil(
        id="AEON",
        name="Aeon",
        intent="Timeless symbolic intelligence and becoming",
        triggers=[r"\baeon\b", r"\btimeless\b", r"\beternal\w*\b", r"\bbecom\w*\b"],
        weight=1.0,
        phase="E",
    ),
]


class SigillinBridge:
    """Registry and activation engine for symbolic sigils.

    Resolves text against registered sigils, returns activation vectors,
    and integrates with the ``sigillin`` and ``aeon`` packages when available.

    Example:
        >>> bridge = SigillinBridge()
        >>> activations = bridge.activate("the mirror reflects the aeon")
        >>> "MIRROR" in activations
        True
    """

    def __init__(self, load_defaults: bool = True) -> None:
        """Initialise SigillinBridge.

        Args:
            load_defaults: Pre-load the built-in GenesisAeon sigils (default True).
        """
        self._registry: dict[str, Sigil] = {}
        if load_defaults:
            for sigil in _GENESIS_SIGILS:
                self.register(sigil)
        self._try_load_external()

    # ------------------------------------------------------------------
    # Registry management
    # ------------------------------------------------------------------

    def register(self, sigil: Sigil) -> None:
        """Register a :class:`Sigil` in the local registry.

        Args:
            sigil: Sigil to register.  Overwrites existing entry with same id.
        """
        self._registry[sigil.id] = sigil

    def unregister(self, sigil_id: str) -> None:
        """Remove a sigil from the registry.

        Args:
            sigil_id: Id of the sigil to remove.

        Raises:
            KeyError: If *sigil_id* is not registered.
        """
        if sigil_id not in self._registry:
            raise KeyError(f"Sigil '{sigil_id}' not found in registry")
        del self._registry[sigil_id]

    @property
    def sigils(self) -> dict[str, Sigil]:
        """Read-only view of the registry."""
        return dict(self._registry)

    # ------------------------------------------------------------------
    # Activation
    # ------------------------------------------------------------------

    def activate(self, text: str) -> dict[str, float]:
        """Compute activation scores for *text* across all registered sigils.

        Args:
            text: Input string to test against all triggers.

        Returns:
            Dict mapping sigil id → activation score (only non-zero entries).
        """
        result: dict[str, float] = {}
        for sid, sigil in self._registry.items():
            score = sigil.activation_score(text)
            if score > 0.0:
                result[sid] = score
        return result

    def top_sigil(self, text: str) -> Sigil | None:
        """Return the sigil with the highest activation for *text*.

        Args:
            text: Input string.

        Returns:
            Highest-scoring :class:`Sigil`, or ``None`` if no trigger fires.
        """
        activations = self.activate(text)
        if not activations:
            return None
        best_id = max(activations, key=lambda k: activations[k])
        return self._registry[best_id]

    def poetic_expansion(self, text: str) -> str:
        """Return a poetic expansion of *text* guided by active sigils.

        Appends the intents of all activated sigils as a symbolic suffix.

        Args:
            text: Input string.

        Returns:
            Expanded string with sigil intents woven in.
        """
        activations = self.activate(text)
        if not activations:
            return text
        intents = " · ".join(
            self._registry[sid].intent
            for sid in sorted(activations, key=lambda k: -activations[k])
        )
        return f"{text} ⟨{intents}⟩"

    # ------------------------------------------------------------------
    # External integration
    # ------------------------------------------------------------------

    def _try_load_external(self) -> None:
        """Attempt to load sigils from installed ``sigillin`` package."""
        try:
            import sigillin  # type: ignore[import-untyped]

            for raw in sigillin.registry():
                sigil = Sigil(
                    id=raw["id"],
                    name=raw.get("name", raw["id"]),
                    intent=raw.get("description", ""),
                    triggers=raw.get("triggers", []),
                    weight=float(raw.get("weight", 1.0)),
                    phase=raw.get("phase"),
                    metadata=raw.get("metadata", {}),
                )
                self.register(sigil)
        except (ImportError, Exception):  # noqa: BLE001
            pass  # external package absent — proceed with defaults

    def __repr__(self) -> str:
        """Return developer-friendly string representation."""
        return f"SigillinBridge(sigils={list(self._registry.keys())})"
