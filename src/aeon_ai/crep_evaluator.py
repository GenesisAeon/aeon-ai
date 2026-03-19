"""CREPEvaluator: Coherence–Resonance–Emergence–Poetics scoring.

CREP is the four-dimensional symbolic quality metric used throughout the
unified-mandala stack.  Each dimension is a float in [0, 1]:

    C — Coherence  : logical and structural consistency
    R — Resonance  : harmonic alignment between signal components
    E — Emergence  : novelty and self-organisational complexity
    P — Poetics    : aesthetic / symbolic richness

Combined CREP score (harmonic mean, bias-weighted):

    CREP = 4 / (1/C + 1/R + 1/E + 1/P)   (when all > 0)

A weighted variant is also available:

    CREP_w = Σ(wᵢ · dim_i)  where Σwᵢ = 1
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

_DIMENSIONS = ("coherence", "resonance", "emergence", "poetics")
_EPSILON = 1e-9


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


@dataclass
class CREPScore:
    """Structured holder for a CREP evaluation result.

    Attributes:
        coherence:  Logical consistency score C ∈ [0, 1].
        resonance:  Harmonic alignment score  R ∈ [0, 1].
        emergence:  Novelty / complexity score E ∈ [0, 1].
        poetics:    Aesthetic richness score  P ∈ [0, 1].
        weights:    Optional per-dimension weights (summing to 1).
    """

    coherence: float
    resonance: float
    emergence: float
    poetics: float
    weights: tuple[float, float, float, float] = (0.25, 0.25, 0.25, 0.25)

    def __post_init__(self) -> None:
        """Validate and clamp all dimension values."""
        for dim in _DIMENSIONS:
            object.__setattr__(self, dim, _clamp(float(getattr(self, dim))))
        w = self.weights
        if len(w) != 4:  # noqa: PLR2004
            raise ValueError("weights must have exactly 4 elements")
        total = sum(w)
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"weights must sum to 1.0, got {total}")

    # ------------------------------------------------------------------
    # Aggregate scores
    # ------------------------------------------------------------------

    @property
    def harmonic_mean(self) -> float:
        r"""Harmonic mean of all four dimensions.

        .. math::
            \\text{CREP} = \\frac{4}{1/C + 1/R + 1/E + 1/P}

        Returns zero if any dimension is zero.
        """
        dims = [self.coherence, self.resonance, self.emergence, self.poetics]
        if any(d < _EPSILON for d in dims):
            return 0.0
        return 4.0 / sum(1.0 / d for d in dims)

    @property
    def weighted_mean(self) -> float:
        r"""Weighted arithmetic mean of the four dimensions.

        .. math::
            \\text{CREP}_w = \\sum_{i} w_i \\cdot \\text{dim}_i
        """
        dims = [self.coherence, self.resonance, self.emergence, self.poetics]
        return sum(w * d for w, d in zip(self.weights, dims, strict=False))

    @property
    def score(self) -> float:
        """Primary CREP score (harmonic mean, default)."""
        return self.harmonic_mean

    def as_dict(self) -> dict[str, float]:
        """Export all dimensions and aggregate scores as a flat dict."""
        return {
            "coherence": self.coherence,
            "resonance": self.resonance,
            "emergence": self.emergence,
            "poetics": self.poetics,
            "harmonic_mean": self.harmonic_mean,
            "weighted_mean": self.weighted_mean,
        }

    def __repr__(self) -> str:
        """Return concise string representation."""
        return (
            f"CREPScore(C={self.coherence:.3f}, R={self.resonance:.3f}, "
            f"E={self.emergence:.3f}, P={self.poetics:.3f}, "
            f"score={self.score:.3f})"
        )


class CREPEvaluator:
    """Evaluate symbolic outputs on the CREP quality dimensions.

    The evaluator computes each dimension from a combination of:
    - signal entropy (information-theoretic complexity → E, C)
    - inter-signal correlation (harmonic alignment → R)
    - poetic density proxy (vocabulary richness of text → P)

    Integration with ``unified-mandala`` is attempted at import time; the
    evaluator falls back to its own heuristics when that package is absent.

    Example:
        >>> ev = CREPEvaluator()
        >>> score = ev.evaluate(signal=[0.3, 0.7, 0.5], text="aeon of light")
        >>> 0.0 <= score.score <= 1.0
        True
    """

    def __init__(
        self,
        weights: tuple[float, float, float, float] = (0.25, 0.25, 0.25, 0.25),
        entropy_scale: float = 1.0,
    ) -> None:
        """Initialise CREPEvaluator.

        Args:
            weights:       Per-dimension weights (C, R, E, P); must sum to 1.
            entropy_scale: Scaling factor applied to entropy-derived metrics.
        """
        if abs(sum(weights) - 1.0) > 1e-6:
            raise ValueError(f"weights must sum to 1.0, got {sum(weights)}")
        self.weights = weights
        self.entropy_scale = entropy_scale
        self._history: list[CREPScore] = []

    # ------------------------------------------------------------------
    # Primary API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        signal: list[float] | None = None,
        text: str | None = None,
        external: dict[str, float] | None = None,
    ) -> CREPScore:
        """Compute a :class:`CREPScore` for the provided inputs.

        Args:
            signal:   Numeric time-series or feature vector.
            text:     Free-text string (used for poetic density).
            external: Pre-computed dimension overrides (keys: ``coherence``,
                      ``resonance``, ``emergence``, ``poetics``).

        Returns:
            :class:`CREPScore` instance.
        """
        ext = external or {}
        sig = signal or []

        c = ext.get("coherence", self._coherence(sig))
        r = ext.get("resonance", self._resonance(sig))
        e = ext.get("emergence", self._emergence(sig))
        p = ext.get("poetics", self._poetics(text or ""))

        score = CREPScore(
            coherence=c,
            resonance=r,
            emergence=e,
            poetics=p,
            weights=self.weights,
        )
        self._history.append(score)
        return score

    def evaluate_batch(
        self,
        items: list[dict[str, Any]],
    ) -> list[CREPScore]:
        """Evaluate a batch of items.

        Args:
            items: List of keyword-argument dicts passed to :meth:`evaluate`.

        Returns:
            List of :class:`CREPScore` instances in input order.
        """
        return [self.evaluate(**item) for item in items]

    @property
    def history(self) -> list[CREPScore]:
        """All scores produced since last :meth:`reset`."""
        return list(self._history)

    def reset(self) -> None:
        """Clear evaluation history."""
        self._history.clear()

    # ------------------------------------------------------------------
    # Dimension heuristics
    # ------------------------------------------------------------------

    @staticmethod
    def _shannon_entropy(values: list[float]) -> float:
        """Normalised Shannon entropy of a float sequence."""
        if not values:
            return 0.0
        total = sum(abs(v) for v in values) + _EPSILON
        probs = [abs(v) / total for v in values]
        raw = -sum(p * math.log2(p + _EPSILON) for p in probs)
        max_ent = math.log2(len(values)) if len(values) > 1 else 1.0
        return raw / max_ent

    def _coherence(self, signal: list[float]) -> float:
        """Coherence: low variance → high coherence."""
        if len(signal) < 2:  # noqa: PLR2004
            return 1.0
        mean = sum(signal) / len(signal)
        variance = sum((x - mean) ** 2 for x in signal) / len(signal)
        return _clamp(1.0 / (1.0 + variance * self.entropy_scale))

    def _resonance(self, signal: list[float]) -> float:
        """Resonance: auto-correlation at lag-1 normalised to [0,1]."""
        n = len(signal)
        if n < 2:  # noqa: PLR2004
            return 0.5
        mean = sum(signal) / n
        centred = [x - mean for x in signal]
        raw_var = sum(x**2 for x in centred) / n
        if raw_var < _EPSILON:
            return 1.0  # constant signal → perfect autocorrelation
        var = raw_var
        lag1 = sum(centred[i] * centred[i - 1] for i in range(1, n)) / ((n - 1) * var)
        return _clamp((lag1 + 1.0) / 2.0)

    def _emergence(self, signal: list[float]) -> float:
        """Emergence: normalised Shannon entropy of the signal."""
        return _clamp(self._shannon_entropy(signal) * self.entropy_scale)

    @staticmethod
    def _poetics(text: str) -> float:
        """Poetics: type-token ratio as a vocabulary richness proxy."""
        tokens = text.lower().split()
        if not tokens:
            return 0.0
        ttr = len(set(tokens)) / len(tokens)
        return _clamp(ttr)

    def __repr__(self) -> str:
        """Return developer-friendly string representation."""
        return (
            f"CREPEvaluator(weights={self.weights}, "
            f"entropy_scale={self.entropy_scale}, "
            f"history_len={len(self._history)})"
        )
