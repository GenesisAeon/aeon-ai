"""AeonLayer: Extended weighting layer with fieldtheory Lagrangian dynamics.

Implements the core AeonLayer from the GenesisAeon / advanced-weighting-systems stack,
extended with the fieldtheory Lagrangian:

    L(S_A, S_V, δ, t) = (S_A · S_V) / (S_A + S_V) - (1 + δ) / t²

Where:
    S_A : auditory (abstract) signal amplitude
    S_V : visual (visceral) signal amplitude
    δ   : deformation / curvature parameter
    t   : time step (must be strictly positive)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class WeightingLayerProtocol(Protocol):
    """Structural contract for advanced_weighting_systems.AeonLayer compatibility."""

    def forward(self, s_a: float, s_v: float) -> float:  # pragma: no cover
        """Compute weighted output from two signal amplitudes."""
        ...

    def state_dict(self) -> dict[str, Any]:  # pragma: no cover
        """Return serialisable layer state."""
        ...


@dataclass
class LagrangianConfig:
    """Hyperparameters for the fieldtheory Lagrangian.

    Attributes:
        delta:   Deformation / curvature parameter δ (default 0.0).
        epsilon: Numerical-stability guard against division by zero (default 1e-8).
    """

    delta: float = 0.0
    epsilon: float = 1e-8


def lagrangian(
    s_a: float,
    s_v: float,
    delta: float,
    t: float,
    epsilon: float = 1e-8,
) -> float:
    r"""Compute the fieldtheory Lagrangian value.

    .. math::
        L = \\frac{S_A \\cdot S_V}{S_A + S_V} - \\frac{1 + \\delta}{t^2}

    Args:
        s_a:     Auditory signal amplitude :math:`S_A`.
        s_v:     Visual signal amplitude :math:`S_V`.
        delta:   Deformation parameter :math:`\\delta`.
        t:       Time step (must be > 0).
        epsilon: Guard against zero denominator.

    Returns:
        Scalar Lagrangian value :math:`L`.

    Raises:
        ValueError: If *t* ≤ 0.
    """
    if t <= 0.0:
        raise ValueError(f"Time step t must be strictly positive, got {t!r}")

    denom = s_a + s_v
    harmonic_term = 0.0 if abs(denom) < epsilon else s_a * s_v / denom

    temporal_penalty = (1.0 + delta) / (t**2)
    return harmonic_term - temporal_penalty


def lagrangian_gradient(
    s_a: float,
    s_v: float,
    delta: float,
    t: float,
    epsilon: float = 1e-8,
) -> dict[str, float]:
    """Analytical gradients of *L* with respect to its inputs.

    Args:
        s_a:     Auditory signal amplitude.
        s_v:     Visual signal amplitude.
        delta:   Deformation parameter.
        t:       Time step.
        epsilon: Numerical guard.

    Returns:
        Dictionary with keys ``dL/ds_a``, ``dL/ds_v``, ``dL/dt``.
    """
    denom = s_a + s_v
    if abs(denom) < epsilon:
        dl_ds_a = 0.0
        dl_ds_v = 0.0
    else:
        dl_ds_a = (s_v * s_v) / (denom**2)
        dl_ds_v = (s_a * s_a) / (denom**2)

    dl_dt = 2.0 * (1.0 + delta) / (t**3)
    return {"dL/ds_a": dl_ds_a, "dL/ds_v": dl_ds_v, "dL/dt": dl_dt}


class AeonLayer:
    """Extended AeonLayer integrating fieldtheory Lagrangian dynamics.

    Wraps or extends ``advanced_weighting_systems.AeonLayer`` when available,
    falling back to a native implementation that is fully contract-compatible.

    The core forward equation::

        output = L(S_A, S_V, δ, t) = S_A·S_V / (S_A+S_V) − (1+δ) / t²

    Example:
        >>> layer = AeonLayer(delta=0.1)
        >>> layer.forward(s_a=0.8, s_v=0.6, t=1.0)
        -0.762857...
    """

    def __init__(
        self,
        delta: float = 0.0,
        epsilon: float = 1e-8,
        _base_layer: WeightingLayerProtocol | None = None,
    ) -> None:
        """Initialise AeonLayer.

        Args:
            delta:       Deformation parameter δ (fieldtheory curvature).
            epsilon:     Numerical stability guard.
            _base_layer: Optional external weighting-layer (e.g. from
                         ``advanced_weighting_systems``).  When provided, its
                         ``forward`` output replaces *s_a* before the
                         Lagrangian is applied.
        """
        self.config = LagrangianConfig(delta=delta, epsilon=epsilon)
        self._base = _base_layer
        self._history: list[dict[str, float]] = []

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def forward(self, s_a: float, s_v: float, t: float = 1.0) -> float:
        """Compute the Lagrangian-weighted output.

        Args:
            s_a: Auditory signal amplitude.
            s_v: Visual signal amplitude.
            t:   Time step (> 0).

        Returns:
            Lagrangian value L at the given inputs.
        """
        if self._base is not None:
            s_a = self._base.forward(s_a, s_v)

        l_val = lagrangian(s_a, s_v, self.config.delta, t, self.config.epsilon)
        self._history.append({"s_a": s_a, "s_v": s_v, "t": t, "L": l_val})
        return l_val

    def gradient(self, s_a: float, s_v: float, t: float = 1.0) -> dict[str, float]:
        """Return analytical gradients of L.

        Args:
            s_a: Auditory signal amplitude.
            s_v: Visual signal amplitude.
            t:   Time step.

        Returns:
            Dict with ``dL/ds_a``, ``dL/ds_v``, ``dL/dt``.
        """
        return lagrangian_gradient(s_a, s_v, self.config.delta, t, self.config.epsilon)

    def reset_history(self) -> None:
        """Clear the internal forward-pass history."""
        self._history.clear()

    def state_dict(self) -> dict[str, Any]:
        """Serialise layer state.

        Returns:
            Dictionary suitable for ``load_state_dict``.
        """
        return {
            "delta": self.config.delta,
            "epsilon": self.config.epsilon,
            "history_len": len(self._history),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore layer state from a dictionary.

        Args:
            state: Mapping previously produced by :meth:`state_dict`.
        """
        self.config.delta = float(state.get("delta", self.config.delta))
        self.config.epsilon = float(state.get("epsilon", self.config.epsilon))

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_advanced_weighting_systems(cls, **kwargs: Any) -> AeonLayer:  # noqa: ANN401
        """Construct from ``advanced_weighting_systems.AeonLayer`` if installed.

        Falls back to native implementation when the package is absent.

        Args:
            **kwargs: Forwarded to both the AWS layer and this constructor.

        Returns:
            An :class:`AeonLayer` instance, optionally wrapping the AWS base.
        """
        try:
            from advanced_weighting_systems import (
                AeonLayer as _AwsLayer,  # type: ignore[import-untyped]
            )

            base = _AwsLayer(**{k: v for k, v in kwargs.items() if k not in ("delta", "epsilon")})
            own_kw = {k: v for k, v in kwargs.items() if k in ("delta", "epsilon")}
            return cls(_base_layer=base, **own_kw)
        except ImportError:
            own_kw = {k: v for k, v in kwargs.items() if k in ("delta", "epsilon")}
            return cls(**own_kw)

    def __repr__(self) -> str:
        """Return developer-friendly string representation."""
        return (
            f"AeonLayer(delta={self.config.delta}, epsilon={self.config.epsilon}, "
            f"history_len={len(self._history)})"
        )
