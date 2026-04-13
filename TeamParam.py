from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from typing import Callable, List, Sequence, Tuple, Union

import numpy as np

from GameParams import GameParams

# Callable types for injectable throw models (any compatible function may be supplied).
SampleD = Callable[[float], float]
CompletionFn = Callable[[float, float], float]
# Discrete throw model: (distance, weight); weights are normalized to probabilities.
DistanceWeight = Tuple[float, float]

TSpec = Union[CompletionFn, Sequence[CompletionFn], Sequence[float]]
PSpec = Union[SampleD, Sequence[DistanceWeight]]


def _is_discrete_p(p: object) -> bool:
    if callable(p) or isinstance(p, (str, bytes)):
        return False
    try:
        pairs = list(p)  # type: ignore[arg-type]
    except TypeError:
        return False
    if len(pairs) == 0:
        return False
    first = pairs[0]
    try:
        return len(first) == 2  # type: ignore[arg-type]
    except TypeError:
        return False


def _is_t_list(t: object) -> bool:
    if callable(t):
        return False
    try:
        fns = list(t)  # type: ignore[arg-type]
    except TypeError:
        return False
    return len(fns) > 0 and all(callable(f) for f in fns)


def _is_t_probs(t: object) -> bool:
    """True if T is a non-empty sequence of numeric completion probabilities (not callables)."""
    if callable(t):
        return False
    try:
        xs = list(t)  # type: ignore[arg-type]
    except TypeError:
        return False
    if len(xs) == 0:
        return False
    for x in xs:
        if isinstance(x, bool) or not isinstance(x, Real):
            return False
    return True


@dataclass
class TeamParam:
    """
    Per-team throw model.

    P (distance):
      - Callable ``d_max -> d``: sample a throw distance (existing behavior).
      - List of ``(distance, weight)``: discrete outcomes; weights are normalized to
        probabilities and one row is sampled per throw.

    T (completion probability):
      - Callable ``(d, d_max) -> p`` in [0, 1].
      - List of callables with the same indexing/binning rules as below.
      - List of numeric probabilities in [0, 1] (constants per bucket):
        * If P is discrete, len(T) must equal len(P); outcome i uses T[i].
        * If P is a callable, d is binned into len(T) equal subintervals of [0, d_max]
          and T[k] is used.

    attack_sign: +1 advances the disc toward E_A (high x); -1 toward E_B (low x).
    """

    T: TSpec
    P: PSpec
    attack_sign: int = 1

    def sample_throw(
        self,
        position: float,
        prm: GameParams,
        rng: np.random.Generator,
    ) -> Tuple[float, bool]:
        """
        At field coordinate `position`, draw throw distance from P and completion from T.

        Returns (d, complete) with d clamped to [0, prm.d_max]. P and T may ignore `position`
        today; keep it for state-dependent policies (closures or future hooks).
        """
        assert np.isfinite(position)
        dmax = prm.d_max

        if isinstance(self.P, (list, tuple)) and len(self.P) == 0:
            raise ValueError("P cannot be an empty list")
        if isinstance(self.T, (list, tuple)) and len(self.T) == 0:
            raise ValueError("T cannot be an empty list")

        discrete_idx: int | None = None
        if _is_discrete_p(self.P):
            pairs: List[Tuple[float, float]] = [tuple(float(x) for x in row) for row in self.P]  # type: ignore[union-attr]
            dists = np.array([a for a, _ in pairs], dtype=float)
            w = np.array([b for _, b in pairs], dtype=float)
            sw = float(w.sum())
            if sw <= 0:
                raise ValueError("P as a list of (distance, weight): weights must sum to a positive value")
            probs = w / sw
            discrete_idx = int(rng.choice(len(pairs), p=probs))
            d = float(max(0.0, min(dists[discrete_idx], dmax)))
        else:
            d = float(self.P(dmax))  # type: ignore[operator, call-arg]
            d = float(max(0.0, min(d, dmax)))

        if _is_t_list(self.T):
            fns = list(self.T)  # type: ignore[arg-type]
            if discrete_idx is not None:
                if len(fns) != len(self.P):  # type: ignore[arg-type]
                    raise ValueError(
                        "When P is a list of (distance, weight) and T is a list of functions, "
                        "len(T) must equal len(P)"
                    )
                t_fn = fns[discrete_idx]
            else:
                if dmax <= 0:
                    k = 0
                else:
                    k = min(int(d / dmax * len(fns)), len(fns) - 1)
                t_fn = fns[k]
            p_complete = float(t_fn(d, dmax))
        elif _is_t_probs(self.T):
            probs = [float(x) for x in self.T]  # type: ignore[union-attr]
            if discrete_idx is not None:
                if len(probs) != len(self.P):  # type: ignore[arg-type]
                    raise ValueError(
                        "When P is a list of (distance, weight) and T is a list of probabilities, "
                        "len(T) must equal len(P)"
                    )
                p_complete = probs[discrete_idx]
            else:
                if dmax <= 0:
                    k = 0
                else:
                    k = min(int(d / dmax * len(probs)), len(probs) - 1)
                p_complete = probs[k]
        else:
            p_complete = float(self.T(d, dmax))  # type: ignore[operator, call-arg]

        p_complete = float(min(max(p_complete, 0.0), 1.0))
        complete = bool(rng.random() < p_complete)
        return d, complete
