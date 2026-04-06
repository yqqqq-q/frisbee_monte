from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Tuple

import numpy as np

from GameParams import GameParams

# Callable types for injectable throw models (any compatible function may be supplied).
SampleD = Callable[[float], float]
CompletionFn = Callable[[float, float], float]


@dataclass
class TeamParam:
    """
    Per-team throw model: T(d, d_max) is completion probability; P(d_max) samples throw distance.
    Pass custom callables for T and P to define team-specific behavior. Value iteration still
    needs a discrete pdf on d_grid; pass the same T there and build P separately.
    """

    T: CompletionFn
    P: SampleD

    def sample_throw(
        self,
        position: float,
        prm: GameParams,
        rng: np.random.Generator,
    ) -> Tuple[float, bool]:
        """
        At field coordinate `position`, draw throw distance d ~ P and completion ~ Bernoulli(T(d, d_max)).

        Returns (d, complete) with d clamped to [0, prm.d_max]. P and T may ignore `position` today;
        keep it for state-dependent policies (closures or future hooks).
        """
        assert np.isfinite(position)
        dmax = prm.d_max
        d = float(self.P(dmax))
        d = float(max(0.0, min(d, dmax)))
        p_complete = float(min(max(float(self.T(d, dmax)), 0.0), 1.0))
        complete = bool(rng.random() < p_complete)
        return d, complete
