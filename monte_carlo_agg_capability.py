"""
Monte Carlo study: aggressiveness (Beta throw distances) vs capability
(exponential completion decay), using existing GameParams and TeamParam.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from GameParams import GameParams
from TeamParam import TeamParam

# Reproducibility: single stream for a simulation block (set before building teams).
_SIM_RNG: np.random.Generator | None = None
GLOBAL_SEED = 42


def set_simulation_rng(rng: np.random.Generator) -> None:
    """Bind the RNG used by ``make_distance_sampler`` / ``build_team`` (TeamParam's P has no rng arg)."""
    global _SIM_RNG
    _SIM_RNG = rng


def _rng() -> np.random.Generator:
    if _SIM_RNG is None:
        raise RuntimeError("Call set_simulation_rng(np.random.default_rng(...)) before build_team / make_distance_sampler.")
    return _SIM_RNG


def make_distance_sampler(agg: float) -> Callable[[float], float]:
    """
    z ~ Beta(alpha, beta), z in [0, 1]; d = z * d_max.

    alpha = 1 + 5 * agg, beta = 1 + 5 * (1 - agg), agg in [0, 1].

    Uses the RNG last passed to ``set_simulation_rng`` (shared with ``TeamParam.sample_throw``).
    """
    alpha = 1.0 + 5.0 * float(agg)
    beta = 1.0 + 5.0 * (1.0 - float(agg))

    def sample_d(d_max: float) -> float:
        z = float(_rng().beta(alpha, beta))
        return z * float(d_max)

    return sample_d


def completion_probability(beta_cap: float) -> Callable[[float, float], float]:
    """T(d) = exp(-beta_cap * d); smaller beta_cap => stronger team."""

    bc = float(beta_cap)

    def T(d: float, _d_max: float) -> float:
        return float(np.exp(-bc * float(d)))

    return T


def build_team(beta_cap: float, agg: float, attack_sign: int) -> TeamParam:
    """Continuous distance policy + exponential completion decay."""
    return TeamParam(
        T=completion_probability(beta_cap),
        P=make_distance_sampler(agg),
        attack_sign=int(attack_sign),
    )


@dataclass(frozen=True)
class PossessionSimStats:
    team_a_score: int
    team_b_score: int
    total_throws: int
    completions: int
    turnovers: int
    total_distance: float


def simulate_possessions(
    team_a: TeamParam,
    team_b: TeamParam,
    gp: GameParams,
    rng: np.random.Generator,
    n_possessions: int,
    max_plays: int = 10_000,
) -> PossessionSimStats:
    """
    Possession-based simulation consistent with samecapability_aggressivevsdiscrete.py:
    each outer index runs until a goal; incomplete throws switch offense (alternate on failure);
    score when a completed throw lands in the attacker's end zone.

    Tracks aggregate throws, completions, turnovers (incompletions), and total attempted distance.
    """
    team_a_score = 0
    team_b_score = 0
    total_throws = 0
    completions = 0
    turnovers = 0
    total_distance = 0.0

    offense = team_a
    x = float(rng.uniform(0.8 * gp.D_full, gp.D_full))

    for _ in range(n_possessions):
        for _ in range(max_plays):
            d, complete = offense.sample_throw(x, gp, rng)
            total_throws += 1
            total_distance += float(d)
            if complete:
                completions += 1
            else:
                turnovers += 1

            x = gp.position_after_throw(x, d, offense.attack_sign)
            if complete:
                if offense is team_a and gp.in_E_A(x):
                    team_a_score += 1
                    offense = team_b
                    x = float(rng.uniform(0, 0.2 * gp.D_full))
                    break
                if offense is team_b and gp.in_E_B(x):
                    team_b_score += 1
                    offense = team_a
                    x = float(rng.uniform(0.8 * gp.D_full, gp.D_full))
                    break
            else:
                offense = team_b if offense is team_a else team_a

    return PossessionSimStats(
        team_a_score=team_a_score,
        team_b_score=team_b_score,
        total_throws=total_throws,
        completions=completions,
        turnovers=turnovers,
        total_distance=total_distance,
    )


def run_sweep(
    gp: GameParams | None = None,
    beta_caps: tuple[float, ...] = (0.08, 0.05, 0.03),
    aggs: np.ndarray | None = None,
    n_possessions: int = 5_000,
    seed: int = GLOBAL_SEED,
) -> pd.DataFrame:
    """
    Full factorial over (beta_cap_A, agg_A, beta_cap_B, agg_B).
    Each row: one matchup simulation (n_possessions goal-ending rounds).
    """
    if gp is None:
        gp = GameParams()
    if aggs is None:
        aggs = np.linspace(0.0, 1.0, 11)

    rows: list[dict[str, float]] = []
    rng = np.random.default_rng(seed)
    set_simulation_rng(rng)

    for beta_cap_a in beta_caps:
        for agg_a in aggs:
            for beta_cap_b in beta_caps:
                for agg_b in aggs:
                    team_a = build_team(beta_cap_a, float(agg_a), 1)
                    team_b = build_team(beta_cap_b, float(agg_b), -1)
                    stats = simulate_possessions(team_a, team_b, gp, rng, n_possessions)

                    total_goals = stats.team_a_score + stats.team_b_score
                    win_rate_a = (
                        float(stats.team_a_score) / float(total_goals) if total_goals > 0 else float("nan")
                    )
                    avg_distance = (
                        float(stats.total_distance) / float(stats.total_throws)
                        if stats.total_throws > 0
                        else float("nan")
                    )
                    completion_rate = (
                        float(stats.completions) / float(stats.total_throws)
                        if stats.total_throws > 0
                        else float("nan")
                    )
                    turnover_rate = (
                        float(stats.turnovers) / float(stats.total_throws)
                        if stats.total_throws > 0
                        else float("nan")
                    )

                    rows.append(
                        {
                            "beta_cap_A": float(beta_cap_a),
                            "beta_cap_B": float(beta_cap_b),
                            "agg_A": float(agg_a),
                            "agg_B": float(agg_b),
                            "win_rate_A": win_rate_a,
                            "avg_distance": avg_distance,
                            "completion_rate": completion_rate,
                            "turnover_rate": turnover_rate,
                        }
                    )

    return pd.DataFrame(rows)


def main() -> pd.DataFrame:
    df = run_sweep(n_possessions=10)
    print(df.head(12).to_string(index=False))
    print(f"... ({len(df)} rows)")
    return df


if __name__ == "__main__":
    main()
