"""
Monte Carlo study: aggressiveness (Beta throw distances) vs capability
(exponential completion decay), using existing GameParams and TeamParam.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import lgamma
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from GameParams import GameParams
from TeamParam import TeamParam

# Reproducibility: single stream for a simulation block (set before building teams).
_SIM_RNG: np.random.Generator | None = None
GLOBAL_SEED = 42
TP_INPUT_MIN = -10.0
TP_INPUT_MAX_FRAC_DMAX = 0.65


def _series_cmap_color(index: int, n_series: int, cmap_name: str) -> tuple[float, float, float, float]:
    """Series color sampled evenly along a named matplotlib colormap."""
    cmap = plt.get_cmap(cmap_name)
    if n_series <= 0:
        return tuple(float(c) for c in cmap(0.5))
    t = (index + 0.5) / float(n_series)
    return tuple(float(c) for c in cmap(t))


def _gist_heat_color(index: int, n_series: int) -> tuple[float, float, float, float]:
    """Win-rate curves: ``gist_heat``."""
    return _series_cmap_color(index, n_series, "gist_heat")


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
    z ~ Beta(alpha, beta), z in [0, 1];
    d = TP_INPUT_MIN + z * (TP_INPUT_MAX_FRAC_DMAX * d_max - TP_INPUT_MIN).

    alpha = 1 + 5 * agg, beta = 1 + 5 * (1 - agg), agg in [0, 1].

    Uses the RNG last passed to ``set_simulation_rng`` (shared with ``TeamParam.sample_throw``).
    """
    alpha = 1.0 + 5.0 * float(agg)
    beta = 1.0 + 5.0 * (1.0 - float(agg))

    def sample_d(d_max: float) -> float:
        z = float(_rng().beta(alpha, beta))
        d_hi = TP_INPUT_MAX_FRAC_DMAX * float(d_max)
        return TP_INPUT_MIN + z * (d_hi - TP_INPUT_MIN)

    return sample_d


def completion_probability(beta_cap: float):
    """
    Build distance-based completion probability with capability in [0, 1].
    - beta_cap = 1.0 is strongest.
    - Smaller beta_cap means weaker completion at a given distance.
    - For beta_cap = 1.0:
        T(TP_INPUT_MIN) = 1.0
        T(TP_INPUT_MAX_FRAC_DMAX * d_max) = 0.7
    """
    beta_cap = float(np.clip(beta_cap, 0.0, 1.0))
    # 0 < gamma < 1 gives a concave-down shape on the normalized interval.
    gamma = 0.5

    def T(d: float, d_max: float) -> float:
        d_lo = TP_INPUT_MIN
        d_hi = TP_INPUT_MAX_FRAC_DMAX * float(d_max)
        span = max(d_hi - d_lo, 1e-9)
        x = np.clip((float(d) - d_lo) / span, 0.0, 1.0)

        # Strongest team (beta=1) drops to 0.7 at x=1.
        # Weaker teams (smaller beta) have larger drop across distance.
        drop_amplitude = 1.0 - 0.7 * beta_cap
        p = 1.0 - drop_amplitude * (x ** gamma)
        return float(np.clip(p, 0.0, 1.0))

    return T

def build_team(beta_cap: float, agg: float, attack_sign: int) -> TeamParam:
    """Continuous distance policy + quadratic completion decay."""
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
    verbose: bool = False,
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
    if verbose:
        print(
            f"Simulated {n_possessions} possessions, {total_throws} throws, {completions} completions, "
            f"{turnovers} turnovers, total distance {total_distance:.1f}"
        )
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
                    stats = simulate_possessions(team_a, team_b, gp, rng, n_possessions, verbose=False)

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


def _beta_pdf(z: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    """Numerically stable Beta(alpha, beta) density on z in (0, 1)."""
    z_safe = np.clip(z, 1e-9, 1.0 - 1e-9)
    log_norm = lgamma(alpha) + lgamma(beta) - lgamma(alpha + beta)
    log_pdf = (alpha - 1.0) * np.log(z_safe) + (beta - 1.0) * np.log(1.0 - z_safe) - log_norm
    return np.exp(log_pdf)


def plot_model_functions(
    gp: GameParams,
    out_dir: Path,
    aggs_for_plot: tuple[float, ...] = (0.1, 0.3, 0.5, 0.7, 0.9),
    beta_caps: tuple[float, ...] = (0.08, 0.05, 0.03),
) -> None:
    """
    Plot the underlying policy/completion functions for selected aggs and beta-caps.
    """
    z = np.linspace(0.001, 0.999, 500)
    d_max_hi = TP_INPUT_MAX_FRAC_DMAX * gp.d_max
    d = np.linspace(TP_INPUT_MIN, d_max_hi, 500)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    n_agg = len(aggs_for_plot)
    for i, agg in enumerate(aggs_for_plot):
        alpha = 1.0 + 5.0 * float(agg)
        beta = 1.0 + 5.0 * (1.0 - float(agg))
        pdf = _beta_pdf(z, alpha, beta)
        ax1.plot(
            TP_INPUT_MIN + z * (d_max_hi - TP_INPUT_MIN),
            pdf,
            color=_series_cmap_color(i, n_agg, "spring"),
            label=f"agg={agg:.1f}",
        )

    ax1.set_title("Throw Distance Distribution")
    ax1.set_xlabel("Throw distance d")
    ax1.set_ylabel("Density")
    ax1.grid(alpha=0.3)
    ax1.legend(loc="upper right", fontsize=8, framealpha=0.95)

    n_beta = len(beta_caps)
    for i, beta_cap in enumerate(beta_caps):
        T = completion_probability(float(beta_cap))
        p = np.array([T(di, gp.d_max) for di in d], dtype=float)
        ax2.plot(d, p, color=_series_cmap_color(i, n_beta, "summer"), label=f"beta_cap={beta_cap:.2f}")

    ax2.set_title("Completion Function")
    ax2.set_xlabel("Throw distance d")
    ax2.set_ylabel("P(complete)")
    ax2.set_ylim(0.0, 1.02)
    ax2.grid(alpha=0.3)
    ax2.legend(loc="lower right", fontsize=8, framealpha=0.95)

    fig.tight_layout()
    fig.savefig(out_dir / "model_functions_aggs_beta_caps.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_win_rate_vs_aggressiveness_by_capability_gap(
    gp: GameParams,
    out_dir: Path,
    aggs: np.ndarray,
    capability_gaps: tuple[float, ...] = (1.0, 1.2, 1.4, 1.6, 1.8),
    team_a_capability: float = 0.2,
    team_a_agg: float = 0.4,
    n_possessions: int = 5_000,
    seed: int = GLOBAL_SEED,
) -> pd.DataFrame:
    """
    Team B decision plot: how aggressiveness affects *your* win rate vs a fixed Team A.

    Team A is fixed at ``team_a_capability`` skill and ``team_a_agg`` aggressiveness.
    Team B skill is ``capability_gap * team_a_capability`` (clipped to [0, 1]); interpret
    ``capability_gap`` as how much of Team A's skill you have (so Team A is stronger
    when this fraction is < 1, and equal when it is 1).

    The x-axis sweeps *your* (Team B) aggressiveness over ``aggs``.
    """
    rng = np.random.default_rng(seed)
    set_simulation_rng(rng)
    rows: list[dict[str, float]] = []
    agg_a = float(np.clip(team_a_agg, 0.0, 1.0))
    beta_cap_a_fixed = float(np.clip(team_a_capability, 0.0, 1.0))

    for gap in capability_gaps:
        beta_cap_a = beta_cap_a_fixed
        beta_cap_b = float(np.clip(float(gap) * beta_cap_a, 0.0, 1.0))

        for agg_b in aggs:
            agg_b = float(agg_b)
            team_a = build_team(beta_cap_a, agg_a, 1)
            team_b = build_team(beta_cap_b, agg_b, -1)
            stats = simulate_possessions(team_a, team_b, gp, rng, n_possessions, verbose=False)
            total_goals = stats.team_a_score + stats.team_b_score
            win_rate_b = (
                float(stats.team_b_score) / float(total_goals) if total_goals > 0 else float("nan")
            )
            win_rate_a = (
                float(stats.team_a_score) / float(total_goals) if total_goals > 0 else float("nan")
            )
            skill_mult_a_vs_b = float("inf") if beta_cap_b <= 0.0 else float(beta_cap_a / beta_cap_b)
            rows.append(
                {
                    "agg_A": agg_a,
                    "agg_B": agg_b,
                    "capability_gap": float(gap),
                    "beta_cap_A": beta_cap_a,
                    "beta_cap_B": beta_cap_b,
                    "skill_mult_A_vs_B": skill_mult_a_vs_b,
                    "win_rate_B": win_rate_b,
                    "win_rate_A": win_rate_a,
                }
            )

    result_df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(13, 5.2))
    n_gaps = len(capability_gaps)
    for i, gap in enumerate(capability_gaps):
        sub = result_df[result_df["capability_gap"] == float(gap)].sort_values("agg_B")
        beta_a = float(sub["beta_cap_A"].iloc[0])
        beta_b = float(sub["beta_cap_B"].iloc[0])
        if beta_b <= 0.0:
            label = "You are far weaker than Team A"
        elif np.isclose(beta_a, beta_b):
            label = "Even skill vs Team A"
        else:
            mult = beta_a / beta_b
            if mult > 1.0:
                label = f"Team A is {mult:.2f}× stronger (Team B Skill={beta_b:.2f})"
            else:
                label = f"You are {1.0 / mult:.2f}× stronger (Team B Skill={beta_b:.2f})"
        ax.plot(
            sub["agg_B"],
            sub["win_rate_B"],
            marker="o",
            linewidth=1.8,
            color=_gist_heat_color(i, n_gaps),
            label=label,
        )

    ax.set_xlabel("Team B aggressiveness")
    ax.set_ylabel("Team B win rate")
    ax.set_title(
        "Win Rate vs Aggressiveness\n"
        f"(Team A: skill={beta_cap_a_fixed:.2f}, aggressiveness={agg_a:.2f})"
    )
    ys = result_df["win_rate_B"].to_numpy(dtype=float)
    mask = np.isfinite(ys)
    if not np.any(mask):
        ax.set_ylim(0.0, 1.0)
    else:
        y_lo, y_hi = float(np.min(ys[mask])), float(np.max(ys[mask]))
        span = y_hi - y_lo
        pad = max(1e-3, 0.08 * span) if span > 0 else 0.03
        ax.set_ylim(max(0.0, y_lo - pad), min(1.0, y_hi + pad))
    ax.grid(alpha=0.3)
    legend = ax.legend(
        # loc = 'left',
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        fontsize=8,
        framealpha=0.95,
    )
    fig.tight_layout()
    fig.savefig(
        out_dir / f"win_rate_vs_aggressiveness_capability_gaps_team_a_{beta_cap_a_fixed:.2f}_agg_{agg_a:.2f}.png",
        dpi=180,
        bbox_inches="tight",
        bbox_extra_artists=(legend,),
    )
    plt.close(fig)
    return result_df


def compute_best_response_surface(
    gp: GameParams,
    aggs: np.ndarray,
    beta_cap_a: float,
    beta_cap_b: float,
    n_possessions: int = 5_000,
    seed: int = GLOBAL_SEED,
    verbose_sim: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Team A payoff surface over the full aggressiveness grid.

    Returns
    -------
    W:
        ``W[i, j] = win_rate_A`` at ``agg_A = aggs[i]``, ``agg_B = aggs[j]``.
    agg_A_star:
        Team A best response for each fixed ``agg_B``.
    agg_B_star:
        Team B best response for each fixed ``agg_A`` (minimizes Team A payoff).
    """
    rng = np.random.default_rng(seed)
    set_simulation_rng(rng)
    aggs = np.asarray(aggs, dtype=float)
    W = np.empty((len(aggs), len(aggs)), dtype=float)

    for i, agg_a in enumerate(aggs):
        for j, agg_b in enumerate(aggs):
            team_a = build_team(beta_cap_a, float(agg_a), 1)
            team_b = build_team(beta_cap_b, float(agg_b), -1)
            stats = simulate_possessions(team_a, team_b, gp, rng, n_possessions, verbose=verbose_sim)
            total_goals = stats.team_a_score + stats.team_b_score
            W[i, j] = (
                float(stats.team_a_score) / float(total_goals) if total_goals > 0 else float("nan")
            )

    a_star_idx = np.nanargmax(W, axis=0)
    b_star_idx = np.nanargmin(W, axis=1)
    return W, aggs[a_star_idx], aggs[b_star_idx]


def plot_best_response_surface(
    out_dir: Path,
    aggs: np.ndarray,
    W: np.ndarray,
    agg_A_star: np.ndarray,
    agg_B_star: np.ndarray,
    beta_cap_a: float,
    beta_cap_b: float,
) -> None:
    """
    Plot Team A payoff surface with both players' best-response curves.
    """
    fig, ax = plt.subplots(figsize=(7.5, 6.2))
    im = ax.imshow(
        W,
        origin="lower",
        aspect="auto",
        extent=[float(aggs[0]), float(aggs[-1]), float(aggs[0]), float(aggs[-1])],
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
    )
    plt.colorbar(im, ax=ax, label="Team A win rate")

    # For each fixed agg_B (x-axis), Team A chooses agg_A (y-axis) to maximize W.
    ax.plot(aggs, agg_A_star, color="red", linewidth=2.2, label=r"$\beta_A^*(\beta_B)$")
    # For each fixed agg_A (y-axis), Team B chooses agg_B (x-axis) to minimize W.
    ax.plot(agg_B_star, aggs, color="white", linewidth=2.2, label=r"$\beta_B^*(\beta_A)$")

    ax.set_xlabel("Team B aggressiveness")
    ax.set_ylabel("Team A aggressiveness")
    ax.set_title(
        "Best-Response Surface\n"
        f"(beta_cap_A={beta_cap_a:.2f}, beta_cap_B={beta_cap_b:.2f})"
    )
    ax.grid(alpha=0.15)
    ax.legend(loc="upper right", framealpha=0.95)
    fig.tight_layout()
    fig.savefig(
        out_dir / f"best_response_surface_beta_cap_A_{beta_cap_a:.2f}_beta_cap_B_{beta_cap_b:.2f}.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)


def _equilibrium_cell_from_W(W: np.ndarray, aggs: np.ndarray) -> tuple[int, int, str]:
    """
    Select a reference grid cell (i, j) for equilibrium-style aggressiveness reporting.

    Prefer a pure-strategy Nash cell on the discrete grid (mutual best responses).
    If none exist, run sequential best-response dynamics; on a repeat, snap mean
    aggressiveness over the detected cycle to the nearest grid indices.
    """
    aggs = np.asarray(aggs, dtype=float)
    n = int(W.shape[0])
    a_br = np.nanargmax(W, axis=0)
    b_br = np.nanargmin(W, axis=1)

    pure: list[tuple[int, int]] = []
    for ii in range(n):
        for jj in range(n):
            if int(a_br[jj]) == ii and int(b_br[ii]) == jj:
                pure.append((ii, jj))
    if pure:
        i, j = max(pure, key=lambda ij: (W[ij[0], ij[1]], -ij[0], -ij[1]))
        return i, j, "pure_nash"

    mid = n // 2
    i, j = mid, mid
    path: list[tuple[int, int]] = []
    for _ in range(500):
        if int(a_br[j]) == i and int(b_br[i]) == j:
            return i, j, "br_sink"
        try:
            k = path.index((i, j))
            cycle = path[k:]
            mean_a = float(np.mean([aggs[p[0]] for p in cycle]))
            mean_b = float(np.mean([aggs[p[1]] for p in cycle]))
            i_snap = int(np.argmin(np.abs(aggs - mean_a)))
            j_snap = int(np.argmin(np.abs(aggs - mean_b)))
            return i_snap, j_snap, "cycle_mean"
        except ValueError:
            path.append((i, j))
        i, j = int(a_br[j]), int(b_br[i])

    return i, j, "br_cap"


def sweep_capability_equilibrium_grid(
    gp: GameParams,
    aggs: np.ndarray,
    capability_grid: np.ndarray | None = None,
    n_possessions: int = 5_000,
    seed: int = GLOBAL_SEED,
) -> tuple[
    pd.DataFrame,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    For each (team A capability, team B capability) on a square grid in [0.6, 1.0],
    estimate the aggressiveness payoff surface and record equilibrium-style aggressiveness
    and Team A win rate at the selected cell.

    Returns
    -------
    df:
        Long-form table with one row per capability pair.
    caps_a, caps_b:
        1-D capability coordinates (sorted ascending).
    Z_agg_A, Z_agg_B, Z_win_A:
        2-D arrays indexed by (i_cap_a, i_cap_b) matching ``caps_a``, ``caps_b`` order
        (rows = A capability, columns = B capability).
    Z_gap, Z_mean_cap:
        Same shape: capability gap ``beta_cap_A - beta_cap_B`` and mean capability.
    Z_kind:
        Same shape, string labels: ``pure_nash``, ``br_sink``, ``cycle_mean``, or ``br_cap``.
    """
    if capability_grid is None:
        capability_grid = np.linspace(0.6, 1.0, 9)
    caps_a = np.asarray(capability_grid, dtype=float)
    caps_b = np.asarray(capability_grid, dtype=float)
    aggs = np.asarray(aggs, dtype=float)

    na, nb = len(caps_a), len(caps_b)
    Z_agg_A = np.empty((na, nb), dtype=float)
    Z_agg_B = np.empty((na, nb), dtype=float)
    Z_win_A = np.empty((na, nb), dtype=float)
    Z_kind = np.empty((na, nb), dtype=object)
    Z_gap = np.empty((na, nb), dtype=float)
    Z_mean_cap = np.empty((na, nb), dtype=float)

    rows: list[dict[str, float | str]] = []

    for ia, beta_cap_a in enumerate(caps_a):
        for ib, beta_cap_b in enumerate(caps_b):
            cell_seed = int(seed) + ia * 1_000_003 + ib * 17_389
            W, _, _ = compute_best_response_surface(
                gp=gp,
                aggs=aggs,
                beta_cap_a=float(beta_cap_a),
                beta_cap_b=float(beta_cap_b),
                n_possessions=n_possessions,
                seed=cell_seed,
                verbose_sim=False,
            )
            i_eq, j_eq, kind = _equilibrium_cell_from_W(W, aggs)
            agg_a_eq = float(aggs[i_eq])
            agg_b_eq = float(aggs[j_eq])
            win_a = float(W[i_eq, j_eq])

            Z_agg_A[ia, ib] = agg_a_eq
            Z_agg_B[ia, ib] = agg_b_eq
            Z_win_A[ia, ib] = win_a
            Z_kind[ia, ib] = kind
            Z_gap[ia, ib] = float(beta_cap_a) - float(beta_cap_b)
            Z_mean_cap[ia, ib] = 0.5 * (float(beta_cap_a) + float(beta_cap_b))

            rows.append(
                {
                    "beta_cap_A": float(beta_cap_a),
                    "beta_cap_B": float(beta_cap_b),
                    "capability_gap": Z_gap[ia, ib],
                    "mean_capability": Z_mean_cap[ia, ib],
                    "agg_A_eq": agg_a_eq,
                    "agg_B_eq": agg_b_eq,
                    "win_rate_A_eq": win_a,
                    "equilibrium_kind": kind,
                }
            )

    df = pd.DataFrame(rows)
    return df, caps_a, caps_b, Z_agg_A, Z_agg_B, Z_win_A, Z_gap, Z_mean_cap, Z_kind


def plot_capability_interaction_heatmaps(
    out_dir: Path,
    caps_a: np.ndarray,
    caps_b: np.ndarray,
    Z_agg_A: np.ndarray,
    Z_agg_B: np.ndarray,
    Z_win_A: np.ndarray,
    n_possessions: int,
) -> None:
    """2×2 panel: equilibrium aggressiveness (A, B) and Team A win rate vs capabilities."""
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 9.5))
    extent = [
        float(caps_b[0]),
        float(caps_b[-1]),
        float(caps_a[0]),
        float(caps_a[-1]),
    ]

    def _panel(ax: plt.Axes, Z: np.ndarray, title: str, cbar_label: str, cmap: str, vmin: float, vmax: float) -> None:
        im = ax.imshow(
            Z,
            origin="lower",
            aspect="auto",
            extent=extent,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        plt.colorbar(im, ax=ax, label=cbar_label)
        ax.set_xlabel("Team B capability (beta_cap)")
        ax.set_ylabel("Team A capability (beta_cap)")
        ax.set_title(title)
        ax.grid(alpha=0.2)

    _panel(
        axes[0, 0],
        Z_agg_A,
        r"Equilibrium-style aggressiveness $\beta_A^{*}$",
        r"$\beta_A$",
        "magma",
        float(np.nanmin(Z_agg_A)),
        float(np.nanmax(Z_agg_A)),
    )
    _panel(
        axes[0, 1],
        Z_agg_B,
        r"Equilibrium-style aggressiveness $\beta_B^{*}$",
        r"$\beta_B$",
        "cividis",
        float(np.nanmin(Z_agg_B)),
        float(np.nanmax(Z_agg_B)),
    )
    _panel(axes[1, 0], Z_win_A, "Team A win rate at selected cell", "P(A wins)", "viridis", 0.0, 1.0)

    ax_sc = axes[1, 1]
    # Long-form scatter: gap vs mean capability, colored by beta_A*
    ga = np.array([float(caps_a[i]) for i in range(Z_agg_A.shape[0]) for _ in range(Z_agg_A.shape[1])])
    gb = np.array([float(caps_b[j]) for _ in range(Z_agg_A.shape[0]) for j in range(Z_agg_A.shape[1])])
    z_flat = Z_agg_A.ravel()
    x_gap = ga - gb
    y_mean = 0.5 * (ga + gb)
    sc = ax_sc.scatter(x_gap, y_mean, c=z_flat, cmap="plasma", s=120, edgecolors="k", linewidths=0.35)
    plt.colorbar(sc, ax=ax_sc, label=r"$\beta_A^{*}$")
    ax_sc.set_xlabel(r"Capability gap $\beta_A - \beta_B$")
    ax_sc.set_ylabel(r"Mean capability $(\beta_A + \beta_B)/2$")
    ax_sc.set_title(r"$\beta_A^{*}$ vs gap and mean skill (grid cells)")
    ax_sc.grid(alpha=0.3)

    fig.suptitle(f"Capability × capability interaction (n_possessions={n_possessions} per grid cell)", fontsize=12)
    fig.tight_layout()
    out_path = out_dir / f"capability_interaction_heatmaps_n{n_possessions}.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> pd.DataFrame:
    gp = GameParams()
    out_dir = Path(__file__).resolve().parent / "exp_agg_capability"
    out_dir.mkdir(parents=True, exist_ok=True)
    aggs = np.linspace(0.2, 0.9, 24)
    cap_grid = np.linspace(0.6, 1.0, 7)
    n_poss = 10
    df, caps_a, caps_b, Z_agg_A, Z_agg_B, Z_win_A, _, _, _ = sweep_capability_equilibrium_grid(
        gp=gp,
        aggs=aggs,
        capability_grid=cap_grid,
        n_possessions=n_poss,
        seed=GLOBAL_SEED,
    )
    df.to_csv(out_dir / f"capability_equilibrium_sweep_n{n_poss}.csv", index=False)
    plot_capability_interaction_heatmaps(
        out_dir=out_dir,
        caps_a=caps_a,
        caps_b=caps_b,
        Z_agg_A=Z_agg_A,
        Z_agg_B=Z_agg_B,
        Z_win_A=Z_win_A,
        n_possessions=n_poss,
    )
    print(f"Saved {len(df)} sweep rows and heatmaps under {out_dir}")
    return df


if __name__ == "__main__":
    main()


    # W, agg_A_star, agg_B_star = compute_best_response_surface(
    #     gp=gp,
    #     aggs=aggs,
    #     beta_cap_a=0.5,
    #     beta_cap_b=0.4,
    #     n_possessions=100,
    #     seed=GLOBAL_SEED,
    # )
    # plot_best_response_surface(
    #     out_dir=out_dir,
    #     aggs=aggs,
    #     W=W,
    #     agg_A_star=agg_A_star,
    #     agg_B_star=agg_B_star,
    #     beta_cap_a=0.5,
    #     beta_cap_b=0.4,
    # )