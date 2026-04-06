from __future__ import annotations

from dataclasses import dataclass
from math import lgamma
from typing import Callable, Dict, List, Optional, Tuple, Union
from GameParams import GameParams
from TeamParam import TeamParam
import numpy as np

Possession = str  # 'A' or 'B'


# Type aliases: distance in [0, d_max], return completion probability in [0, 1]
SampleD = Callable[[float], float]
CompletionFn = Callable[[float, float], float]


def clamp(x: float, lo: float, hi: float) -> float:
    return float(min(max(x, lo), hi))


def step_transition(
    x: float,
    p: Possession,
    d: float,
    complete: bool,
    prm: GameParams,
) -> Tuple[float, Possession, Optional[str]]:
    """
    One throw attempt from state (x, p) with chosen distance d and completion outcome.

    Ordering: OOB is determined by the intended landing spot; if OOB, turnover at boundary
    (no scoring on that throw). Otherwise completion is resolved; incomplete uses tilde/hat
    placement from the model.

    Returns (x_next, p_next, terminal) where terminal is 'A_wins', 'B_wins', or None.
    """
    d = float(max(0.0, min(d, prm.d_max)))

    if p == "A":
        x_land = x + d
        if x_land > prm.x_max:
            return prm.x_max, "B", None
        if x_land < prm.x_min:
            return prm.x_min, "B", None

        if complete:
            if prm.in_E_A(x_land):
                return x_land, "A", "A_wins"
            return x_land, "A", None

        x_tilde = min(x_land, prm.x_max)
        return x_tilde, "B", None

    # p == 'B'
    x_land = x - d
    if x_land < prm.x_min:
        return prm.x_min, "A", None
    if x_land > prm.x_max:
        return prm.x_max, "A", None

    if complete:
        if prm.in_E_B(x_land):
            return x_land, "B", "B_wins"
        return x_land, "B", None

    x_hat = max(x_land, prm.x_min)
    return x_hat, "A", None


def simulate_episode(
    x0: float,
    p0: Possession,
    prm: GameParams,
    team_A: TeamParam,
    team_B: TeamParam,
    rng: np.random.Generator,
    max_steps: int = 50_000,
) -> int:
    """Return 1 if Team A wins, 0 if Team B wins."""
    x, p = float(x0), p0

    if p == "A" and prm.in_E_A(x):
        return 1
    if p == "B" and prm.in_E_B(x):
        return 0

    for _ in range(max_steps):
        if p == "A":
            d, complete = team_A.sample_throw(x, prm, rng)
        else:
            d, complete = team_B.sample_throw(x, prm, rng)

        x, p, term = step_transition(x, p, d, complete, prm)
        if term == "A_wins":
            return 1
        if term == "B_wins":
            return 0

    raise RuntimeError(f"Episode did not terminate in {max_steps} steps")


def estimate_W_monte_carlo(
    x0: float,
    p0: Possession,
    prm: GameParams,
    team_A: TeamParam,
    team_B: TeamParam,
    n_paths: int = 10_000,
    seed: Optional[int] = None,
) -> Tuple[float, float]:
    """
    Monte Carlo estimate of W(x0, p0); returns (mean, stderr of mean).
    """
    rng = np.random.default_rng(seed)
    wins = np.empty(n_paths, dtype=np.int8)
    for i in range(n_paths):
        wins[i] = simulate_episode(x0, p0, prm, team_A, team_B, rng)
    mean = float(wins.mean())
    se = float(wins.std(ddof=1) / np.sqrt(n_paths)) if n_paths > 1 else float("nan")
    return mean, se


# -----------------------------------------------------------------------------
# Policy builders (parametric distance distributions)
# -----------------------------------------------------------------------------


def make_beta_sampler(a: float, b: float) -> SampleD:
    """Scaled Beta(a,b) on [0, d_max]."""

    def sample(d_max: float) -> float:
        return float(np.random.beta(a, b) * d_max)

    return sample


def make_uniform_sampler() -> SampleD:
    def sample(d_max: float) -> float:
        return float(np.random.uniform(0.0, d_max))

    return sample


def make_two_point_sampler(d_short_frac: float, p_long: float) -> SampleD:
    """
    With probability p_long throw d_long = d_max * (1 - small), else short throw
    d_short = d_max * d_short_frac.
    """

    def sample(d_max: float) -> float:
        if np.random.random() < p_long:
            return float(0.99 * d_max)
        return float(d_short_frac * d_max)

    return sample


# -----------------------------------------------------------------------------
# Example completion models T_i(d, d_max)
# -----------------------------------------------------------------------------


def T_linear(d: float, d_max: float, base: float = 0.95, slope: float = 0.7) -> float:
    return max(0.0, base - slope * (d / d_max))


def T_exponential(d: float, d_max: float, decay: float = 3.0) -> float:
    return float(np.exp(-decay * (d / d_max)))


def T_constant(p: float) -> CompletionFn:
    def T(d: float, d_max: float) -> float:
        return p

    return T


# -----------------------------------------------------------------------------
# Value iteration (discrete x, discrete d-quadrature for the integral)
# -----------------------------------------------------------------------------


def _interp_W(
    x: float, Wx: np.ndarray, xs: np.ndarray, prm: GameParams
) -> float:
    """Linear interpolation of W on grid; clamp to boundaries for A/B slices."""
    x = clamp(x, prm.x_min, prm.x_max)
    return float(np.interp(x, xs, Wx))


def value_iteration(
    prm: GameParams,
    pdf_A: np.ndarray,
    pdf_B: np.ndarray,
    d_grid: np.ndarray,
    T_A: CompletionFn,
    T_B: CompletionFn,
    n_x: int = 801,
    tol: float = 1e-8,
    max_iter: int = 10_000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Solve for W(x, A) and W(x, B) on a uniform spatial grid.

    pdf_A[k], pdf_B[k] are nonnegative weights proportional to P_i(d_k); they are
    normalized internally so sum_k pdf_*[k] * dd = 1 (trapezoid rule on d_grid).

    Returns (x_grid, W_A, W_B).
    """
    xs = np.linspace(prm.x_min, prm.x_max, n_x)
    WA = np.zeros(n_x)
    WB = np.zeros(n_x)

    # Absorbing: S_A = {(x,A): x in E_A}, S_B = {(x,B): x in E_B}
    for i, x in enumerate(xs):
        if prm.in_E_A(x):
            WA[i] = 1.0
        if prm.in_E_B(x):
            WB[i] = 0.0

    dd = np.diff(d_grid)
    w_trap = np.zeros_like(d_grid)
    w_trap[0] = 0.5 * dd[0]
    w_trap[-1] = 0.5 * dd[-1]
    w_trap[1:-1] = 0.5 * (dd[:-1] + dd[1:])

    pA = pdf_A * w_trap
    pB = pdf_B * w_trap
    pA /= pA.sum()
    pB /= pB.sum()

    active_A = ~np.array([prm.in_E_A(x) for x in xs], dtype=bool)
    active_B = ~np.array([prm.in_E_B(x) for x in xs], dtype=bool)

    for it in range(max_iter):
        WA_old = WA.copy()
        WB_old = WB.copy()

        # W(x, A)
        for i in np.where(active_A)[0]:
            x = xs[i]
            acc = 0.0
            for pa, d in zip(pA, d_grid):
                x_land = x + d
                if x_land > prm.x_max:
                    acc += pa * _interp_W(prm.x_max, WB_old, xs, prm)
                    continue
                if x_land < prm.x_min:
                    acc += pa * _interp_W(prm.x_min, WB_old, xs, prm)
                    continue
                t = clamp(float(T_A(d, prm.d_max)), 0.0, 1.0)
                if prm.in_E_A(x_land):
                    acc += pa * (t * 1.0 + (1.0 - t) * _interp_W(min(x_land, prm.x_max), WB_old, xs, prm))
                else:
                    acc += pa * (
                        t * _interp_W(x_land, WA_old, xs, prm)
                        + (1.0 - t) * _interp_W(min(x_land, prm.x_max), WB_old, xs, prm)
                    )
            WA[i] = acc

        # W(x, B)
        for i in np.where(active_B)[0]:
            x = xs[i]
            acc = 0.0
            for pb, d in zip(pB, d_grid):
                x_land = x - d
                if x_land < prm.x_min:
                    acc += pb * _interp_W(prm.x_min, WA_old, xs, prm)
                    continue
                if x_land > prm.x_max:
                    acc += pb * _interp_W(prm.x_max, WA_old, xs, prm)
                    continue
                t = clamp(float(T_B(d, prm.d_max)), 0.0, 1.0)
                if prm.in_E_B(x_land):
                    acc += pb * (t * 0.0 + (1.0 - t) * _interp_W(max(x_land, prm.x_min), WA_old, xs, prm))
                else:
                    acc += pb * (
                        t * _interp_W(x_land, WB_old, xs, prm)
                        + (1.0 - t) * _interp_W(max(x_land, prm.x_min), WA_old, xs, prm)
                    )
            WB[i] = acc

        delta = max(np.max(np.abs(WA - WA_old)), np.max(np.abs(WB - WB_old)))
        if delta < tol:
            break

    return xs, WA, WB


def beta_pdf_on_grid(d_grid: np.ndarray, d_max: float, a: float, b: float) -> np.ndarray:
    """Unnormalized Beta(a,b) density scaled to [0, d_max], evaluated at d_grid."""
    u = np.clip(d_grid / d_max, 1e-12, 1.0 - 1e-12)
    log_coef = lgamma(a + b) - lgamma(a) - lgamma(b)
    dens = np.exp(log_coef) * (u ** (a - 1)) * ((1.0 - u) ** (b - 1)) / d_max
    return dens


# -----------------------------------------------------------------------------
# Optimization hooks (Monte Carlo objective)
# -----------------------------------------------------------------------------


def optimize_team_A_beta_monte_carlo(
    x0: float,
    p0: Possession,
    prm: GameParams,
    team_A_T: CompletionFn,
    team_B: TeamParam,
    beta_grid: List[Tuple[float, float]],
    n_paths: int = 5000,
    seed: int = 0,
) -> Dict[str, Union[Tuple[float, float], float]]:
    """Grid search over Beta(a,b) for Team A throws; fix B's TeamParam."""
    best = (-1.0, (np.nan, np.nan))
    rng = np.random.default_rng(seed)
    details: List[Tuple[Tuple[float, float], float]] = []

    for a, b in beta_grid:
        # fresh RNG stream per policy for fair comparison
        sub = np.random.default_rng(rng.integers(1_000_000_000))

        def sample_A(dm: float) -> float:
            return float(sub.beta(a, b) * dm)

        team_A = TeamParam(T=team_A_T, P=sample_A)
        wins = np.empty(n_paths, dtype=np.int8)
        for i in range(n_paths):
            wins[i] = simulate_episode(x0, p0, prm, team_A, team_B, sub)
        mean = float(wins.mean())
        details.append(((a, b), mean))
        if mean > best[0]:
            best = (mean, (a, b))

    return {
        "best_mean_W": best[0],
        "best_beta_ab": best[1],
        "all_results": details,
    }


def optimize_both_teams_beta_monte_carlo(
    x0: float,
    p0: Possession,
    prm: GameParams,
    team_A_T: CompletionFn,
    team_B_T: CompletionFn,
    beta_grid: List[Tuple[float, float]],
    n_paths: int = 4000,
    seed: int = 0,
    maximize_A: bool = True,
) -> Dict[str, object]:
    """
    Small joint grid: search (aA,bA) x (aB,bB). Objective is mean terminal payoff
    for A (1 win, 0 loss) if maximize_A else 1 - that (best for B).
    """
    rng = np.random.default_rng(seed)
    best_val = -1.0
    best_pair: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None
    results: List[Tuple[Tuple[float, float], Tuple[float, float], float]] = []

    for abA in beta_grid:
        for abB in beta_grid:
            sub = np.random.default_rng(rng.integers(1_000_000_000))
            aA, bA = abA
            aB, bB = abB

            def sample_A(dm: float) -> float:
                return float(sub.beta(aA, bA) * dm)

            def sample_B(dm: float) -> float:
                return float(sub.beta(aB, bB) * dm)

            team_A = TeamParam(T=team_A_T, P=sample_A)
            team_B = TeamParam(T=team_B_T, P=sample_B)
            wins = np.empty(n_paths, dtype=np.int8)
            for i in range(n_paths):
                wins[i] = simulate_episode(x0, p0, prm, team_A, team_B, sub)
            mean_A = float(wins.mean())
            obj = mean_A if maximize_A else 1.0 - mean_A
            results.append((abA, abB, mean_A))
            if obj > best_val:
                best_val = obj
                best_pair = (abA, abB)

    return {
        "best_objective": best_val,
        "best_beta_A": best_pair[0] if best_pair else None,
        "best_beta_B": best_pair[1] if best_pair else None,
        "mean_A_win_rate": results,
        "maximize_A": maximize_A,
    }


# -----------------------------------------------------------------------------
# Demo
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    prm = GameParams(D_full=100.0, alpha=0.2)
    x0 = 0.5 * prm.D_full
    p0 = "A"

    sample_conservative = make_beta_sampler(2.0, 5.0)
    sample_aggressive = make_beta_sampler(5.0, 2.0)

    team_A_agg = TeamParam(T=T_linear, P=sample_aggressive)
    team_A_con = TeamParam(T=T_linear, P=sample_conservative)
    team_B_con = TeamParam(T=T_linear, P=sample_conservative)

    m1, se1 = estimate_W_monte_carlo(
        x0, p0, prm, team_A_agg, team_B_con, n_paths=8000, seed=42
    )
    m2, se2 = estimate_W_monte_carlo(
        x0, p0, prm, team_A_con, team_B_con, n_paths=8000, seed=43
    )
    print("Monte Carlo W(x0, A), midfield:")
    print(f"  A aggressive vs B conservative: {m1:.4f} ± {se1:.4f}")
    print(f"  Both conservative:              {m2:.4f} ± {se2:.4f}")

    d_grid = np.linspace(0.0, prm.d_max, 400)
    pdf_A = beta_pdf_on_grid(d_grid, prm.d_max, 5.0, 2.0)
    pdf_B = beta_pdf_on_grid(d_grid, prm.d_max, 2.0, 5.0)
    xs, WA, WB = value_iteration(
        prm, pdf_A, pdf_B, d_grid, team_A_agg.T, team_B_con.T
    )
    idx = int(np.argmin(np.abs(xs - x0)))
    print("\nValue iteration at x0 (compare to MC):")
    print(f"  W(x0, A) ≈ {WA[idx]:.4f}, W(x0, B) ≈ {WB[idx]:.4f}")

    grid = [(1.0, 1.0), (2.0, 5.0), (5.0, 2.0), (3.0, 3.0)]
    opt = optimize_team_A_beta_monte_carlo(
        x0, p0, prm, T_linear, team_B_con, grid, n_paths=3000, seed=7
    )
    print("\nGrid search best Beta for Team A (B fixed conservative MC):")
    print(f"  best (a,b) = {opt['best_beta_ab']}, mean W = {opt['best_mean_W']:.4f}")
