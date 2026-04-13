"""
Sweep throw-mix aggressiveness across multiple team capability levels.

Interpolates discrete weights between a balanced mix (P_discrete) and a more
aggressive mix (P_aggressive), same buckets as samecapability_aggressivevsdiscrete.py.
Both teams share the same capability scale (bucket completion × scale); Team A
stays on the balanced mix; Team B's mix moves from balanced → aggressive (α).
"""

from __future__ import annotations

from TeamParam import TeamParam
from GameParams import GameParams
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
import matplotlib.pyplot as plt

P_discrete = [
    (-10, 0.05),  # backhand reset
    (-1.0, 0.10),
    (7.0, 0.30),   # short reset/dump
    (12.0, 0.30),  # short gain
    (20.0, 0.25),  # medium progression
    (35.0, 0.10),  # long throw
    (50.0, 0.05),  # deep shot
]

# Same distance buckets as P_discrete; more weight on long throws (fixed mix).
P_aggressive = [
    (-10, 0.05),  # backhand reset
    (-1.0, 0.10),
    (7.0, 0.10),
    (12.0, 0.15),
    (20.0, 0.20),
    (35.0, 0.30),
    (50.0, 0.25),
]

T_list = [
    0.98,
    0.95,
    0.90,
    0.85,
    0.80,
    0.70,
    0.60,
]


def bucket_completion_probs(T_fns, P, gp: GameParams) -> list[float]:
    dists = [float(d) for d, _ in P]
    out: list[float] = []
    for i in range(len(P)):
        t = T_fns[i]
        out.append(float(t(dists[i], gp.d_max)) if callable(t) else float(t))
    return out


def T_probs_for_capability(scale: float, gp: GameParams) -> list[float]:
    base = bucket_completion_probs(T_list, P_discrete, gp)
    return [float(min(1.0, max(0.0, p * scale))) for p in base]


def blend_throw_mix(
    P_lo: list[tuple[float, float]],
    P_hi: list[tuple[float, float]],
    alpha: float,
) -> list[tuple[float, float]]:
    """alpha=0 → P_lo weights; alpha=1 → P_hi weights (same distance buckets)."""
    if len(P_lo) != len(P_hi):
        raise ValueError("P_lo and P_hi must have the same number of buckets")
    a = float(np.clip(alpha, 0.0, 1.0))
    out: list[tuple[float, float]] = []
    for (d, w0), (_, w1) in zip(P_lo, P_hi):
        out.append((float(d), (1.0 - a) * float(w0) + a * float(w1)))
    return out


def discrete_probs(P: list[tuple[float, float]]) -> np.ndarray:
    w = np.array([b for _, b in P], dtype=float)
    return w / w.sum()


def simulate_possessions(
    team_a: TeamParam,
    team_b: TeamParam,
    gp: GameParams,
    rng: np.random.Generator,
    n_possessions: int,
    max_plays: int = 10_000,
) -> tuple[int, int]:
    team_a_score = 0
    team_b_score = 0
    offense = team_a

    for _ in range(n_possessions):
        if offense is team_a:
            x = float(rng.uniform(0.6 * gp.D_full, gp.D_full))
        else:
            x = float(rng.uniform(0, 0.4 * gp.D_full))

        for _ in range(max_plays):
            d, complete = offense.sample_throw(x, gp, rng)
            x = gp.position_after_throw(x, d, offense.attack_sign)
            if complete:
                if offense is team_a and gp.in_E_A(x):
                    team_a_score += 1
                    break
                if offense is team_b and gp.in_E_B(x):
                    team_b_score += 1
                    break
            else:
                offense = team_b if offense is team_a else team_a
    return team_a_score, team_b_score


def main() -> None:
    gp = GameParams()
    rng = np.random.default_rng(42)

    capability_scales = np.linspace(0.4, 1.0, 10)
    alphas = np.linspace(0.0, 1.0, 21)
    n_sims = 25_000

    dists = np.array([d for d, _ in P_discrete], dtype=float)

    n_s = len(capability_scales)
    n_a = len(alphas)
    b_share_grid = np.full((n_s, n_a), np.nan, dtype=float)

    for si, scale in enumerate(capability_scales):
        T_probs = T_probs_for_capability(float(scale), gp)
        for ai, a in enumerate(alphas):
            P_b = blend_throw_mix(P_discrete, P_aggressive, float(a))
            team_a = TeamParam(T=T_probs, P=P_discrete, attack_sign=1)
            team_b = TeamParam(T=T_probs, P=P_b, attack_sign=-1)
            sa, sb = simulate_possessions(team_a, team_b, gp, rng, n_sims)
            total = sa + sb
            b_share_grid[si, ai] = float(sb / total) if total > 0 else float("nan")

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))

    cmap_lines = cm.get_cmap("viridis")
    norm = mcolors.Normalize(
        vmin=float(capability_scales.min()),
        vmax=float(capability_scales.max()),
    )
    for si, scale in enumerate(capability_scales):
        color = cmap_lines(norm(float(scale)))
        axes[0].plot(
            alphas,
            b_share_grid[si],
            "o-",
            color=color,
            markersize=3,
            linewidth=1.4,
            label=f"{scale:.2f}" if si % 3 == 0 or si == n_s - 1 else "_nolegend_",
        )
    axes[0].axhline(0.5, color="gray", linestyle="--", linewidth=1, label="even split")
    axes[0].set_xlabel("Aggressiveness α (0 = balanced mix, 1 = aggressive mix)")
    axes[0].set_ylabel("Share of scores (Team B)")
    axes[0].set_title(
        f"Balanced (A) vs blended mix (B), by team capability scale\n({n_sims:,} possessions / point)"
    )
    axes[0].set_ylim(0, 1)
    axes[0].legend(title="cap. scale", loc="best", fontsize=7, title_fontsize=8)
    axes[0].grid(True, alpha=0.3)
    sm = cm.ScalarMappable(cmap=cmap_lines, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=axes[0], label="Capability scale", fraction=0.046, pad=0.04)

    aa, ss = np.meshgrid(alphas, capability_scales)
    pcm = axes[1].pcolormesh(
        aa,
        ss,
        b_share_grid,
        shading="auto",
        cmap="RdYlBu_r",
        vmin=0.0,
        vmax=1.0,
    )
    axes[1].set_xlabel("Aggressiveness α (B's mix: balanced → aggressive)")
    axes[1].set_ylabel("Capability scale (same for A and B)")
    axes[1].set_title("Team B score share (heatmap)")
    fig.colorbar(pcm, ax=axes[1], label="Team B share", fraction=0.046, pad=0.04)

    # Throw mix reference (α only; independent of capability)
    fig2, axb = plt.subplots(figsize=(7.5, 3.2))
    xb = np.arange(len(dists))
    wbar = 0.25
    show_alphas = [0.0, 0.5, 1.0]
    bar_colors = ["#4C72B0", "#8172B3", "#DD8452"]
    for j, aa in enumerate(show_alphas):
        p = discrete_probs(blend_throw_mix(P_discrete, P_aggressive, aa))
        axb.bar(
            xb + (j - 1) * wbar,
            p,
            wbar,
            label=f"α={aa:.1f}",
            color=bar_colors[j],
            alpha=0.85,
        )
    axb.set_xticks(xb)
    axb.set_xticklabels([f"{d:.0f} m" for d in dists])
    axb.set_ylabel("P(attempt this distance)")
    axb.set_title("Throw mix at sample α (weights only; same buckets as main figure)")
    axb.legend(loc="upper left", fontsize=8)
    axb.grid(True, axis="y", alpha=0.3)
    fig2.tight_layout()
    mix_path = "samecapability_aggressive_level_mix_reference.png"
    fig2.savefig(mix_path, dpi=150)
    plt.close(fig2)

    fig.tight_layout()
    out_path = "samecapability_aggressive_level_sweep.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    print(f"Saved {out_path}, {mix_path}")
    print("B_share at α∈{0,0.5,1} (rows = capability scale):")
    for si, scale in enumerate(capability_scales):
        idx0, idxh, idx1 = 0, n_a // 2, n_a - 1
        print(
            f"  scale={scale:.3f}: "
            f"α=0 → {b_share_grid[si, idx0]:.4f}, "
            f"α=0.5 → {b_share_grid[si, idxh]:.4f}, "
            f"α=1 → {b_share_grid[si, idx1]:.4f}"
        )


if __name__ == "__main__":
    main()
