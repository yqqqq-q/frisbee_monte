from TeamParam import TeamParam
from GameParams import GameParams
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

P_discrete = [
    (-10, 0.05),  # backhand reset
    (-1.0, 0.10),
    (7.0, 0.25),   # short reset/dump
    (12.0, 0.20),  # short gain
    (20.0, 0.20),  # medium progression
    (35.0, 0.10),  # long throw
    (50.0, 0.05),  # deep shot
    (70.0, 0.05),  # very long throw (not attempted)
]

# Same distance buckets as P_discrete; more weight on long throws (fixed mix).
P_aggressive = [
    (-10, 0.02),  # backhand reset
    (-1.0, 0.03),
    (7.0, 0.05),
    (12.0, 0.05),
    (20.0, 0.15),
    (35.0, 0.25),
    (50.0, 0.25),
    (70.0, 0.20),  # very long throw (not attempted)
]

T_list = [
    1,
    0.99,
    0.95,
    0.90,
    0.88,
    0.85,
    0.80,
    0.75,
]


def bucket_completion_probs(T_fns, P, gp: GameParams) -> list[float]:
    dists = [float(d) for d, _ in P]
    out: list[float] = []
    for i in range(len(P)):
        t = T_fns[i]
        out.append(float(t(dists[i], gp.d_max)) if callable(t) else float(t))
    return out


def T_probs_for_capability(scale: float, gp: GameParams) -> list[float]:
    """Scale reference bucket completion rates; clip to [0, 1]. Ten scales → ten capability teams."""
    base = bucket_completion_probs(T_list, P_discrete, gp)
    return [float(min(1.0, max(0.0, p * scale))) for p in base]


def simulate_possessions(
    team_a: TeamParam,
    team_b: TeamParam,
    gp: GameParams,
    rng: np.random.Generator,
    n_possessions: int,
    max_plays: int = 10_000,
) -> tuple[
    int,
    int,
    int,
    int,
    list[list[float]],
    list[list[float]],
    list[int],
    list[int],
]:
    """
    Returns scores, turnover counts (incomplete throws while that team has offense),
    per-possession lists of field x immediately before each turnover throw,
    and per-goal counts of that team's turnovers on the scoring possession before the goal.
    """
    team_a_score = 0
    team_b_score = 0
    x_turnover_a: list[list[float]] = []
    x_turnover_b: list[list[float]] = []
    to_before_goal_a: list[int] = []
    to_before_goal_b: list[int] = []
    offense = team_a
    x = float(rng.uniform(0.8 * gp.D_full, gp.D_full))
    for i in range(n_possessions):
        # print("1")
        x_turnover_a.append([])
        x_turnover_b.append([])
        for _ in range(max_plays):
            d, complete = offense.sample_throw(x, gp, rng)
            x = gp.position_after_throw(x, d, offense.attack_sign)
            if complete:
                if offense is team_a and gp.in_E_A(x):
                    team_a_score += 1
                    to_before_goal_a.append(len(x_turnover_a[i]))
                    offense = team_b  # reset to other team for next possession
                    x = float(rng.uniform(0, 0.2 * gp.D_full))
                    break
                if offense is team_b and gp.in_E_B(x):
                    team_b_score += 1
                    to_before_goal_b.append(len(x_turnover_b[i]))
                    offense = team_a  # reset to other team for next possession
                    x = float(rng.uniform(0.8 * gp.D_full, gp.D_full))
                    break
            else:
                if offense is team_a:
                    x_turnover_a[i].append(float(x))
                else:
                    x_turnover_b[i].append(float(x))
                offense = team_b if offense is team_a else team_a
            

    n_turnover_a = sum(len(sub) for sub in x_turnover_a)
    n_turnover_b = sum(len(sub) for sub in x_turnover_b)
    return (
        team_a_score,
        team_b_score,
        n_turnover_a,
        n_turnover_b,
        x_turnover_a,
        x_turnover_b,
        to_before_goal_a,
        to_before_goal_b,
    )


def discrete_probs(P: list[tuple[float, float]]) -> np.ndarray:
    w = np.array([b for _, b in P], dtype=float)
    return w / w.sum()


def flatten_turnover_xs(nested: list[list[float]]) -> np.ndarray:
    return np.array([x for poss in nested for x in poss], dtype=float)


def main() -> None:
    gp = GameParams()
    rng = np.random.default_rng(42)

    # Ten capability levels: same P mixes; only completion profile T changes.
    capability_scales = np.linspace(0.3, 1.0, 70)
    lvl_idx = [0, len(capability_scales) // 2, len(capability_scales) - 1]
    n_sims = 100_000

    dists = np.array([d for d, _ in P_discrete], dtype=float)
    t_ref = np.array(bucket_completion_probs(T_list, P_discrete, gp), dtype=float)
    p_bal = discrete_probs(P_discrete)
    p_agg = discrete_probs(P_aggressive)
    exp_complete_bal = float(np.sum(p_bal * t_ref))
    exp_complete_agg = float(np.sum(p_agg * t_ref))

    T_by_scale = [
        np.array(T_probs_for_capability(float(s), gp), dtype=float) for s in capability_scales
    ]
    exp_complete_bal_levels = np.array([float(np.sum(p_bal * t)) for t in T_by_scale])
    exp_complete_agg_levels = np.array([float(np.sum(p_agg * t)) for t in T_by_scale])

    b_share = []
    a_scores_list: list[int] = []
    b_scores_list: list[int] = []
    turnovers_a_per_scale: list[int] = []
    turnovers_b_per_scale: list[int] = []
    x_turnover_by_scale_idx: dict[int, tuple[list[list[float]], list[list[float]]]] = {}
    mean_to_before_goal_a: list[float] = []
    mean_to_before_goal_b: list[float] = []
    to_before_goal_by_scale_idx: dict[int, tuple[list[int], list[int]]] = {}

    for i, scale in enumerate(capability_scales):
        # print(f"Simulating capability scale {scale:.3f} ({i+1}/{len(capability_scales)})...")
        T_probs = T_probs_for_capability(float(scale), gp)
        team_a = TeamParam(T=T_probs, P=P_discrete, attack_sign=1)
        team_b = TeamParam(T=T_probs, P=P_aggressive, attack_sign=-1)
        sa, sb, to_a, to_b, xa, xb, tba, tbb = simulate_possessions(
            team_a, team_b, gp, rng, n_sims
        )
        a_scores_list.append(sa)
        b_scores_list.append(sb)
        turnovers_a_per_scale.append(to_a)
        turnovers_b_per_scale.append(to_b)
        mean_to_before_goal_a.append(float(np.mean(tba)) if tba else np.nan)
        mean_to_before_goal_b.append(float(np.mean(tbb)) if tbb else np.nan)
        if i in lvl_idx:
            x_turnover_by_scale_idx[i] = (xa, xb)
            to_before_goal_by_scale_idx[i] = (tba, tbb)
        total = sa + sb
        b_share.append(sb / total if total > 0 else np.nan)
        print(
            f"  scale={scale:.3f}: balanced A={a_scores_list[i]}, aggressive B={b_scores_list[i]}, "
            f"B_share={b_share[i]:.4f}"
        )

    b_share = np.array(b_share, dtype=float)
    a_share = 1.0 - b_share
    turnovers_a_per_scale = np.array(turnovers_a_per_scale, dtype=float)
    turnovers_b_per_scale = np.array(turnovers_b_per_scale, dtype=float)
    to_rate_a = turnovers_a_per_scale / n_sims
    to_rate_b = turnovers_b_per_scale / n_sims
    mean_to_before_goal_a = np.array(mean_to_before_goal_a, dtype=float)
    mean_to_before_goal_b = np.array(mean_to_before_goal_b, dtype=float)

    # --- Figure 1: Monte Carlo — aggressive mix share of scores vs capability ---
    fig1, ax1 = plt.subplots(figsize=(8, 4.5))
    ax1.plot(capability_scales, a_share, "o-", color="#4C5392", label="Balanced Strategy")
    ax1.plot(capability_scales, b_share, "o-", color="#A74E3E", label="Aggressive Strategy")
    ax1.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="even split")
    ax1.set_xlabel("Team Capability Scale")
    ax1.set_ylabel("Share of scores")
    ax1.set_title(
        f"Balanced vs Aggressive"
    )
    ax1.set_ylim(0.3, 0.7)
    ax1.legend(loc="best")
    ax1.grid(True, alpha=0.3)
    fig1.tight_layout()
    fig1.savefig("exp1\\aggressiveness_capability_analysis_reset.png", dpi=150)
    plt.close(fig1)

    # --- Figure 2a: Throw mix vs distance; bucket completion vs capability (twin axis) ---
    xb = np.arange(len(dists))
    wbar = 0.40
    cmap = mcolors.LinearSegmentedColormap.from_list(
    "custom_cmap",
    ["#78be71", "#1fb44c", "#287733"]
)


    fig2a, ax2a = plt.subplots(figsize=(6.3, 4.0))
    ax2a.bar(xb - wbar / 2, p_bal, wbar, label="Balanced", color="#4C5392")
    ax2a.bar(xb + wbar / 2, p_agg, wbar, label="Aggressive", color="#A74E3E")
    ax2a.set_xticks(xb)
    ax2a.set_xticklabels([f"{d:.0f} m" for d in dists])
    ax2a.set_ylabel("P(attempt)")
    ax2a.set_title("Strategy Difference")
    ax2a.legend(loc="upper left", fontsize=8)

    ax2ar = ax2a.twinx()
    colors = cmap(np.linspace(0, 1, len(lvl_idx)))
    for j, i in enumerate(lvl_idx):
        ax2ar.plot(
            xb,
            T_by_scale[i],
            "o-",
            color=colors[j],
            linewidth=1.6,
            markersize=4,
            label=f"scale={capability_scales[i]:.2f}",
        )
    ax2ar.set_ylabel("P(complete)")
    bar_top = max(float(np.max(p_bal)), float(np.max(p_agg)))
    ax2a.set_ylim(0, bar_top * 1.30)
    ax2ar.set_ylim(0, 1.25)
    ax2ar.legend(loc="upper right", fontsize=7)
    fig2a.tight_layout()
    fig2a.savefig("exp1\\strategy_mix_throw_distance.png", dpi=150)
    plt.close(fig2a)

    # --- Figure 2b: Marginal per-throw outcome vs capability ---
    fig2b, ax2b = plt.subplots(figsize=(6.5, 4.2))
    ax2b.plot(
        capability_scales,
        exp_complete_bal_levels,
        "o-",
        color="#4C5392",
        label="Balanced: E[Complete)]",
    )
    ax2b.plot(
        capability_scales,
        exp_complete_agg_levels,
        "s-",
        color="#A74E3E",
        label="Aggressive: E[Complete)]",
    )
    ax2b.plot(
        capability_scales,
        1.0 - exp_complete_bal_levels,
        "o--",
        color="#4C5392",
        alpha=0.55,
        label="Balanced: E[Turnover)]",
    )
    ax2b.plot(
        capability_scales,
        1.0 - exp_complete_agg_levels,
        "s--",
        color="#A74E3E",
        alpha=0.55,
        label="Aggressive: E[Turnover)]",
    )
    ax2b.set_xlabel("Capability Scale")
    ax2b.set_ylabel("Marginal Probability per Throw")
    # ax2b.set_title("Per-throw outcome vs capability (same mixes as Fig. 1)")
    ax2b.set_ylim(0, 1)
    ax2b.legend(fontsize=7, loc="best")
    ax2b.grid(True, alpha=0.3)
    fig2b.tight_layout()
    fig2b.savefig("exp1\\strategy_mix_marginal_outcome.png", dpi=150)
    plt.close(fig2b)

    # --- Figure 3a: Turnover rate vs capability (all scales) ---
    fig3a, ax3a = plt.subplots(figsize=(7, 4.2))
    ax3a.plot(
        capability_scales,
        to_rate_a,
        "o-",
        color="#4C5392",
        label="Balanced",
    )
    ax3a.plot(
        capability_scales,
        to_rate_b,
        "s-",
        color="#A74E3E",
        label="Aggressive",
    )
    ax3a.set_xlabel("Capability Scale")
    ax3a.set_ylabel("Mean Turnovers per Play")
    ax3a.legend(fontsize=8, loc="best")
    # ax3a.set_ylim(0.4, 0.6)
    ax3a.grid(True, alpha=0.3)
    fig3a.tight_layout()
    fig3a.savefig("exp1\\turnover_rate_vs_capability.png", dpi=150)
    plt.close(fig3a)

    # # --- Figure 3b: Field x at turnover for low / mid / high capability scale ---
    # bins = np.linspace(0, gp.D_full, 29)
    # fig3b, axes3b = plt.subplots(1, 3, figsize=(12.5, 4.2), sharey=True)
    # for ax, idx in zip(axes3b, lvl_idx):
    #     xa_n, xb_n = x_turnover_by_scale_idx[idx]
    #     x_turn_a_arr = flatten_turnover_xs(xa_n)
    #     x_turn_b_arr = flatten_turnover_xs(xb_n)
    #     s = float(capability_scales[idx])
    #     ax.hist(
    #         x_turn_a_arr,
    #         bins=bins,
    #         density=True,
    #         alpha=0.55,
    #         color="#4C5392",
    #         label=f"Balanced (n={len(x_turn_a_arr):,})",
    #     )
    #     ax.hist(
    #         x_turn_b_arr,
    #         bins=bins,
    #         density=True,
    #         alpha=0.55,
    #         color="#A74E3E",
    #         label=f"Aggressive (n={len(x_turn_b_arr):,})",
    #     )
    #     ax.set_xlabel("Field position x at turnover [m]")
    #     ax.set_title(f"capability scale = {s:.2f}")
    #     ax.legend(fontsize=7, loc="best")
    #     ax.grid(True, alpha=0.3)

    # axes3b[0].set_ylabel("Density")
    # fig3b.tight_layout()
    # fig3b.savefig("exp1\\turnover_x_by_capability_scale.png", dpi=150)
    # plt.close(fig3b)

    # --- Figure 4a: Mean team turnovers on scoring possession before goal vs capability ---
    fig4a, ax4a = plt.subplots(figsize=(7, 4.2))
    ax4a.plot(
        capability_scales,
        mean_to_before_goal_a,
        "o-",
        color="#4C5392",
        label="Balanced",
    )
    ax4a.plot(
        capability_scales,
        mean_to_before_goal_b,
        "s-",
        color="#A74E3E",
        label="Aggressive",
    )
    ax4a.set_xlabel("Capability Scale")
    ax4a.set_ylabel("Mean turnovers on possession before goal")
    ax4a.legend(fontsize=8, loc="best")
    ax4a.grid(True, alpha=0.3)
    fig4a.tight_layout()
    fig4a.savefig("exp1\\turnovers_before_goal_mean_vs_capability.png", dpi=150)
    plt.close(fig4a)

    # =========================
    # FIGURE 4: Turnovers before goal (LOG SCALE)
    # =========================
#3

    bins = np.linspace(0, gp.D_full, 29)
    fig3b, axes3b = plt.subplots(1, 3, figsize=(12.5, 4.2), sharey=True)

    for ax, idx in zip(axes3b, lvl_idx):
        xa_n, xb_n = x_turnover_by_scale_idx[idx]

        x_turn_a_arr = flatten_turnover_xs(xa_n)
        x_turn_b_arr = flatten_turnover_xs(xb_n)

        s = float(capability_scales[idx])

        ax.hist(
            x_turn_a_arr,
            bins=bins,
            density=True,
            alpha=0.55,
            color="#4C5392",
            label=f"Balanced (n={len(x_turn_a_arr):,})",
        )
        ax.hist(
            x_turn_b_arr,
            bins=bins,
            density=True,
            alpha=0.55,
            color="#A74E3E",
            label=f"Aggressive (n={len(x_turn_b_arr):,})",
        )

        ax.set_xlabel("Field position x at turnover [m]")
        ax.set_title(f"capability scale = {s:.2f}")
        ax.legend(fontsize=7, loc="best")
        ax.grid(True, alpha=0.3)

    # --- LOG SCALE ---
    for ax in axes3b:
        ax.set_yscale("log")
        ax.set_ylim(1e-4, None)   # lower bound to avoid log(0)

    axes3b[0].set_ylabel("Density")

    fig3b.tight_layout()
    fig3b.savefig("exp1\\turnover_x_by_capability_scale.png", dpi=150)
    plt.close(fig3b)


    print(
        "Saved aggressiveness_capability_analysis_reset.png, strategy_mix_throw_distance.png, "
        "strategy_mix_marginal_outcome.png, turnover_rate_vs_capability.png, "
        "turnover_x_by_capability_scale.png, turnovers_before_goal_mean_vs_capability.png, "
        "turnovers_before_goal_hist_by_capability.png"
    )
    for i, s in enumerate(capability_scales):
        print(
            f"  scale={s:.3f}: balanced A={a_scores_list[i]}, aggressive B={b_scores_list[i]}, "
            f"B_share={b_share[i]:.4f}"
        )


if __name__ == "__main__":
    main()
