"""
visualize.py  —  Generates all four diagnostic figures from experiment results.

Must be run after experiment.py has completed. Reads .npy files from results/
and writes PNGs to results/figures/.

Outputs:
    results/figures/01_learning_curves.png    — reward per episode, 3 conditions
    results/figures/02_convergence.png         — ||Q_{k+1} - Q_k||_inf vs iteration
    results/figures/03_planner_usage.png       — executive control arbitration ratio
    results/figures/04_path_comparison.png     — grid path: heuristic vs NCSF_FULL
    results/figures/05_exploration_coverage.png — unique (s,a) pairs over episodes

Usage: python visualize.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')   # no display needed
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

from gridworld import GridWorld, GOAL, START, GRID_SIZE

# ── Config ─────────────────────────────────────────────────────────────────────

RES_DIR  = 'results'
FIG_DIR  = os.path.join(RES_DIR, 'figures')
SEED     = 42
SMOOTH_W = 25   # rolling-average window for noisy curves

os.makedirs(FIG_DIR, exist_ok=True)

# consistent palette — these colours look decent in both light and dark GitHub previews
COL_H = '#E07B54'   # orange  — heuristic
COL_P = '#5E8FBF'   # blue    — planner
COL_N = '#4CAF50'   # green   — NCSF full


def smooth(arr: np.ndarray, w: int) -> np.ndarray:
    """Causal moving average — keeps length consistent with raw for overlays."""
    return np.convolve(arr, np.ones(w) / w, mode='valid')


def save(fig, name: str) -> None:
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  saved: {name}")


# ── Load ───────────────────────────────────────────────────────────────────────

rewards   = np.load(os.path.join(RES_DIR, 'episode_rewards.npy'))
conv      = np.load(os.path.join(RES_DIR, 'convergence_history.npy'))
usage     = np.load(os.path.join(RES_DIR, 'planner_usage.npy'))
coverage  = np.load(os.path.join(RES_DIR, 'coverage_log.npy'))
path_h    = np.load(os.path.join(RES_DIR, 'path_heuristic.npy'))
path_n    = np.load(os.path.join(RES_DIR, 'path_ncsf.npy'))

env = GridWorld(seed=SEED)   # recreate same obstacle layout

rw_h, rw_p, rw_n = rewards[0], rewards[1], rewards[2]
n_ep = len(rw_h)


# ── Figure 1: Learning curves ──────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(11, 5))

for arr, col, label in [
    (rw_h, COL_H, 'Heuristic Only'),
    (rw_p, COL_P, 'Planner Only'),
    (rw_n, COL_N, 'NCSF Full'),
]:
    ep_axis = np.arange(n_ep)
    ax.plot(ep_axis, arr, alpha=0.15, color=col, linewidth=0.7)
    s = smooth(arr, SMOOTH_W)
    ax.plot(np.arange(SMOOTH_W - 1, n_ep), s,
            color=col, linewidth=2.4,
            label=f'{label}  (μ={np.mean(arr):.1f})')

ax.axhline(np.mean(rw_h), color=COL_H, linestyle=':', alpha=0.55, linewidth=1)
ax.set_xlabel('Episode', fontsize=12)
ax.set_ylabel('Total Episode Reward', fontsize=12)
ax.set_title('Learning Curves — Three Policy Conditions\n'
             'Bounded Optimality: NCSF ≥ Heuristic baseline', fontsize=13)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.25)
fig.tight_layout()
save(fig, '01_learning_curves.png')


# ── Figure 2: Value Iteration convergence ─────────────────────────────────────

fig, ax = plt.subplots(figsize=(9, 4))
ax.semilogy(conv, color=COL_P, linewidth=2.2)
ax.axhline(1e-4, color='crimson', linestyle='--', linewidth=1.2,
           label='ε = 1×10⁻⁴  (convergence threshold)')
ax.set_xlabel('Value Iteration Sweep', fontsize=12)
ax.set_ylabel('||Q_{k+1} − Q_k||_∞   (log scale)', fontsize=12)
ax.set_title('Bellman Operator Convergence\n'
             'Contraction factor γ = 0.95 → geometric decay rate', fontsize=13)
ax.legend(fontsize=11)
ax.grid(True, which='both', alpha=0.25)
fig.tight_layout()
save(fig, '02_convergence.png')


# ── Figure 3: Planner vs Heuristic usage ──────────────────────────────────────

fig, ax = plt.subplots(figsize=(11, 4))
ep_axis = np.arange(n_ep)
ax.plot(ep_axis, usage, alpha=0.18, color=COL_N, linewidth=0.7)
s_usage = smooth(usage, SMOOTH_W)
ax.plot(np.arange(SMOOTH_W - 1, n_ep), s_usage,
        color=COL_N, linewidth=2.4,
        label=f'Planner usage (MA-{SMOOTH_W})')
ax.axhline(np.mean(usage), color='grey', linestyle='--', linewidth=1.1,
           label=f'Mean = {np.mean(usage):.1%}')
ax.fill_between(np.arange(SMOOTH_W - 1, n_ep), s_usage, alpha=0.12, color=COL_N)
ax.set_ylim(-0.05, 1.05)
ax.set_xlabel('Episode', fontsize=12)
ax.set_ylabel('Fraction of Steps → Planner', fontsize=12)
ax.set_title('Executive Control Arbitration Ratio\n'
             'NCSF_FULL: planner overrides heuristic only when confident (ξ = 0.5)',
             fontsize=13)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.25)
fig.tight_layout()
save(fig, '03_planner_usage.png')


# ── Figure 4: Grid path comparison ────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(13, 6))

for ax, path, title, path_col in [
    (axes[0], path_h, 'Heuristic Only — Final Episode Path', COL_H),
    (axes[1], path_n, 'NCSF Full — Final Episode Path',      COL_N),
]:
    # draw grid
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            if not env.grid[r][c]:
                face = '#2c2c2c'
            elif (r, c) == GOAL:
                face = '#FFD700'
            elif (r, c) == START:
                face = '#87CEEB'
            else:
                face = '#f5f5f0'
            rect = mpatches.Rectangle(
                (c, GRID_SIZE - 1 - r), 1, 1,
                facecolor=face, edgecolor='#cccccc', linewidth=0.4
            )
            ax.add_patch(rect)

    # draw path as connected dots
    if len(path) > 1:
        pa   = np.array(path)
        xs   = pa[:, 1] + 0.5
        ys   = GRID_SIZE - pa[:, 0] - 0.5
        ax.plot(xs, ys, color=path_col, linewidth=2.2,
                marker='o', markersize=4.5, alpha=0.85, zorder=5)
        # mark start of path
        ax.plot(xs[0], ys[0], 'o', color='black', markersize=7, zorder=6)

    ax.set_xlim(0, GRID_SIZE)
    ax.set_ylim(0, GRID_SIZE)
    ax.set_xticks(range(GRID_SIZE + 1))
    ax.set_yticks(range(GRID_SIZE + 1))
    ax.grid(True, linewidth=0.4, color='#aaaaaa')
    ax.set_title(title, fontsize=12)
    ax.set_aspect('equal')

legend_handles = [
    mpatches.Patch(color='#87CEEB', label='Start (0,0)'),
    mpatches.Patch(color='#FFD700', label='Goal (7,7)'),
    mpatches.Patch(color='#2c2c2c', label='Obstacle'),
]
fig.legend(handles=legend_handles, loc='lower center', ncol=3,
           fontsize=10, bbox_to_anchor=(0.5, -0.04))
fig.suptitle('Path Comparison on Shared 8×8 GridWorld', fontsize=14, y=1.01)
fig.tight_layout()
save(fig, '04_path_comparison.png')


# ── Figure 5: Exploration coverage (curiosity benefit) ───────────────────────

fig, ax = plt.subplots(figsize=(10, 4))
ep_axis = np.arange(n_ep)
ax.plot(ep_axis, coverage, color=COL_N, linewidth=2.2)
ax.fill_between(ep_axis, coverage, alpha=0.12, color=COL_N)

# theoretical max: |S| × |A| state-action pairs (but many are blocked)
n_passable = len(env.get_all_states())
max_sa = n_passable * 4
ax.axhline(max_sa, color='grey', linestyle='--', linewidth=1,
           label=f'Max reachable |S||A| ≈ {max_sa}')

ax.set_xlabel('Episode', fontsize=12)
ax.set_ylabel('Unique (s, a) Pairs Visited', fontsize=12)
ax.set_title('Exploration Coverage Under Curiosity-Driven Bonus\n'
             'Intrinsic reward drives systematic state-space coverage',
             fontsize=13)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.25)
fig.tight_layout()
save(fig, '05_exploration_coverage.png')

print("\nAll figures saved to results/figures/")
