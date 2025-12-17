"""
experiment.py  —  Three-condition empirical comparison over 500 episodes.

Conditions:
    HEURISTIC_ONLY  — greedy Manhattan-distance policy, no planning
    PLANNER_ONLY    — pure Value Iteration policy, no curiosity
    NCSF_FULL       — arbitration + curiosity (the proposed hybrid)

All three run on the same GridWorld layout (fixed seed) for fair comparison.
Value Iteration is run once before episodes begin and the Q-table is reused.

Outputs (written to results/):
    episode_rewards.npy       (3 × N_EPISODES)
    episode_steps.npy         (3 × N_EPISODES)
    goal_reached.npy          (3 × N_EPISODES, bool)
    convergence_history.npy   (variable length — iterations until VI converged)
    planner_usage.npy         (N_EPISODES — fraction per episode for NCSF_FULL)
    coverage_log.npy          (N_EPISODES — unique (s,a) pairs visited by NCSF_FULL)
    path_heuristic.npy        (last-episode path)
    path_ncsf.npy             (last-episode path)

Usage: python experiment.py
"""

import os
import random
import numpy as np

from gridworld import GridWorld
from heuristic_policy import HeuristicPolicy
from deliberative_planner import DeliberativePlanner
from motivation_engine import MotivationEngine
from executive_control import ExecutiveControl

# ── Config ─────────────────────────────────────────────────────────────────────

N_EPISODES  = 500
MAX_STEPS   = 200    # episode cap — prevents infinite loops on hard maps
SEED        = 42
OUT_DIR     = 'results'

random.seed(SEED)
np.random.seed(SEED)

os.makedirs(OUT_DIR, exist_ok=True)


# ── Shared episode runner ──────────────────────────────────────────────────────

def run_episode_simple(env: GridWorld, policy_fn,
                       record_path: bool = False) -> tuple:
    """
    Rolls out one episode using policy_fn(state) → int action.
    Returns (total_reward, n_steps, goal_reached, path).
    """
    state        = env.reset()
    total_reward = 0.0
    path         = [state] if record_path else []

    for step in range(MAX_STEPS):
        action               = policy_fn(state)
        next_state, reward, done = env.step(action)
        total_reward         += reward
        state                 = next_state
        if record_path:
            path.append(state)
        if done:
            return total_reward, step + 1, True, path

    return total_reward, MAX_STEPS, False, path


# ── Module setup ───────────────────────────────────────────────────────────────

print("Initialising modules...")
env       = GridWorld(seed=SEED)
heuristic = HeuristicPolicy(env)
planner   = DeliberativePlanner(env)
curiosity = MotivationEngine()
arbiter   = ExecutiveControl(env, heuristic, planner)

# Value Iteration runs once up front — planner is fixed for all three conditions
print("\nRunning Value Iteration...")
planner.value_iteration()
conv_hist = planner.get_convergence_history()
np.save(os.path.join(OUT_DIR, 'convergence_history.npy'), np.array(conv_hist))
print(f"  saved {len(conv_hist)} iteration deltas  "
      f"(final δ = {conv_hist[-1]:.2e})")


# ── Condition A: HEURISTIC_ONLY ────────────────────────────────────────────────

print("\n[1/3] HEURISTIC_ONLY ...")
rw_h, st_h, gl_h = [], [], []

for ep in range(N_EPISODES):
    record  = (ep == N_EPISODES - 1)
    r, s, g, last_path_h = run_episode_simple(env, heuristic.get_action, record)
    rw_h.append(r)
    st_h.append(s)
    gl_h.append(g)

np.save(os.path.join(OUT_DIR, 'path_heuristic.npy'), np.array(last_path_h))
print(f"  goal rate {sum(gl_h)/N_EPISODES:.1%} | mean reward {np.mean(rw_h):.2f}")


# ── Condition B: PLANNER_ONLY ─────────────────────────────────────────────────

print("\n[2/3] PLANNER_ONLY ...")
rw_p, st_p, gl_p = [], [], []

for ep in range(N_EPISODES):
    r, s, g, _ = run_episode_simple(env, planner.get_action)
    rw_p.append(r)
    st_p.append(s)
    gl_p.append(g)

print(f"  goal rate {sum(gl_p)/N_EPISODES:.1%} | mean reward {np.mean(rw_p):.2f}")


# ── Condition C: NCSF_FULL ────────────────────────────────────────────────────

print("\n[3/3] NCSF_FULL (curiosity + arbitration) ...")
rw_n, st_n, gl_n = [], [], []
usage_log    = []
coverage_log = []

arbiter.reset_stats()

for ep in range(N_EPISODES):
    state        = env.reset()
    total_reward = 0.0
    ep_planner   = 0
    ep_steps     = 0

    for step in range(MAX_STEPS):
        action, source             = arbiter.select_action(state)
        next_state, r_ext, done   = env.step(action)

        # curiosity augments training signal but we log raw reward for fair comparison
        curiosity.compute_reward(state, action, next_state, r_ext)
        curiosity.update(state, action, next_state)

        total_reward += r_ext
        ep_steps     += 1
        if source == 'planner':
            ep_planner += 1

        state = next_state
        if done:
            break

    rw_n.append(total_reward)
    st_n.append(ep_steps)
    gl_n.append(done)
    usage_log.append(ep_planner / ep_steps if ep_steps > 0 else 0.0)
    coverage_log.append(curiosity.get_coverage())

# record last-episode path for visualisation
last_path_n = []
state = env.reset()
for _ in range(MAX_STEPS):
    action, _ = arbiter.select_action(state)
    state, _, done = env.step(action)
    last_path_n.append(state)
    if done:
        break

np.save(os.path.join(OUT_DIR, 'path_ncsf.npy'), np.array(last_path_n))
np.save(os.path.join(OUT_DIR, 'planner_usage.npy'), np.array(usage_log))
np.save(os.path.join(OUT_DIR, 'coverage_log.npy'), np.array(coverage_log))

print(f"  goal rate {sum(gl_n)/N_EPISODES:.1%} | mean reward {np.mean(rw_n):.2f}")
print(f"  planner used {np.mean(usage_log):.1%} of steps on average")
print(f"  final exploration coverage: {coverage_log[-1]} unique (s,a) pairs")


# ── Save stacked arrays ────────────────────────────────────────────────────────

ep_rewards = np.stack([rw_h, rw_p, rw_n])
ep_steps   = np.stack([st_h, st_p, st_n])
ep_goals   = np.stack([gl_h, gl_p, gl_n])

np.save(os.path.join(OUT_DIR, 'episode_rewards.npy'), ep_rewards)
np.save(os.path.join(OUT_DIR, 'episode_steps.npy'),   ep_steps)
np.save(os.path.join(OUT_DIR, 'goal_reached.npy'),    ep_goals)

print(f"\nAll results saved to {OUT_DIR}/")
