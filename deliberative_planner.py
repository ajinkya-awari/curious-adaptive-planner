"""
deliberative_planner.py  —  Tabular Value Iteration planner.

Implements the Bellman optimality operator B as a γ-contraction mapping
on the space of bounded functions (Sutton & Barto 2018, Ch. 4).
By the Banach Fixed-Point Theorem the iterates Q_k converge to Q* at rate γ^k.

Convergence evidence:  ||Q_{k+1} - Q_k||_∞ is logged each sweep and saved
so the geometric decay can be plotted and verified empirically.

Outputs : writes nothing directly — imported by executive_control, experiment
Usage   : from deliberative_planner import DeliberativePlanner
"""

import numpy as np

from gridworld import GridWorld, GOAL, N_ACTIONS, GRID_SIZE


# ── Planner hyper-parameters ──────────────────────────────────────────────────

GAMMA       = 0.95    # discount — high enough that distant goals matter
MAX_ITER    = 1000
CONV_THRESH = 1e-4    # stop when sup-norm change drops below this

# TODO: experiment with async (in-place) updates — they converge faster in practice
#       but the synchronous version is easier to reason about theoretically


class DeliberativePlanner:
    """
    Synchronous Value Iteration over the full finite state×action space.

    Implements the Bellman backup:
        Q_{k+1}(s,a) = R(s,a) + γ · max_{a'} Q_k(s', a')

    Because the environment is deterministic, the expectation Σ_{s'} P(s'|s,a)
    collapses to a single next state — making each update O(1) per (s,a) pair.

    Reference: Sutton & Barto (2018), Equations 4.9–4.10
    """

    def __init__(self, env: GridWorld):
        self.env    = env
        self.states = env.get_all_states()

        # flat index: row*GRID_SIZE + col — keeps array ops clean
        self._idx   = {s: i for i, s in enumerate(self.states)}
        n           = len(self.states)

        # start with all-zeros Q — convergence holds from any bounded init
        self.Q = np.zeros((n, N_ACTIONS), dtype=np.float64)

        self._convergence_history: list = []

    # ── Planning ───────────────────────────────────────────────────────────────

    def value_iteration(self) -> None:
        """
        Runs synchronous VI sweeps until ||ΔQ||_∞ < CONV_THRESH or MAX_ITER.

        Synchronous = compute all new Q values first, then update all at once.
        Slightly slower than async but the contraction proof is cleaner.
        """
        self._convergence_history = []

        for k in range(MAX_ITER):
            Q_new = np.zeros_like(self.Q)

            for s in self.states:
                if s == GOAL:
                    # terminal absorbing state — Q stays 0 (no future rewards)
                    continue

                si = self._idx[s]

                for a in range(N_ACTIONS):
                    # deterministic env → Σ P(s'|s,a) collapses to single term
                    # Q(s,a) = R(s,a) + γ · max_{a'} Q(s', a')    [Bellman backup]
                    s_next, reward = self.env.transition(s, a)
                    si_next        = self._idx[s_next]

                    if s_next == GOAL:
                        max_future = 0.0
                    else:
                        max_future = float(np.max(self.Q[si_next]))

                    Q_new[si, a] = reward + GAMMA * max_future

            # track the sup-norm distance between successive iterates
            # theory predicts this decays like γ^k — should show geometric decay in plot
            delta = float(np.max(np.abs(Q_new - self.Q)))
            self._convergence_history.append(delta)
            self.Q = Q_new

            if delta < CONV_THRESH:
                print(f"  value iteration converged at sweep {k+1}  (δ = {delta:.2e})")
                break
        else:
            print(f"  warning: reached MAX_ITER={MAX_ITER}, "
                  f"final δ = {self._convergence_history[-1]:.2e}")

    # ── Inference ──────────────────────────────────────────────────────────────

    def get_action(self, state: tuple) -> int:
        """Greedy action: argmax_a Q(s, a)."""
        si = self._idx.get(state)
        if si is None:
            return 0
        return int(np.argmax(self.Q[si]))

    def get_q_values(self, state: tuple) -> np.ndarray:
        """
        Returns Q(s, ·) as a 1-D array over actions.
        ExecutiveControl uses this for the arbitration comparison.
        """
        si = self._idx.get(state)
        if si is None:
            return np.zeros(N_ACTIONS)
        return self.Q[si].copy()

    def get_convergence_history(self) -> list:
        return self._convergence_history
