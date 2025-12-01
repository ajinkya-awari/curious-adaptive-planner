"""
motivation_engine.py  —  Information-gain curiosity reward (tabular version).

Computes surprise-based intrinsic reward using empirical visit counts as
a proxy for the model posterior — a tabular approximation of VIME
(Houthooft et al. 2016, NeurIPS).

Surprise:    r_int(s,a,s') = -log( P_model(s'|s,a) + ε )
Total:       R'(s,a,s')    = R_ext + β · r_int

High surprise → agent hasn't seen this transition much → exploration bonus.
As the agent explores, surprise drops → the agent focuses on extrinsic reward.

Outputs : none  —  imported by experiment
Usage   : from motivation_engine import MotivationEngine
"""

import math
from collections import defaultdict

from gridworld import GRID_SIZE


EPSILON = 1e-6   # prevents log(0); -log(ε) ≈ 13.8 so the cap below is sensible
BETA    = 0.10   # curiosity weight — small so it steers without overwhelming R_ext
                 # TODO: try exponential decay  β(t) = β₀ · 0.999^t

# hard cap so early-episode bonuses don't blow up the reward signal
SURPRISE_CAP = 15.0


class MotivationEngine:
    """
    Tabular surprise estimator using empirical transition counts.

    P_model(s'|s,a) = N[s][a][s'] / Σ_{s''} N[s][a][s'']

    This is the simplest possible implementation of information-gain curiosity.
    Houthooft et al. (2016) use a Bayesian neural network for the model — the
    count-based version here converges to the same thing in the tabular setting.

    Reference: Houthooft et al. (2016), "VIME: Variational Information
               Maximizing Exploration", NeurIPS.
               Also related to Strehl & Littman (2008) E³-style exploration.
    """

    def __init__(self):
        # counts[s_key][action][s'_key] → integer visit count
        self._counts: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

    # ── Helpers ────────────────────────────────────────────────────────────────

    @staticmethod
    def _key(state: tuple) -> int:
        """Flatten (row,col) to a single int — keeps dict keys hashable and fast."""
        r, c = state
        return r * GRID_SIZE + c

    # ── Public interface ───────────────────────────────────────────────────────

    def update(self, state: tuple, action: int, next_state: tuple) -> None:
        """Log an observed transition. Call this after env.step(), every step."""
        s  = self._key(state)
        sp = self._key(next_state)
        self._counts[s][action][sp] += 1

    def compute_reward(self, state: tuple, action: int,
                       next_state: tuple, r_ext: float) -> float:
        """
        Returns R'(s,a,s') = R_ext + β · r_int.

        r_int = -log(P_model(s'|s,a) + ε)
        High when the model hasn't seen this transition → encourages exploration.
        Low (near 0 after log) when the transition is well-understood.
        """
        s  = self._key(state)
        sp = self._key(next_state)

        total = sum(self._counts[s][action].values())

        if total == 0:
            p_model = 0.0   # first time visiting (s,a) → maximum surprise
        else:
            p_model = self._counts[s][action][sp] / total

        r_int = min(-math.log(p_model + EPSILON), SURPRISE_CAP)
        return r_ext + BETA * r_int

    def get_coverage(self) -> int:
        """
        Unique (state, action) pairs visited — a proxy for how much of the
        MDP the agent has explored. Useful for verifying curiosity's effect.
        """
        return sum(len(actions) for actions in self._counts.values())
