"""
executive_control.py  —  Confidence-threshold arbitration between two policies.

Implements the dual-process arbitration rule from bounded-optimality theory
(Russell 1995; Lieder & Griffiths 2020):

    a_planner  = argmax_a  Q(s, a)         # slow, deliberative
    a_heuristic = HeuristicPolicy(s)        # fast, cheap

    if Q(s, a_planner) > Q(s, a_heuristic) + ξ:
        use a_planner
    else:
        use a_heuristic   ← default (saves compute)

The key insight: the composite policy can never be much worse than the heuristic
alone — in the worst case it degrades by 2δ/(1−γ) where δ is planner error.
(This is the bounded-optimality guarantee — see Sutton & Barto 2018, Sec. 4.3.)

Outputs : none  —  imported by experiment
Usage   : from executive_control import ExecutiveControl
"""

from gridworld import GridWorld
from heuristic_policy import HeuristicPolicy
from deliberative_planner import DeliberativePlanner


XI_DEFAULT = 0.5   # confidence threshold ξ — raise to make planner more conservative
                   # TODO: make ξ adaptive — decay as episode count grows and planner
                   #       becomes more trusted


class ExecutiveControl:
    """
    Arbiter that chooses between fast-heuristic and slow-planner actions.

    Implements the arbitration logic as a simple threshold comparison on the
    planner's own Q-values — using Q(s, a_heuristic) as a baseline, so the
    planner only overrides when it has genuine confidence.

    Logging source ('planner' vs 'heuristic') per step lets us plot how
    reliance on the planner shifts over the course of training.
    """

    def __init__(self, env: GridWorld, heuristic: HeuristicPolicy,
                 planner: DeliberativePlanner, xi: float = XI_DEFAULT):
        self.env       = env
        self.heuristic = heuristic
        self.planner   = planner
        self.xi        = xi

        self._n_planner   = 0
        self._n_heuristic = 0

    # ── Action selection ───────────────────────────────────────────────────────

    def select_action(self, state: tuple) -> tuple:
        """
        Returns (action, source_str).

        source_str is 'planner' or 'heuristic' — useful for usage-ratio logging.
        """
        a_h    = self.heuristic.get_action(state)
        a_p    = self.planner.get_action(state)
        q_vals = self.planner.get_q_values(state)

        # evaluate both actions under the planner's model (Algorithm 1)
        vp_planner   = q_vals[a_p]
        vp_heuristic = q_vals[a_h]

        if vp_planner > vp_heuristic + self.xi:
            self._n_planner += 1
            return a_p, 'planner'
        else:
            # heuristic is good enough — don't pay the deliberation cost
            self._n_heuristic += 1
            return a_h, 'heuristic'

    # ── Stats ──────────────────────────────────────────────────────────────────

    def get_usage_ratio(self) -> float:
        """Fraction of steps where planner was selected (0.0–1.0)."""
        total = self._n_planner + self._n_heuristic
        return self._n_planner / total if total > 0 else 0.0

    def reset_stats(self) -> None:
        self._n_planner   = 0
        self._n_heuristic = 0
