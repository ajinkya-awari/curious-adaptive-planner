"""
heuristic_policy.py  —  Greedy Manhattan-distance heuristic policy.

Acts as the "fast, cheap prior" in the arbitration framework — analogous to
the intuitive System 1 in dual-process theory (Kahneman 2011). Corresponds
to π_L: a policy that is usually decent but does no search.

Outputs : none  —  imported by executive_control and experiment
Usage   : from heuristic_policy import HeuristicPolicy
"""

import random

from gridworld import GridWorld, GOAL, ACTION_DELTAS, N_ACTIONS


class HeuristicPolicy:
    """
    Greedy policy: always move toward GOAL using Manhattan distance as proxy.

    Falls back to a random valid action when every move increases distance
    (e.g. stuck in a concave alcove). The fallback is intentionally noisy —
    a perfect heuristic would defeat the purpose of showing planner benefit.
    """

    def __init__(self, env: GridWorld):
        self.env = env

    def get_action(self, state: tuple) -> int:
        """
        Returns the action minimising |Δrow| + |Δcol| to GOAL.
        Ties broken randomly so the heuristic doesn't wedge into a fixed
        loop when two directions are equidistant.
        """
        r,  c  = state
        gr, gc = GOAL

        best_dist = float('inf')
        best_acts = []

        for a, (dr, dc) in ACTION_DELTAS.items():
            nr, nc = r + dr, c + dc
            if not (0 <= nr < 8 and 0 <= nc < 8) or not self.env.grid[nr][nc]:
                continue
            dist = abs(nr - gr) + abs(nc - gc)
            if dist < best_dist:
                best_dist = dist
                best_acts = [a]
            elif dist == best_dist:
                best_acts.append(a)

        if best_acts:
            return random.choice(best_acts)

        # completely boxed in (shouldn't happen on well-formed maps, but be safe)
        valid = self.env.get_valid_actions(state)
        return random.choice(valid) if valid else random.randint(0, N_ACTIONS - 1)
