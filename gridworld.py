"""
gridworld.py  —  8×8 deterministic GridWorld for tabular RL benchmarking.

Agent starts at (0,0), goal at (7,7). About 20% of non-terminal cells are
blocked as obstacles. State space is (row, col) tuples; action space is
{0=up, 1=down, 2=left, 3=right}.

Outputs : none  —  imported by all other modules
Usage   : from gridworld import GridWorld, GRID_SIZE, N_ACTIONS
"""

import numpy as np

# ── Shared constants ───────────────────────────────────────────────────────────

GRID_SIZE     = 8
N_ACTIONS     = 4
OBSTACLE_FRAC = 0.20

REWARD_GOAL   =  10.0
REWARD_STEP   =  -1.0
REWARD_WALL   =  -5.0    # hitting a wall or out-of-bounds cell

# action index → (Δrow, Δcol)
ACTION_DELTAS = {
    0: (-1,  0),   # up
    1: ( 1,  0),   # down
    2: ( 0, -1),   # left
    3: ( 0,  1),   # right
}

START = (0, 0)
GOAL  = (GRID_SIZE - 1, GRID_SIZE - 1)


class GridWorld:
    """
    Deterministic 8×8 grid navigation task.

    Obstacles are fixed per seed so experiments are reproducible.
    start and goal cells are always passable regardless of obstacle placement.
    """

    def __init__(self, seed: int = 42):
        self.rng   = np.random.default_rng(seed)
        self.grid  = self._build_grid()   # True = passable, False = wall
        self.state = START
        self.done  = False

    # ── Setup ──────────────────────────────────────────────────────────────────

    def _build_grid(self) -> np.ndarray:
        passable = np.ones((GRID_SIZE, GRID_SIZE), dtype=bool)

        candidates = [
            (r, c)
            for r in range(GRID_SIZE)
            for c in range(GRID_SIZE)
            if (r, c) not in (START, GOAL)
        ]
        n_obstacles = int(len(candidates) * OBSTACLE_FRAC)
        chosen_idxs = self.rng.choice(len(candidates), size=n_obstacles, replace=False)

        for idx in chosen_idxs:
            r, c = candidates[idx]
            passable[r][c] = False

        return passable

    # ── Core interface ─────────────────────────────────────────────────────────

    def reset(self) -> tuple:
        self.state = START
        self.done  = False
        return self.state

    def step(self, action: int) -> tuple:
        """Returns (next_state, reward, done)."""
        if self.done:
            raise RuntimeError("episode over — call reset() first")

        dr, dc  = ACTION_DELTAS[action]
        r,  c   = self.state
        nr, nc  = r + dr, c + dc

        # out-of-bounds or obstacle → stay put, pay the wall penalty
        if not (0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE) or not self.grid[nr][nc]:
            reward     = REWARD_WALL
            next_state = self.state
        else:
            next_state = (nr, nc)
            reward     = REWARD_GOAL if next_state == GOAL else REWARD_STEP

        self.state = next_state
        self.done  = (next_state == GOAL)
        return next_state, reward, self.done

    def transition(self, state: tuple, action: int) -> tuple:
        """
        Pure (non-mutating) transition lookup — used by the planner.
        Returns (next_state, reward).
        """
        dr, dc = ACTION_DELTAS[action]
        r,  c  = state
        nr, nc = r + dr, c + dc

        if not (0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE) or not self.grid[nr][nc]:
            return state, REWARD_WALL

        next_state = (nr, nc)
        reward     = REWARD_GOAL if next_state == GOAL else REWARD_STEP
        return next_state, reward

    # ── Utility ────────────────────────────────────────────────────────────────

    def get_all_states(self) -> list:
        """Every passable cell — the full finite state space S."""
        return [
            (r, c)
            for r in range(GRID_SIZE)
            for c in range(GRID_SIZE)
            if self.grid[r][c]
        ]

    def get_valid_actions(self, state: tuple) -> list:
        """Actions that don't immediately hit a wall — useful for the heuristic."""
        r, c  = state
        valid = []
        for a, (dr, dc) in ACTION_DELTAS.items():
            nr, nc = r + dr, c + dc
            if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE and self.grid[nr][nc]:
                valid.append(a)
        return valid

    def render(self) -> None:
        """ASCII render — handy for quick sanity checks."""
        for r in range(GRID_SIZE):
            row = ''
            for c in range(GRID_SIZE):
                if   (r, c) == self.state: row += 'A'
                elif (r, c) == GOAL:       row += 'G'
                elif (r, c) == START:      row += 'S'
                elif self.grid[r][c]:      row += '.'
                else:                      row += '#'
            print(row)
        print()
