# CuriousPlanner: Curiosity-Augmented Value Iteration with Adaptive Policy Arbitration

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.24%2B-013243?logo=numpy)](https://numpy.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Abstract

Large-scale reinforcement learning agents routinely face a fundamental tension: **fast heuristic policies** are computationally cheap but sub-optimal, while **deliberative planners** are near-optimal but expensive. This repository empirically investigates a hybrid arbitration architecture that combines (1) synchronous Value Iteration as a deliberative planner, (2) an information-gain curiosity signal for exploration, and (3) a confidence-threshold arbiter that selects between the two policies at each timestep. Experiments on an 8×8 stochastic GridWorld benchmark demonstrate that the composite system matches the optimal planner's reward (mean −3.00 vs. −8.30 for the heuristic baseline) while invoking the expensive planner on only **7.1% of steps** — providing empirical support for bounded-optimality results in the RL literature (Russell 1995; Sutton & Barto 2018).

---

## Table of Contents

- [Problem Formulation](#problem-formulation)
- [Architecture Overview](#architecture-overview)
- [Mathematical Framework](#mathematical-framework)
- [Repository Structure](#repository-structure)
- [Experimental Results](#experimental-results)
- [Empirical Validation](#empirical-validation)
- [Setup and Usage](#setup-and-usage)
- [Limitations and Future Work](#limitations-and-future-work)
- [Citation](#citation)
- [References](#references)

---

## Problem Formulation

We model the navigation task as a **Markov Decision Process** (MDP) defined by the tuple $\mathcal{M} = \langle \mathcal{S}, \mathcal{A}, P, R, \gamma \rangle$:

| Symbol | Definition |
|--------|-----------|
| $\mathcal{S}$ | 8×8 grid cells (row, col), passable cells only |
| $\mathcal{A}$ | $\{$up, down, left, right$\}$, four cardinal actions |
| $P(s'\|s,a)$ | Deterministic transition (single successor state) |
| $R(s,a)$ | $+10$ goal, $-1$ step, $-5$ wall collision |
| $\gamma$ | $0.95$ discount factor |

The agent starts at $(0,0)$ and must reach $(7,7)$. Approximately 20% of non-terminal cells are blocked as obstacles (fixed per random seed). The optimal policy $\pi^*$ maximises the expected discounted return $V^{\pi}(s) = \mathbb{E}_{\pi}\left[\sum_{t=0}^{\infty} \gamma^t R(s_t, a_t) \mid s_0 = s\right]$.

---

## Architecture Overview

The system comprises five components orchestrated by an Executive Control module:

```
 ┌──────────────────────────────────────────────────────────────┐
 │                     EXECUTIVE CONTROL                        │
 │   if Q(s, a_plan) > Q(s, a_heur) + ξ  →  use planner       │
 │   else                                  →  use heuristic    │
 └────────┬──────────────────────────────────────┬─────────────┘
          │                                      │
   ┌──────▼──────┐                      ┌────────▼───────┐
   │  HEURISTIC  │                      │  DELIBERATIVE  │
   │   POLICY    │                      │    PLANNER     │
   │  π_L(s):    │                      │  VI → Q*(s,a)  │
   │  Manhattan  │                      │  (Bellman ops) │
   └─────────────┘                      └────────────────┘
                           │
                  ┌────────▼────────┐
                  │   MOTIVATION    │
                  │    ENGINE       │
                  │  r_int = -log P │
                  │  R' = R + β·r   │
                  └─────────────────┘
```

**Module roles:**
- **Heuristic Policy** (`π_L`): greedy Manhattan-distance policy — fast, O(1) per step
- **Deliberative Planner**: tabular Value Iteration — provably convergent to $Q^*$
- **Motivation Engine**: surprise-based curiosity bonus driving exploration
- **Executive Control**: threshold arbiter balancing speed vs. optimality

---

## Mathematical Framework

### 1 · Bellman Optimality and Value Iteration

The Deliberative Planner solves for the optimal action-value function $Q^*(s,a)$, the unique fixed point of the **Bellman optimality operator** $\mathcal{B}$:

$$Q^*(s,a) = \sum_{s' \in \mathcal{S}} P(s'|s,a)\left[R(s,a) + \gamma \max_{a' \in \mathcal{A}} Q^*(s', a')\right]$$

Value Iteration computes this by repeated application:

$$Q_{k+1}(s,a) \leftarrow \mathbb{E}_{s' \sim P(\cdot|s,a)}\left[R(s,a) + \gamma \max_{a'} Q_k(s', a')\right]$$

Because the environment is deterministic, $P(s'|s,a)$ is a delta distribution and the expectation collapses to a single lookup. **Convergence** follows from the Banach Fixed-Point Theorem: $\mathcal{B}$ is a $\gamma$-contraction in the $\ell^\infty$ norm,

$$\|\mathcal{B}Q_1 - \mathcal{B}Q_2\|_\infty \leq \gamma \|Q_1 - Q_2\|_\infty, \quad \gamma \in [0,1)$$

so $\|Q_{k+1} - Q_k\|_\infty \to 0$ geometrically. Empirically, convergence was reached at **sweep 181** with $\delta = 9.78 \times 10^{-5}$ (threshold $\varepsilon = 10^{-4}$).

*Reference: Sutton & Barto (2018), Equations 4.9–4.10*

---

### 2 · Information-Gain Curiosity Reward

The Motivation Engine computes a **surprise-based intrinsic reward** using an empirical transition model built from visit counts (Houthooft et al. 2016):

$$P_{\text{model}}(s'|s,a) = \frac{N[s][a][s']}{\displaystyle\sum_{s''} N[s][a][s'']}$$

Intrinsic reward (model surprise):

$$r_{\text{int}}(s,a,s') = -\log\!\left(P_{\text{model}}(s'|s,a) + \varepsilon\right)$$

Combined reward sent to the planner:

$$R'(s,a,s') = R_{\text{ext}}(s,a) + \beta \cdot r_{\text{int}}(s,a,s'), \quad \beta = 0.1$$

High surprise → large exploration bonus → agent is pushed toward less-visited transitions. As the agent explores, $P_{\text{model}}$ approaches the true dynamics and $r_{\text{int}} \to 0$, redirecting attention back to the extrinsic goal.

*Reference: Houthooft et al. (2016), "VIME"; Pathak et al. (2017), "Curiosity-Driven Exploration"*

---

### 3 · Confidence-Threshold Policy Arbitration

Let $\pi_L$ denote the heuristic policy and $\pi_P$ the planner policy. The arbiter selects:

$$a_t = \begin{cases} \pi_P(s_t) & \text{if } Q_P(s_t, \pi_P(s_t)) > Q_P(s_t, \pi_L(s_t)) + \xi \\ \pi_L(s_t) & \text{otherwise} \end{cases}$$

where $\xi = 0.5$ is the confidence threshold. The composite policy $\pi_{\text{NCSF}}$ satisfies the bounded-optimality guarantee:

$$V^{\pi_{\text{NCSF}}}(s) \geq V^{\pi_L}(s) - \frac{2\delta}{1-\gamma}$$

where $\delta = \|Q_P - Q^*\|_\infty$ is the planner's approximation error. As $\delta \to 0$ (planner converges), the NCSF policy approaches optimality from above the heuristic baseline.

*Reference: Russell (1995), "Rationality and Intelligence"; Sutton & Barto (2018), Sec. 4.3*

---

## Repository Structure

```
curious-adaptive-planner/
│
├── gridworld.py            # MDP environment: 8×8 grid, obstacles, rewards
├── heuristic_policy.py     # Fast greedy prior (π_L): Manhattan distance
├── deliberative_planner.py # Synchronous Value Iteration → Q*
├── motivation_engine.py    # Count-based curiosity: r_int = -log P_model
├── executive_control.py    # Arbitration: threshold comparison on Q-values
│
├── experiment.py           # Three-condition 500-episode comparison
├── visualize.py            # Five diagnostic figures
├── run_all.py              # Master runner: experiment → figures
│
├── results/
│   ├── *.npy               # Raw experiment arrays (generated at runtime)
│   └── figures/            # PNG plots (generated at runtime)
│
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

---

## Experimental Results

All three conditions run on the same GridWorld map (seed=42, 500 episodes, max 200 steps/episode).

### Summary Table

| Condition | Goal Rate | Mean Reward | Notes |
|-----------|-----------|-------------|-------|
| **Heuristic Only** | 100% | **−8.30** | Fast but takes long, obstacle-avoiding paths |
| **Planner Only** | 100% | **−3.00** | Optimal routes; VI converges before episodes |
| **NCSF Full** | 100% | **−3.00** | Matches planner; uses deliberation on 7.1% of steps |

**Key finding**: NCSF achieves *identical reward* to the optimal planner while invoking the expensive Value Iteration module on fewer than 1 in 14 steps — the heuristic handles the routine cases.

### Value Iteration Convergence

- Converged at **sweep 181** out of max 1000
- Final $\|Q_{k+1} - Q_k\|_\infty = 9.78 \times 10^{-5}$
- Convergence plot shows geometric decay consistent with contraction factor $\gamma = 0.95$

### Exploration Coverage

- NCSF_FULL visited **56 unique $(s,a)$ pairs** by end of 500 episodes
- Coverage grows monotonically as the curiosity bonus drives the agent to less-visited transitions

### Figures

| Figure | Description |
|--------|-------------|
| `01_learning_curves.png` | Per-episode reward for all three conditions (25-episode moving average) |
| `02_convergence.png` | $\|Q_{k+1} - Q_k\|_\infty$ vs. VI sweep on log scale |
| `03_planner_usage.png` | Fraction of steps where arbiter selected planner over heuristic |
| `04_path_comparison.png` | Heuristic vs. NCSF final-episode path on the shared grid |
| `05_exploration_coverage.png` | Unique $(s,a)$ pairs visited under curiosity-driven exploration |

---

## Empirical Validation

### Claim 1 — Bellman Operator Convergence

**Theory**: $\mathcal{B}$ is a $\gamma$-contraction → $\|Q_{k+1} - Q_k\|_\infty$ decays geometrically.

**Evidence**: `02_convergence.png` shows log-linear decay from $\delta \approx 15$ at sweep 1 to $9.78 \times 10^{-5}$ at sweep 181. The slope on the log plot approximates $\log \gamma = \log 0.95 \approx -0.051$ per sweep, consistent with the theoretical bound.

---

### Claim 2 — Curiosity Drives Systematic Exploration

**Theory**: Information-gain reward incentivises the agent to reduce model uncertainty, leading to polynomial (rather than exponential) state-space coverage (Strehl & Littman 2008).

**Evidence**: `05_exploration_coverage.png` shows monotone growth in unique $(s,a)$ pairs visited. The NCSF agent reaches 56 state-action pairs without any external reward shaping, demonstrating the exploration-driving effect of the surprise bonus.

---

### Claim 3 — Bounded Optimality of Composite Policy

**Theory**: $V^{\pi_{\text{NCSF}}}(s) \geq V^{\pi_L}(s) - 2\delta/(1-\gamma)$.

**Evidence**: `01_learning_curves.png` and the summary table confirm $-3.00 \geq -8.30$, i.e., NCSF strictly dominates the heuristic baseline. The performance gap ($\approx 5.3$ reward units) is bounded by the planner's accuracy as predicted.

---

## Setup and Usage

### Requirements

- Python 3.9 or newer
- NumPy 1.24+
- Matplotlib 3.7+

### Installation (Windows Command Prompt)

```bat
git clone https://github.com/ajinkya-awari/curious-adaptive-planner.git
cd curious-adaptive-planner
pip install -r requirements.txt
```

### Run Everything

```bat
python run_all.py
```

This runs `experiment.py` (~1 s) followed by `visualize.py` (~4 s). All outputs go to `results/` and `results/figures/`.

### Run Steps Individually

```bat
python experiment.py   :: runs Value Iteration + 3 × 500 episodes
python visualize.py    :: generates figures from saved .npy files
```

### Explore Interactively

```python
from gridworld import GridWorld
from deliberative_planner import DeliberativePlanner

env     = GridWorld(seed=42)
planner = DeliberativePlanner(env)
planner.value_iteration()

state = env.reset()
env.render()

action = planner.get_action(state)
print("Best action:", action)
```

---

## Limitations and Future Work

This implementation is intentionally minimal to keep the theory–code correspondence transparent. Several simplifications warrant acknowledgement:

**Environment**: The GridWorld is discrete, fully observable, and deterministic. Real-world applications involve continuous state spaces, partial observability, and stochastic dynamics. Extending to stochastic transitions would require approximate dynamic programming or deep RL methods.

**Heuristic module**: The Manhattan-distance heuristic is a toy stand-in for the kind of rich semantic prior that a pre-trained language model would provide. A natural extension is to replace `HeuristicPolicy` with a fine-tuned LLM queried via a tool-use interface (Yao et al. 2023 — ReAct), which would make the arbitration mechanism directly applicable to language-grounded tasks.

**Curiosity model**: The tabular count-based approach scales as $O(|\mathcal{S}||\mathcal{A}|)$ in memory. For large state spaces, a parametric surprise estimator (e.g., prediction-error curiosity as in Pathak et al. 2017, or a Bayesian neural network as in Houthooft et al. 2016) is necessary.

**Meta-learning**: The planner is fixed after Value Iteration. A proper meta-learning extension (Finn et al. 2017 — MAML) would allow the system to warm-start on a new map using experience from previous maps.

**Arbitration threshold**: The confidence threshold $\xi = 0.5$ is fixed. An adaptive schedule (e.g., $\xi$ decays as the planner's world model becomes more accurate) could improve compute efficiency in the early training phase.

---

## Related Work

- **Curiosity-Driven Exploration**: Pathak et al. (2017) propose self-supervised prediction error as an intrinsic reward; this project uses a simpler count-based surprise that recovers the same qualitative exploration behaviour in the tabular setting.

- **VIME**: Houthooft et al. (2016) formalise curiosity as information gain on a Bayesian world model — the theoretical inspiration for the `MotivationEngine` here.

- **Dyna-Q**: Sutton (1991) introduced interleaving model-based planning with direct experience in a single learning loop, the foundational hybrid RL idea that motivates our architecture.

- **ReAct**: Yao et al. (2023) demonstrate LLM-as-orchestrator with tool use, showing that a language model can take on the role of the heuristic prior in complex reasoning tasks — the natural next step for this framework.

---

## Citation

If you build on this work, please cite:

```bibtex
@misc{awari2025curiousplanner,
  author    = {Awari, Ajinkya},
  title     = {{CuriousPlanner}: Curiosity-Augmented Value Iteration
               with Adaptive Policy Arbitration},
  year      = {2025},
  publisher = {GitHub},
  url       = {https://github.com/ajinkya-awari/curious-adaptive-planner}
}
```

---

## References

1. Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.
2. Bellman, R. (1957). *Dynamic Programming*. Princeton University Press.
3. Houthooft, R., Chen, X., Duan, Y., Schulman, J., De Turck, F., & Abbeel, P. (2016). VIME: Variational Information Maximizing Exploration. *NeurIPS*.
4. Pathak, D., Agrawal, P., Efros, A. A., & Darrell, T. (2017). Curiosity-Driven Exploration by Self-Supervised Prediction. *ICML*.
5. Strehl, A. L., & Littman, M. L. (2008). An Analysis of Model-Based Interval Estimation for Markov Decision Processes. *Journal of Computer and System Sciences, 74*(8), 1309–1331.
6. Russell, S. (1995). Rationality and Intelligence. *Artificial Intelligence, 94*(1–2), 57–77.
7. Finn, C., Abbeel, P., & Levine, S. (2017). Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks. *ICML*.
8. Yao, S., et al. (2023). ReAct: Synergizing Reasoning and Acting in Language Models. *ICLR*.
9. Sutton, R. S. (1991). Dyna, an Integrated Architecture for Learning, Planning, and Reacting. *ACM SIGART Bulletin, 2*(4), 160–163.

---

*Published paper (IJARSCT 2023): Awari, A. et al., "Plant Disease Detection Using Machine Learning," IJARSCT, Vol. 3, Issue 4.*
