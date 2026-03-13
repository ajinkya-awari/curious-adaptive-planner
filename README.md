# CuriousPlanner: Investigating Curiosity-Augmented Value Iteration with Adaptive Policy Arbitration

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.24%2B-013243?logo=numpy)](https://numpy.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Research Snapshot

This project investigates whether a hybrid agent that delegates most decisions to a cheap heuristic policy, while invoking a costly optimal planner only when confident it will improve outcomes, can reliably match the performance of the optimal planner alone.

**Research question:** Does confidence-threshold arbitration between a fast greedy policy and a deliberative Value Iteration planner preserve near-optimal performance while substantially reducing deliberative computation?

Experiments on a controlled 8x8 GridWorld benchmark show that the hybrid agent matches optimal planner reward (mean -3.00) while invoking deliberative planning on fewer than 1 in 14 decision steps.

---

## Project Overview

This repository implements a modular reinforcement learning system composed of five components: a GridWorld environment, a greedy heuristic policy, a synchronous Value Iteration planner, a count-based curiosity engine, and a confidence-threshold Executive Control module that arbitrates between the heuristic and the planner at each timestep.

The system is built entirely from scratch in Python using only NumPy, with no RL libraries, so that every algorithmic decision is transparent and traceable. The primary goal is to study how the interaction between these components affects agent behaviour - specifically exploration coverage, convergence properties, and the computational cost of near-optimal decision-making.

---

## Research Questions

1. **Arbitration efficiency:** Does confidence-threshold arbitration between a fast heuristic and a slow optimal planner preserve the planner's performance while substantially reducing how often the planner is invoked?

2. **Curiosity and coverage:** Does an information-gain curiosity signal (surprise = -log P_model) drive more systematic state-space coverage compared to pure exploitation, and does this coverage grow monotonically over episodes?

3. **Convergence behaviour:** Does synchronous Value Iteration on a small finite MDP exhibit the geometric convergence rate expected from the Bellman contraction property, and at what sweep does it reach practical convergence?

---

## Methodology

### Environment

A deterministic 8x8 GridWorld MDP with approximately 20% randomly placed obstacles (seed=42). The agent starts at (0,0) and must reach (7,7). Rewards are +10 for reaching the goal, -1 per step, and -5 for hitting a wall. All three experimental conditions share the same obstacle layout for a fair comparison.

### Heuristic Policy

A greedy Manhattan-distance policy that always moves toward the goal. This serves as the fast, cheap prior - directionally sensible but incapable of lookahead or obstacle reasoning beyond one step.

### Deliberative Planner

Synchronous Value Iteration over the full state-action space. The Bellman backup is applied iteratively until the sup-norm change between successive Q-tables drops below a threshold of 1e-4. Convergence is tracked per sweep to observe the empirical decay rate.

### Motivation Engine

A count-based surprise estimator. The intrinsic reward for a transition (s, a, s') is:

```
r_int = -log( N[s][a][s'] / sum_s'' N[s][a][s''] + epsilon )
```

This is added to the extrinsic reward with weight beta = 0.1. The signal is high for rarely-visited transitions and decays naturally as the agent accumulates experience - no manual annealing schedule is required.

### Executive Control

A threshold arbiter that selects between the two policies at each step:

```
       | planner action    if Q(s, a_planner) > Q(s, a_heuristic) + 0.5
a_t = <
       | heuristic action  otherwise
```

The planner's own Q-values are used to evaluate both actions. The planner overrides the heuristic only when it has genuine confidence in a better choice.

### Experimental Design

Three conditions were compared over 500 episodes each on the same GridWorld map:

- **Heuristic Only** - greedy Manhattan policy, no planning
- **Planner Only** - pure Value Iteration policy, no curiosity
- **NCSF Full** - arbitration plus curiosity (the proposed hybrid)

Value Iteration was run once before episodes began and the resulting Q-table was shared across conditions.

---

## Experimental Findings

**Convergence:** Value Iteration converged at sweep 181 out of a maximum of 1000. The sup-norm delta decayed from approximately 15.0 at sweep 1 to 9.78e-5 at termination. The decay is approximately log-linear, consistent with the geometric rate expected from the Bellman contraction property at discount factor gamma = 0.95.

**Arbitration usage:** Over 500 episodes, the Executive Control module selected the planner on an average of 7.1% of steps. This fraction remained relatively stable across episodes, suggesting that the structure of which states favour the planner is consistent once Value Iteration has converged.

**Exploration coverage:** The NCSF agent visited 56 unique (state, action) pairs by the end of 500 episodes. Coverage grew monotonically with no plateau, indicating that the curiosity signal continued directing the agent toward less-visited transitions throughout training.

**Reward comparison:** The heuristic achieves a mean episode reward of -8.30 despite reaching the goal every episode, because its reactive obstacle-avoidance produces path-inefficient trajectories. Both the Planner Only and NCSF Full conditions achieve -3.00, showing that the arbitration mechanism is sufficient to recover the planner's optimal routing without invoking it on every step.

---

## Results

| Condition | Goal Rate | Mean Episode Reward | Planner Usage |
|-----------|-----------|---------------------|---------------|
| Heuristic Only | 100% | -8.30 | 0% |
| Planner Only | 100% | -3.00 | 100% |
| NCSF Full | 100% | -3.00 | 7.1% |

The central observation is that the NCSF agent matches optimal planner performance while delegating the vast majority of decisions to the cheap heuristic. The 5.3-unit reward gap between the heuristic and NCSF reflects the planner's ability to find shorter, obstacle-aware paths - a capability the greedy heuristic cannot reproduce without lookahead.

### Figures

**Learning Curves - All Three Conditions**

![Learning Curves](results/figures/01_learning_curves.png)

**Value Iteration Convergence**

![Convergence](results/figures/02_convergence.png)

**Executive Control Arbitration Ratio**

![Planner Usage](results/figures/03_planner_usage.png)

**Path Comparison: Heuristic vs NCSF**

![Path Comparison](results/figures/04_path_comparison.png)

**Exploration Coverage Under Curiosity Bonus**

![Exploration Coverage](results/figures/05_exploration_coverage.png)

---

## Repository Structure

```
curious-adaptive-planner/
|
+-- gridworld.py            # MDP environment: grid construction, step, transition
+-- heuristic_policy.py     # Greedy Manhattan-distance prior policy
+-- deliberative_planner.py # Synchronous Value Iteration with convergence logging
+-- motivation_engine.py    # Count-based curiosity: surprise = -log P_model
+-- executive_control.py    # Confidence-threshold arbitration and usage tracking
|
+-- experiment.py           # Three-condition 500-episode comparison
+-- visualize.py            # Generates five diagnostic figures from saved results
+-- run_all.py              # Runs experiment then visualize in sequence
|
+-- results/
|   +-- figures/            # PNG diagnostic plots (committed)
|
+-- requirements.txt
+-- LICENSE
+-- README.md
```

Each module corresponds to a single conceptual component so that individual pieces can be studied, modified, or replaced independently. The heuristic policy, for instance, can be swapped for any callable that maps a state to an action without changing the rest of the system.

---

## Research Context

This project was built to investigate a specific question about computational efficiency in sequential decision-making: under what conditions can a cheap approximate policy substitute for an expensive optimal one, and how large is the resulting performance cost? The arbitration mechanism studied here is a minimal version of dual-process ideas that appear in cognitive science and have been explored in the RL literature under the framing of bounded optimality (Russell 1995).

The curiosity component was included to examine whether a simple count-based surprise signal is sufficient to drive meaningful exploration in a small tabular environment, without requiring the parametric world models used in more complex settings such as VIME (Houthooft et al. 2016) or prediction-error curiosity (Pathak et al. 2017).

The deliberate choice to use no RL libraries reflects a preference for transparency over convenience: every update rule, every convergence check, and every arbitration decision is directly visible in the source code and traceable to a specific algorithmic choice.

---

## Setup and Usage

### Requirements

- Python 3.9 or newer
- NumPy 1.24+
- Matplotlib 3.7+

### Installation

```
git clone https://github.com/ajinkya-awari/curious-adaptive-planner.git
cd curious-adaptive-planner
pip install -r requirements.txt
```

### Run Full Pipeline

```
python run_all.py
```

Runs the three-condition experiment and generates all five figures. Total runtime under 10 seconds.

---

## References

1. Sutton, R. S., & Barto, A. G. (2018). Reinforcement Learning: An Introduction (2nd ed.). MIT Press.
2. Bellman, R. (1957). Dynamic Programming. Princeton University Press.
3. Russell, S. (1995). Rationality and Intelligence. Artificial Intelligence, 94(1-2), 57-77.
4. Houthooft, R., et al. (2016). VIME: Variational Information Maximizing Exploration. NeurIPS.
5. Pathak, D., et al. (2017). Curiosity-Driven Exploration by Self-Supervised Prediction. ICML.
6. Strehl, A. L., & Littman, M. L. (2008). An Analysis of Model-Based Interval Estimation for MDPs. JCSS, 74(8).
7. Kahneman, D. (2011). Thinking, Fast and Slow. Farrar, Straus and Giroux.

---

## Citation

```bibtex
@misc{awari2025curiousplanner,
  author    = {Awari, Ajinkya},
  title     = {{CuriousPlanner}: Investigating Curiosity-Augmented Value Iteration
               with Adaptive Policy Arbitration},
  year      = {2025},
  publisher = {GitHub},
  url       = {https://github.com/ajinkya-awari/curious-adaptive-planner}
}
```

---

