"""The reinforcement-learning operator policy: a UCB1 bandit.

One arm per operator.  During evolution every operator choice (fresh trees,
mutations) is an action; the reward is the resulting offspring's fitness
improvement, clipped to [0, 1].  UCB1 balances trying every operator early
against exploiting the ones that keep producing fitter features on this
particular dataset.
"""

from __future__ import annotations

import math

import numpy as np


class OperatorBandit:
    def __init__(self, arms: list[str], ucb_c: float = 1.4):
        self.ucb_c = ucb_c
        self.pulls: dict[str, int] = {a: 0 for a in arms}
        self.mean_reward: dict[str, float] = {a: 0.0 for a in arms}
        self.total_pulls = 0

    def choose(self, rng: np.random.Generator, valid: list[str]) -> str:
        """Pick an operator among `valid` by UCB1; unpulled arms first."""
        unpulled = [a for a in valid if self.pulls.get(a, 0) == 0]
        if unpulled:
            return unpulled[int(rng.integers(len(unpulled)))]
        log_total = math.log(max(self.total_pulls, 1))
        best, best_score = valid[0], -math.inf
        for a in valid:
            score = self.mean_reward[a] + self.ucb_c * math.sqrt(log_total / self.pulls[a])
            if score > best_score:
                best, best_score = a, score
        return best

    def update(self, arm: str, reward: float) -> None:
        reward = min(max(float(reward), 0.0), 1.0)
        if arm not in self.pulls:
            self.pulls[arm] = 0
            self.mean_reward[arm] = 0.0
        self.pulls[arm] += 1
        self.total_pulls += 1
        n = self.pulls[arm]
        self.mean_reward[arm] += (reward - self.mean_reward[arm]) / n

    def stats(self) -> dict[str, dict[str, float]]:
        return {
            a: {"pulls": self.pulls[a], "mean_reward": round(self.mean_reward[a], 4)}
            for a in sorted(self.pulls, key=lambda a: -self.mean_reward[a])
        }


class UniformPolicy:
    """Ablation / rl_policy=False: uniform random operator choice."""

    def choose(self, rng: np.random.Generator, valid: list[str]) -> str:
        return valid[int(rng.integers(len(valid)))]

    def update(self, arm: str, reward: float) -> None:
        pass

    def stats(self) -> dict:
        return {}
