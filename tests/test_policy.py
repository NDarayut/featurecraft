import numpy as np

from featurecraft.policy import OperatorBandit, UniformPolicy


def test_ucb_explores_every_arm_first():
    rng = np.random.default_rng(0)
    bandit = OperatorBandit(["a", "b", "c"])
    chosen = set()
    for _ in range(3):
        arm = bandit.choose(rng, ["a", "b", "c"])
        chosen.add(arm)
        bandit.update(arm, 0.1)
    assert chosen == {"a", "b", "c"}


def test_ucb_converges_to_best_arm():
    rng = np.random.default_rng(0)
    bandit = OperatorBandit(["good", "bad"], ucb_c=0.5)
    for _ in range(200):
        arm = bandit.choose(rng, ["good", "bad"])
        bandit.update(arm, 0.9 if arm == "good" else 0.05)
    assert bandit.pulls["good"] > bandit.pulls["bad"] * 2
    stats = bandit.stats()
    assert list(stats)[0] == "good"  # sorted by mean reward


def test_reward_clipping():
    bandit = OperatorBandit(["a"])
    bandit.update("a", 5.0)
    bandit.update("a", -3.0)
    assert 0.0 <= bandit.mean_reward["a"] <= 1.0


def test_deterministic_given_seed():
    def run():
        rng = np.random.default_rng(42)
        b = OperatorBandit(["a", "b", "c"])
        picks = []
        for i in range(50):
            arm = b.choose(rng, ["a", "b", "c"])
            picks.append(arm)
            b.update(arm, (i % 3) / 3)
        return picks

    assert run() == run()


def test_uniform_policy():
    rng = np.random.default_rng(0)
    p = UniformPolicy()
    assert p.choose(rng, ["only"]) == "only"
    p.update("only", 1.0)  # no-op
    assert p.stats() == {}
