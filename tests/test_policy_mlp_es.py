import pytest

np = pytest.importorskip("numpy")

from repro_shiptraj_r1.policy_mlp_es import MLPESPolicy, grpo_es_step


def test_mlp_es_step_runs():
    policy = MLPESPolicy(hidden=4)
    hist = [(120.0, 30.0), (120.01, 30.01)]
    fut = [(120.02, 30.02), (120.03, 30.03)]

    before = policy.w1.copy()
    stats = grpo_es_step(policy, hist, fut, num_samples=8, lr=1e-2, seed=7)
    assert "mean_reward" in stats
    assert (policy.w1 != before).any()
