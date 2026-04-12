from repro_shiptraj_r1.policy import GaussianVelocityPolicy, grpo_step


def test_grpo_step_runs_and_updates_theta():
    policy = GaussianVelocityPolicy(theta=1.0, sigma=0.001)
    hist = [(120.0, 30.0), (120.01, 30.01)]
    fut = [(120.02, 30.02), (120.03, 30.03)]

    before = policy.theta
    stats = grpo_step(policy, hist, fut, num_samples=6, lr=1e-2)
    assert "mean_reward" in stats
    assert isinstance(policy.theta, float)
    assert policy.theta != before
