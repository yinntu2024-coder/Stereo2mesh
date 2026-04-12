from repro_shiptraj_r1.policy_linear import LinearFeaturePolicy, grpo_linear_step


def test_linear_policy_update_runs():
    policy = LinearFeaturePolicy()
    hist = [(120.0, 30.0), (120.01, 30.01)]
    fut = [(120.02, 30.02), (120.03, 30.03)]

    before = [row[:] for row in policy.w]
    stats = grpo_linear_step(policy, hist, fut, num_samples=6, lr=1e-3)

    assert "mean_reward" in stats
    changed = any(abs(policy.w[r][c] - before[r][c]) > 0 for r in range(2) for c in range(3))
    assert changed
