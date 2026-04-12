from repro_shiptraj_r1.colregs import tcpa_dcpa
from repro_shiptraj_r1.metrics import ade, fde
from repro_shiptraj_r1.prompting import parse_answer_trajectory
from repro_shiptraj_r1.rewards import colregs_reward, format_reward


def test_metrics_positive():
    pred = [(120.10, 30.20), (120.15, 30.22)]
    gt = [(120.11, 30.19), (120.14, 30.23)]
    assert ade(pred, gt) > 0
    assert fde(pred, gt) > 0


def test_parse_and_format_reward():
    text = '<think>ok</think><answer>{"trajectory": [[120.1,30.1],[120.2,30.2]]}</answer>'
    traj = parse_answer_trajectory(text)
    assert len(traj) == 2
    assert format_reward(text, expected_len=2) == 1.0


def test_colregs_reward_in_range():
    own = [(120.0, 30.0), (120.001, 30.0), (120.002, 30.0)]
    nei = [(120.002, 30.001), (120.001, 30.001), (120.000, 30.001)]
    tcpa, dcpa = tcpa_dcpa(own, nei)
    assert isinstance(tcpa, float)
    assert dcpa >= 0
    r = colregs_reward(own, nei, safety_dcpa_m=500.0)
    assert 0.0 <= r <= 1.0
