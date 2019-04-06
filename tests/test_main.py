import numpy as np
from rollout import Step
from data import AdvantageDataset
from config import LunarLander
import pytest
import statistics


@pytest.fixture
def config():
    return LunarLander()


@pytest.fixture
def rollout():
    return [
        [Step(np.random.random(10), 0, 1.0, True), Step(np.random.random(10), 1, 1.0, True),
         Step(np.random.random(10), 1, 1.0, True)],
        [Step(np.random.random(10), 3, 1.0, True), Step(np.random.random(10), 1, 1.0, True),
         Step(np.random.random(10), 1, 1.0, True)],
        [Step(np.random.random(10), 1, 0.0, True), Step(np.random.random(10), 1, 0.0, True),
         Step(np.random.random(10), 8, 1.0, True)]
    ]


def test_advantage_dataset(config, rollout):
    a = AdvantageDataset(rollout, state_transform=config.state_transform, discount_factor=config.discount_factor)

    assert a[0][1] == 0
    assert a[1][1] == 1
    assert a[3][1] == 3
    assert a[8][1] == 8

    all_advantage = []

    for episode in rollout:
        cum_reward = 0.0
        local_steps = []
        for step in reversed(episode):
            cum_reward = step.reward + cum_reward * config.discount_factor
            step.advantage = cum_reward
            local_steps.append(step.advantage)
        local_steps = list(reversed(local_steps))
        all_advantage = all_advantage + local_steps

    mu = statistics.mean(all_advantage)
    sigma = statistics.stdev(all_advantage)

    for i, (state, action, reward, advantage) in enumerate(a):
        test_advantage = (all_advantage[i] - mu) / (sigma + 1e-12)
        assert advantage == test_advantage

