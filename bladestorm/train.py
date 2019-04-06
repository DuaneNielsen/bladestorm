import ray
import config
from models.fully_connected import PPOWrap
import os
import gym
from rollout import single_episode


@ray.remote
class ExperienceEnv(object):
    def __init__(self, config):
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        self.config = config
        self.policy = PPOWrap(config.features, config.action_map, config.hidden)
        self.env = gym.make(config.gym_env_string)

    def rollout(self, policy_weights, num_episodes=2):
        self.policy.load_state_dict(policy_weights)

        rollout = []
        for _ in range(num_episodes):
            rollout.append(single_episode(self.env, self.config, self.policy))
        return rollout


if __name__ == "__main__":
    ray.init()

    config = config.LunarLander()
    config.experience_threads = 2

    policy = PPOWrap(config.features, config.action_map, config.hidden)
    policy_weights = policy.state_dict()

    experience = []
    gatherers = [ExperienceEnv.remote(config) for _ in range(config.experience_threads)]

    for i in range(config.experience_threads):
        experience.append(gatherers[i].rollout.remote(policy_weights))

    rollout = []

    for i in range(config.experience_threads):
        ready, waiting = ray.wait(experience)
        rollout = rollout + ray.get(ready[0])
