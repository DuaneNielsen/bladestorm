import os
import uuid

import gym
import ray

from messages import RedisTransport, EpisodeMessage
from models.fully_connected import PPOWrap
from rollout import single_episode


@ray.remote(num_cpus=1)
class ExperienceEnv(object):
    def __init__(self, config):
        print(ray.services.get_node_ip_address())
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        self.config = config
        self.policy = PPOWrap(config.features, config.action_map, config.hidden)
        self.env = gym.make(config.gym_env_string)
        self.t = RedisTransport()
        self.uuid = uuid.uuid4()

    @ray.method(num_return_vals=3)
    def rollout(self, policy_weights, num_episodes=2, instr=None):
        if instr is not None:
            instr.worker_start()
        self.policy.load_state_dict(policy_weights)

        rollout = []
        stats = []
        for id in range(num_episodes):
            episode, total_reward = single_episode(self.env, self.config, self.policy)
            rollout.append(episode)
            msg = EpisodeMessage(self.uuid, id, len(episode), total_reward)
            stats.append(Stat(total_reward, len(episode)))
            self.t.publish('rollout', msg)

        if instr is not None:
            instr.worker_return()
        return rollout, stats, instr


class Stat:
    def __init__(self, total_reward, epi_length):
        self.total_reward = total_reward
        self.epi_length = epi_length