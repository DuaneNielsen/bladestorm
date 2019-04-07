import ray
import config
from models.fully_connected import PPOWrap
import os
import gym
from rollout import single_episode
from data import AdvantageDataset
import torch
from torch.utils.data import DataLoader
import math
from messages import EpisodeMessage, RedisTransport, Server, StopAllMessage, KillMessage
import uuid


class TensorBoardListener(Server):
    def __init__(self, transport, config):
        super().__init__(transport, 'rollout')
        self.config = config
        self.register(EpisodeMessage, self.episode)
        self.register(StopAllMessage, self.stopAll)
        self.tb_step = 0
        self.tb = self.config.getSummaryWriter(config.gym_env_string)

    def episode(self, msg):
        self.tb.add_scalar('reward', msg.total_reward, self.tb_step)
        self.tb.add_scalar('epi_len', msg.steps, self.tb_step)
        self.tb_step += 1

    def stopAll(self, msg):
        super(msg)
        self.tb_step = 0
        self.tb = self.config.getSummaryWriter(config.gym_env_string)


@ray.remote
class ExperienceEnv(object):
    def __init__(self, config):
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        self.config = config
        self.policy = PPOWrap(config.features, config.action_map, config.hidden)
        self.env = gym.make(config.gym_env_string)
        self.t = RedisTransport()
        self.uuid = uuid.uuid4()

    def rollout(self, policy_weights, num_episodes=2):
        self.policy.load_state_dict(policy_weights)

        rollout = []
        for id in range(num_episodes):
            episode, total_reward = single_episode(self.env, self.config, self.policy)
            rollout.append(episode)
            msg = EpisodeMessage(self.uuid, id, len(episode), total_reward)
            self.t.publish('rollout', msg)

        return rollout


def ppo_loss(newprob, oldprob, advantage, clip=0.2):
    ratio = newprob / oldprob

    clipped_ratio = ratio.clamp(1.0 - clip, 1.0 + clip)
    clipped_step = clipped_ratio * advantage
    full_step = ratio * advantage
    min_step = torch.stack((full_step, clipped_step), dim=1)
    min_step, clipped = torch.min(min_step, dim=1)

    # logging.debug(f'ADVTG {advantage[0].data}')
    # logging.debug(f'NEW_P {newprob[0].data}')
    # logging.debug(f'OLD_P {oldprob[0].data}')
    # logging.debug(f'RATIO {ratio[0].data}')
    # logging.debug(f'CLIP_ {clipped_step[0].data}')

    min_step *= -1
    return min_step.mean()


def train_policy(policy, rollout_dataset, config):
    optim = torch.optim.Adam(lr=1e-4, params=policy.new.parameters())
    policy = policy.train()
    policy = policy.to(config.device)

    batches = math.floor(len(rollout_dataset) / config.max_minibatch_size) + 1
    batch_size = math.floor(len(rollout_dataset) / batches)
    steps_per_batch = math.floor(12 / batches) if math.floor(12 / batches) > 0 else 1
    #config.tb.add_scalar('batches', batches, config.tb_step)

    rollout_loader = DataLoader(rollout_dataset, batch_size=batch_size, shuffle=True)
    batches_p = 0
    for i, (observation, action, reward, advantage) in enumerate(rollout_loader):
        batches_p += 1
        for step in range(steps_per_batch):

            observation = observation.to(config.device)
            advantage = advantage.float().to(config.device)
            action = action.squeeze().to(config.device)
            optim.zero_grad()

            if config.debug:
                print(f'ACT__ {action[0].data}')

            new_logprob = policy(observation.squeeze().view(-1, policy.features)).squeeze()
            new_prob = torch.exp(torch.distributions.Categorical(logits=new_logprob).log_prob(action))
            new_logprob.retain_grad()
            old_logprob = policy(observation.squeeze().view(-1, policy.features), old=True).squeeze()
            old_prob = torch.exp(torch.distributions.Categorical(logits=old_logprob).log_prob(action))
            policy.backup()

            loss = ppo_loss(new_prob, old_prob, advantage, clip=0.2)
            loss.backward()
            optim.step()

            if config.debug:
                updated_logprob = policy(observation.squeeze().view(-1, policy.features)).squeeze()
                print(f'CHNGE {(torch.exp(updated_logprob) - torch.exp(new_logprob)).data[0]}')
                print(f'NEW_G {torch.exp(new_logprob.grad.data[0])}')

            # if config.device is 'cuda':
            #     config.tb.add_scalar('memory_allocated', torch.cuda.memory_allocated(), config.tb_step)
            #     config.tb.add_scalar('memory_cached', torch.cuda.memory_cached(), config.tb_step)
    # logging.info(f'processed {batches_p} batches')
    # if config.gpu_profile:
    #     gpu_profile(frame=sys._getframe(), event='line', arg=None)




if __name__ == "__main__":
    ray.init(local_mode=True)

    config = config.CartPole()
    config.experience_threads = 10

    main_uuid = uuid.uuid4()
    main_t = RedisTransport()
    tbl = TensorBoardListener(RedisTransport(), config)
    tbl.start()

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

    dataset = AdvantageDataset(rollout, config.state_transform, config.discount_factor)

    train_policy(policy, dataset, config)

    main_t.publish('rollout', KillMessage(main_uuid))