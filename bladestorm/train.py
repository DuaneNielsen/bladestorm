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
from messages import EpisodeMessage, RedisTransport, Server, StopAllMessage, KillMessage, ResetMessage
import uuid
from peewee import PostgresqlDatabase, Model, CharField, TimestampField, BlobField, FloatField, IntegerField, Proxy
import datetime
from statistics import mean
import pickle
import tensorboardX
from util import timeit, WorkerInstrument, TimingReport
import util


class TensorBoardListener(Server):
    def __init__(self, transport, name):
        super().__init__(transport, 'rollout')
        self.register(EpisodeMessage, self.episode)
        self.register(ResetMessage, self.reset)
        self.tb_step = 0
        self.name = name
        self.tb = tensorboardX.SummaryWriter(name)

    def episode(self, msg):
        self.tb.add_scalar('reward', msg.total_reward, self.tb_step)
        self.tb.add_scalar('epi_len', msg.steps, self.tb_step)
        self.tb_step += 1


class Stat:
    def __init__(self, total_reward, epi_length):
        self.total_reward = total_reward
        self.epi_length = epi_length


def fib(n):
    if n == 1:
        return 1
    else:
        return n * fib(n-1)


@ray.remote(num_cpus=1)
class ExperienceEnv(object):
    def __init__(self, config):
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        self.config = config
        self.policy = PPOWrap(config.features, config.action_map, config.hidden)
        self.env = gym.make(config.gym_env_string)
        self.t = RedisTransport()
        self.uuid = uuid.uuid4()

    #@ray.method(num_return_vals=3)
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

@timeit
def train_policy(policy, rollout_dataset, config):
    optim = torch.optim.Adam(lr=1e-4, params=policy.new.parameters())
    policy = policy.train()
    policy = policy.to(config.device)

    batches = math.floor(len(rollout_dataset) / config.max_minibatch_size) + 1
    batch_size = math.floor(len(rollout_dataset) / batches)
    steps_per_batch = math.floor(12 / batches) if math.floor(12 / batches) > 0 else 1
    # config.tb.add_scalar('batches', batches, config.tb_step)

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


database_proxy = Proxy()


class BaseModel(Model):
    """A base model that will use our Postgresql database"""
    class Meta:
        database = database_proxy


class PolicyStore(BaseModel):
    run = CharField()
    timestamp = TimestampField()
    ave_reward = FloatField()
    ave_episode_length = IntegerField()
    policy = BlobField()


class StatsCollector:
    def __init__(self):
        self.stats = []

    def append(self, stats):
        self.stats = self.stats + stats

    def mean_reward(self):
        return mean([s.total_reward for s in self.stats])

    def mean_episode_length(self):
        return mean([s.epi_length for s in self.stats])

    def total_episode_length(self):
        return sum([s.epi_length for s in self.stats])

    def __str__(self):
        return \
            f"mean reward : {self.mean_reward()}\n" + \
            f"mean epi_l  : {self.mean_episode_length()}\n" + \
            f"totl epi_l : {self.total_episode_length()}"


def save_policy(policy_weights, stats):
    policy_blob = pickle.dumps(policy_weights, 0)
    pstore = PolicyStore(run=config.run_id,
                         timestamp=datetime.datetime.now(),
                         ave_reward=stats.mean_reward(),
                         ave_episode_length=stats.mean_episode_length(),
                         policy=policy_blob
                         )
    pstore.save()


def main():
    global steps
    main_uuid = uuid.uuid4()
    main_t = RedisTransport()
    tbl = TensorBoardListener(RedisTransport(), config.run_dir)
    tbl.start()
    policy = PPOWrap(config.features, config.action_map, config.hidden)
    for epoch in range(config.num_epochs):
        print(f'started epoch {epoch}')
        policy_weights, rollout, stats = multi_rollout(policy)
        print(stats)
        steps += stats.total_episode_length()
        save_policy(policy_weights, stats)

        print(f'starting training {epoch}')
        dataset = AdvantageDataset(rollout, config.state_transform, config.discount_factor)

        train_policy(policy, dataset, config)
    main_t.publish('rollout', KillMessage(main_uuid))

@timeit
def multi_rollout(policy):
    policy_weights = policy.state_dict()
    experience = []
    gatherers = [ExperienceEnv.remote(config) for _ in range(config.experience_threads)]
    for i in range(config.experience_threads):
        instr = WorkerInstrument()
        instr.main_start()
        experience.append(gatherers[i].rollout.remote(policy_weights, config.episode_batch_size, instr))
    rollout = []
    stats = StatsCollector()
    for i in range(config.experience_threads):
        ready, waiting = ray.wait(experience)
        w_rollout, w_stats, instr_r = ray.get(ready[0])
        instr_r.main_stop()
        timings.append(instr_r)
        rollout = rollout + w_rollout
        stats.append(w_stats)
    return policy_weights, rollout, stats


if __name__ == "__main__":
    local_mode = False
    ray.init(local_mode=local_mode)

    config = config.LunarLander()
    config.experience_threads = 1
    config.num_epochs = 10
    config.episode_batch_size = 40

    steps = 0

    db = PostgresqlDatabase('testpython', user='ppo', password='password',
                            host='localhost', port=5432)
    database_proxy.initialize(db)
    db.create_tables([PolicyStore])

    timings = TimingReport()

    total_time = util.SimpleInstrument()
    total_time.start()

    main()

    total_time.end()
    total_time_report = util.SimpleReport()
    total_time_report.append(total_time)

    print(f'local {local_mode} threads {config.experience_threads} epochs {config.num_epochs} episodes {config.episode_batch_size} steps {steps}')
    total_time_report.dump()
    timings.dump()
    # import sys
    # import pstats
    # ps = pstats.Stats(prof, stream=sys.stdout)
    # ps.print_stats()
    # filename = 'profile.prof'  # You can change this if needed
    # prof.dump_stats(filename)