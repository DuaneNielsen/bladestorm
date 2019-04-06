from torch.utils.data.dataset import Dataset
from bisect import bisect_right
import statistics


class AdvantageDataset(Dataset):
    def __init__(self, rollout, state_transform, discount_factor=0.99):
        self.rollout = rollout
        self.discount_factor = discount_factor
        self.state_transform = state_transform

        self.lengths = []

        for episode in self.rollout:
            self.lengths.append(len(episode))

        self.episode_off = []
        offset = 0
        for l in self.lengths:
            self.episode_off.append(offset)
            offset += l

        self.len = sum(self.lengths)

        self.adv = []

        for episode in rollout:
            self.advantage(episode)

        self.mean = statistics.mean(self.adv)
        self.stdev = statistics.stdev(self.adv)

    def advantage(self, episode):

        cum_value = 0.0

        for step in reversed(episode):
            cum_value = step.reward + cum_value * self.discount_factor
            step.advantage = cum_value
            self.adv.append(cum_value)

    @staticmethod
    def find_le(a, x):
        'Find rightmost value less than or equal to x'
        i = bisect_right(a, x)
        if i:
            return i - 1
        raise ValueError

    def __getitem__(self, item):
        episode_i = AdvantageDataset.find_le(self.episode_off, item)
        step_i = item - self.episode_off[episode_i]
        step = self.rollout[episode_i][step_i]
        normalized_advantage = (step.advantage - self.mean) / (self.stdev + 1e-12)
        state_t = self.state_transform(step.state)
        return state_t, step.action, step.reward, normalized_advantage

    def __len__(self):
        return self.len
