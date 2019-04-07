import torch
import random
from tensorboardX import SummaryWriter
import datetime


class DefaultPrePro:
    def __call__(self, observation_t1, observation_t0):
        return observation_t1 - observation_t0


class DefaultStateTransform:
    def __call__(self, observation, insert_batch=False):
        """
        :param observation: the raw observation
        :param insert_batch: add a batch dimension to the front
        :return: tensor in shape (batch, dims)
        """
        if insert_batch:
            return torch.from_numpy(observation).float().unsqueeze(0)
        else:
            return torch.from_numpy(observation).float()


class DefaultActionTransform:
    def __call__(self, action):
        return action


class BaseConfig:
    def __init__(self,
                 gym_env_string,
                 discount_factor=0.99,
                 max_rollout_len=3000,
                 prepro=DefaultPrePro(),
                 state_transform=DefaultStateTransform(),
                 action_transform=DefaultActionTransform()
                 ):
        self.gym_env_string = gym_env_string
        self.discount_factor = discount_factor
        self.max_rollout_len = max_rollout_len
        self.prepro = prepro
        self.state_transform = state_transform
        self.action_transform = action_transform
        self.episode_batch_size = 10
        self.experience_threads = 10
        self.print_tensor_sizes = True
        self.last_tensor_sizes = set()
        self.gpu_profile = False
        self.gpu_profile_fn = f'{datetime.datetime.now():%d-%b-%y-%H-%M-%S}-gpu_mem_prof.txt'
        self.lineno = None
        self.func_name = None
        self.filename = None
        self.module_name = None
        self.tb_step = 0
        self.save_freq = 1000
        self.view_games = False
        self.view_obs = False
        self.num_epochs = 6000
        self.num_rollouts = 60
        self.collected_rollouts = 0
        self.device = 'cpu' if torch.cuda.is_available() else 'cpu'
        self.max_minibatch_size = 400000
        self.resume = False
        self.debug = False
        self.redis_host = 'localhost'
        self.redis_port = '6379'
        self.redis_db = '0'

    def rundir(self, name='default'):
        return f'runs/{name}_{random.randint(0, 1000)}'

    def getSummaryWriter(self, name='default'):
        return SummaryWriter(self.rundir(name))


class DiscreteConfig(BaseConfig):
    def __init__(self,
                 gym_env_string,
                 action_map,
                 default_action=0,
                 discount_factor=0.99,
                 max_rollout_len=3000,
                 prepro=DefaultPrePro(),
                 state_transform=DefaultStateTransform(),
                 ):
        super().__init__(gym_env_string, discount_factor, max_rollout_len, prepro, state_transform)
        self.action_map = action_map
        self.default_action = default_action


class LunarLander(DiscreteConfig):
    def __init__(self):
        super().__init__(
            gym_env_string='LunarLander-v2',
            action_map=[0, 1, 2, 3]
        )
        self.features = 8
        self.hidden = 8
        self.adversarial = False
        self.default_save = ['lunar_lander/solved.wgt']
        self.players = 1


class CartPole(DiscreteConfig):
    def __init__(self):
        super().__init__(
            gym_env_string='CartPole-v1',
            action_map=[0, 1]
        )
        self.features = 4
        self.hidden = 3