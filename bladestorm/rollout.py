class Step:
    def __init__(self, state, action, reward, done):
        self.state = state
        self.reward = reward
        self.action = action
        self.done = done


def single_episode(env, config, policy, v=None, render=False):
    """

    :param config: configuration
    :param env: The simulator, reset will be called at the start of each episode
    :param policy: the policy to run
    :param v: an object with a render method for displaying images, if set pre-processed observations will be rendered
    :param render: if True, env.render will be called for each step
    :return:
    """

    episode = []
    episode_length = 0
    observation_t0 = env.reset()
    action = config.default_action
    observation_t1, reward, done, info = env.step(action)
    state = config.prepro(observation_t1, observation_t0)
    observation_t0 = observation_t1

    done = False
    while not done:
        # take an action on current observation and record result
        observation_tensor = config.state_transform(state, insert_batch=True)
        action_prob = policy(observation_tensor)
        index, action = policy.sample(action_prob)

        observation_t1, reward, done, info = env.step(action.squeeze().item())

        done = done or episode_length > config.max_rollout_len

        episode.append(Step(state, index, reward, done))

        # compute the observation that resulted from our action
        state = config.prepro(observation_t1, observation_t0)
        observation_t0 = observation_t1

        if render:
            env.render(mode='human')
        if v is not None:
            v.render(state)

    return episode


