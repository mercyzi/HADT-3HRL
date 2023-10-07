from abc import ABC, abstractmethod




class IntrinsicReward(ABC):
    """
    Abstract class for intrinsic rewards as exploration bonuses
    """

    def __init__(self, observation_size, action_size):
        """
        Initialise parameters for intrinsic reward
        :param observation_space: observation space of environment
        :param action space: action space of environment
        :param parallel_envs: number of parallel environments
        :param config: configuration for intrinsic reward
        """

        self.obs_size = observation_size
        self.action_size = action_size

        # self.parallel_envs = parallel_envs

        # # set all values from config as attributes
        # for k, v in flatten(cfg).items():
        #     setattr(IntrinsicReward, k, v)

    @abstractmethod
    def compute_intrinsic_reward(self, state, action, next_state, train=False):
        """
        Compute intrinsic reward for given input
        :param state: (batch of) current state(s)
        :param action: (batch of) applied action(s)
        :param next_state: (batch of) next/reached state(s)
        :param train: flag if model should be trained
        :return: dict of 'intrinsic reward' and losses
        """
        raise NotImplementedError
   
    def episode_reset(self, environment_idx):
        """
        Indicate termination of episode/ start of new episode

        :param environment_idx: index of environment for which new episode started
        """
        pass
