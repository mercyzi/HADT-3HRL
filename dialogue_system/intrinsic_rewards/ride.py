import copy

import gym
import torch
import torch.nn.functional as F
import torch.optim as optim

from src.dialogue_system.intrinsic_rewards.intrinsic_reward import IntrinsicReward
from src.dialogue_system.intrinsic_rewards.model import  RIDENetwork

from src.dialogue_system.intrinsic_rewards.dict_count import  DictCount



class RIDE(IntrinsicReward):
    """
    Rewarding Impact-Driven Exploration (RIDE) code

    Paper:
    Raileanu, Roberta, and Tim Rocktäschel (2020).
    RIDE: Rewarding impact-driven exploration for procedurally-generated environments.
    In International Conference on Learning Representations.

    Paper: https://arxiv.org/abs/2002.12292
	
	Open-source code: https://github.com/facebookresearch/impact-driven-exploration
    """
    def __init__(self, observation_size, action_size, **kwargs):
        """
        Initialise parameters for RIDE intrinsic reward definition
        :param observation_space (gym.spaces.space): observation space of environment
        :param action space (gym.spaces.space): action space of environment
        :param parallel_envs (int): number of parallel environments
        :param cfg (Dict): configuration for intrinsic reward
        """
        super(RIDE, self).__init__(observation_size, action_size)
        self.model_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.lr = 4e-8
        self.max_grad_norm = .5
        self.intrinsic_reward_coef = 20
        self.forward_loss_coef = 5
        self.inverse_loss_coef = 1
        # create model architecture
        self.model = RIDENetwork(
            self.obs_size, self.action_size,
        ).to(self.model_device)
        
        # define episodic count (reset at each episode so separate counts for each parallel environment)
        # don't use intrinsic reward coef for episodic counts
        
        
        self.episodic_count = DictCount(observation_size, action_size)
        self.optimiser = optim.Adam(self.model.parameters(), lr=self.lr)

    def _prediction(self, state, action, next_state):
        """
        Compute prediction
        :param state: (batch of) current state(s)
        :param action: (batch of) applied action(s)
        :param next_state: (batch of) next/reached state(s)
        :return: (batch of) control_rewards, count_rewards, forward loss, inverse_loss
        """
        actions_onehot = torch.eye(self.action_size)[action.long()].squeeze().to(self.model_device)
        state_rep, next_state_rep, predicted_action, predicted_next_state_rep = self.model(
            state, next_state, actions_onehot
        )
        # discrete one-hot encoded action
        action_targets = actions_onehot.max(1)[1]
        inverse_loss = F.cross_entropy(predicted_action, action_targets, reduction="none")
        
        forward_loss = 0.5 * ((next_state_rep - predicted_next_state_rep) ** 2).sum(-1)

        control_rewards = 0.5 * ((next_state_rep - state_rep) ** 2).sum(-1)
        count_rewards = self.episodic_count.compute_intrinsic_reward(state,action,next_state)["intrinsic_reward"]
        

        return control_rewards, count_rewards, forward_loss, inverse_loss

    def compute_intrinsic_reward(self, state, action, next_state, train=True):
        """
        Compute intrinsic reward for given input
        :param state: (batch of) current state(s)
        :param action: (batch of) applied action(s)
        :param next_state: (batch of) next/reached state(s)
        :param train: flag if model should be trained
        :return: dict of 'intrinsic reward' and losses
        """
        if train:
            control_rewards, count_rewards, forward_loss, inverse_loss = self._prediction(state, action, next_state)
        else:
            with torch.no_grad():
                control_rewards, count_rewards, forward_loss, inverse_loss = self._prediction(state, action, next_state)

        loss = self.forward_loss_coef * forward_loss.mean() + self.inverse_loss_coef * inverse_loss.mean()

        # optimise RIDE model
        self.optimiser.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimiser.step()

        intrinsic_rewards = count_rewards * control_rewards
        int_reward = intrinsic_rewards.detach()

        return {
            "intrinsic_reward": self.intrinsic_reward_coef * int_reward,
            "forward_loss": forward_loss.mean(),
            "inverse_loss": inverse_loss.mean(),
            "control_rewards": control_rewards.mean(),
            "count_rewards": count_rewards.mean(),
        }

    def episode_reset(self):
        """
        Indicate termination of episode/ start of new episode

        :param environment_idx: index of environment for which new episode started
        """
        self.episodic_count.reset()
