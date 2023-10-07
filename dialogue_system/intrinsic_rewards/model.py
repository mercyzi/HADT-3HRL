import torch
import torch.nn as nn

from src.dialogue_system.intrinsic_rewards.utils import prod
from src.dialogue_system.intrinsic_rewards.utils import build_sequential




class ICMNetwork(nn.Module):
    """Intrinsic curiosity module (ICM) network"""

    def __init__(
        self, obs_size, action_size
    ):
        """
        Initialize parameters and build model.
        :param observation_space: space of each observation
        :param action_size: dimension of each action
        """
        super(ICMNetwork, self).__init__()
        self.state_rep_size = 16

        input_dim = obs_size

        # state representation
        self.state_rep = build_sequential(input_dim, [64,64,16])

        # inverse model
        inverse_model_hiddens = [64]
        inverse_model_hiddens.append(action_size)
        self.inverse_model = build_sequential(self.state_rep_size * 2, inverse_model_hiddens)

        # forward model
        forward_model_hiddens = [64]
        forward_model_hiddens.append(self.state_rep_size)
        self.forward_model = build_sequential(self.state_rep_size + action_size, forward_model_hiddens)

    def forward(self, state, next_state, action):
        """
        Compute forward pass over ICM network
        :param state: current state
        :param next_state: reached state
        :param action: applied action
        :return: predicted_action, predicted_next_state_rep, next_state_rep
        """
        # compute state representations
        state_rep = self.state_rep(state)
        next_state_rep = self.state_rep(next_state)

        # inverse model output
        inverse_input = torch.cat([state_rep, next_state_rep], dim=1)
        predicted_action = self.inverse_model(inverse_input)

        # forward model output
        # print(state_rep)
        # print(action)
        forward_input = torch.cat([state_rep, action], dim=1)
        predicted_next_state_rep = self.forward_model(forward_input)

        return predicted_action, predicted_next_state_rep, next_state_rep

class RNDNetwork(nn.Module):
    """Random Network Distillation (RND) network"""

    def __init__(self, obs_size):
        """
        Initialize parameters and build model.
        :param observation_space: space of each observation
        :param model_dict: dictionary for model configuration
        """
        super(RNDNetwork, self).__init__()
        self.obs_size = obs_size
        self.state_rep_size = 16
        input_dim = obs_size

        # state representation
        self.state_rep = build_sequential(obs_size, [64,64,16])

    def forward(self, state):
        """
        Compute forward pass over RND network
        :param state: state
        :return: state representation
        """
        return self.state_rep(state)

class RIDENetwork(nn.Module):
    """Network for Rewarding Impact-Driven Exploration (RIDE)"""

    def __init__(
        self, obs_size, action_size
    ):
        """
        Initialize parameters and build model.
        :param observation_space: space of each observation
        :param action_size: dimension of each action
        :param model_dict: dictionary for model configuration
        """
        super(RIDENetwork, self).__init__()
        self.state_rep_size = 16

        input_dim = obs_size

        # state representation
        self.state_rep = build_sequential(input_dim, [64,64,16])

        # inverse model
        inverse_model_hiddens = [64]
        inverse_model_hiddens.append(action_size)
        self.inverse_model = build_sequential(self.state_rep_size * 2, inverse_model_hiddens)

        # forward model
        forward_model_hiddens = [64]
        forward_model_hiddens.append(self.state_rep_size)
        self.forward_model = build_sequential(self.state_rep_size + action_size, forward_model_hiddens)

    def forward(self, state, next_state, action):
        """
        Compute forward pass over RIDE network
        :param state: current state
        :param next_state: reached state
        :param action: applied action
        :return: 
            state representation for current state,
            state representation for next state, 
            predicted_action,
            predicted state representation for next state, 
        """
        # compute state representations
        state_rep = self.state_rep(state)
        next_state_rep = self.state_rep(next_state)

        # inverse model output
        inverse_input = torch.cat([state_rep, next_state_rep], dim=1)
        predicted_action = self.inverse_model(inverse_input)

        # forward model output
        forward_input = torch.cat([state_rep, action], dim=1)
        predicted_next_state_rep = self.forward_model(forward_input)

        return state_rep, next_state_rep, predicted_action, predicted_next_state_rep