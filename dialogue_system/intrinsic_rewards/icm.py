import torch
import torch.nn.functional as F
import torch.optim as optim

from src.dialogue_system.intrinsic_rewards.intrinsic_reward import IntrinsicReward
from src.dialogue_system.intrinsic_rewards.model import ICMNetwork

class ICM(IntrinsicReward):
    """
    Intrinsic curiosity module (ICM) class

    Paper:
    Pathak, D., Agrawal, P., Efros, A. A., & Darrell, T. (2017).
    Curiosity-driven exploration by self-supervised prediction.
    In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops (pp. 16-17).

    Paper: http://openaccess.thecvf.com/content_cvpr_2017_workshops/w5/html/Pathak_Curiosity-Driven_Exploration_by_CVPR_2017_paper.html

	Open-source code: https://arxiv.org/abs/1705.05363
    """
    def __init__(self, observation_size, action_size, **kwargs):
        """
        Initialise parameters for ICM intrinsic reward definition
        :param observation_space: observation space of environment
        :param action space: action space of environment
        :param parallel_envs (int): number of parallel environments
        :param config: configuration for intrinsic reward
        """
        super(ICM, self).__init__(observation_size, action_size)
        self.model_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.lr = 8e-7
        self.max_grad_norm = .5
        self.intrinsic_reward_coef = 4
        self.forward_loss_coef = 10
        self.inverse_loss_coef = 1
        
        self.model = ICMNetwork(
            self.obs_size, self.action_size,
        ).to(self.model_device)
        self.optimiser = optim.Adam(self.model.parameters(), lr=self.lr)

    def _prediction(self, state, action, next_state):
        """
        Compute prediction
        :param state: (batch of) current state(s)
        :param action: (batch of) applied action(s)
        :param next_state: (batch of) next/reached state(s)
        :return: (batch of) forward loss, inverse_loss
        """
        # action = torch.tensor([2])
        actions_onehot = torch.eye(self.action_size)[action.long()].squeeze().to(self.model_device)
        # print(actions_onehot)
        # assert 0
        predicted_action, predicted_next_state_rep, next_state_rep = self.model(
            state, next_state, actions_onehot
        )
        # discrete one-hot encoded action
        action_targets = actions_onehot.max(1)[1]
        inverse_loss = F.cross_entropy(predicted_action, action_targets, reduction="none")
        
        forward_loss = 0.5 * ((next_state_rep - predicted_next_state_rep) ** 2).sum(-1)
        return forward_loss, inverse_loss

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
            forward_loss, inverse_loss = self._prediction(state, action, next_state)
        else:
            with torch.no_grad():
                forward_loss, inverse_loss = self._prediction(state, action, next_state)
        
        loss = self.forward_loss_coef * forward_loss.mean() + self.inverse_loss_coef * inverse_loss.mean()

        # optimise ICM model
        self.optimiser.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimiser.step()

        int_reward = forward_loss.detach()

        return {
            "intrinsic_reward": self.intrinsic_reward_coef * int_reward,
            "forward_loss": forward_loss.mean().item(),
            "inverse_loss": inverse_loss.mean().item(),
        }