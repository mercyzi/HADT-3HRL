'''
@Author: John
@Email: johnjim0816@gmail.com
@Date: 2020-06-10 15:27:16
@LastEditor: John
LastEditTime: 2022-11-16 06:24:40
@Discription: 
@Environment: python 3.7.7
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.distributions import Normal
import random
import numpy as np
import math
from collections import deque
import operator
import os
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity # 经验回放的容量
        self.buffer = [] # 缓冲区
        self.position = 0 
    
    def push(self, state, action, reward, next_state, done):
        ''' 缓冲区是一个队列，容量超出时去掉开始存入的转移(transition)
        '''
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity 
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size) # 随机采出小批量转移
        state, action, reward, next_state, done =  zip(*batch) # 解压成状态，动作等
        return state, action, reward, next_state, done
    
    def __len__(self):
        ''' 返回当前存储的量
        '''
        return len(self.buffer)
# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)

class QNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim):
        super(QNetwork, self).__init__()

        # Q1 architecture
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, num_actions)

        # Q2 architecture
        self.linear4 = nn.Linear(num_inputs, hidden_dim)
        self.linear5 = nn.Linear(hidden_dim, hidden_dim)
        self.linear6 = nn.Linear(hidden_dim, num_actions)

        self.apply(weights_init_)

    def forward(self, state):
        xu = state
        
        x1 = F.relu(self.linear1(xu))
        x1 = F.relu(self.linear2(x1))
        x1 = self.linear3(x1)

        x2 = F.relu(self.linear4(xu))
        x2 = F.relu(self.linear5(x2))
        x2 = self.linear6(x2)

        return x1, x2


class PolicyNet(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim):
        super(PolicyNet, self).__init__()
        
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, num_actions)

        self.apply(weights_init_)


    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)

        probs = F.softmax(x, -1)
        z = probs == 0.0
        z = z.float() * 1e-8
        return x, probs + z

# class cfg:
#     def __init__(self):
#         self.epsilon_start = 0.95 # epsilon start value
#         self.epsilon_end = 0.01 # epsilon end value
#         self.epsilon_decay = 500 # epsilon decay rate
#         self.lr = 1e-3 # learning rate 
#         self.gamma = 0.99 # discount factor
#         self.tau = 0.005 # soft update factor
#         self.alpha = 0.1 # Temperature parameter α determines the relative importance of the entropy term against the reward # 0.1
#         self.automatic_entropy_tuning = False # automatically adjust α
#         self.batch_size = 64 # batch size # 256
#         self.hidden_dim = 256 # hidden dimension # 256
#         self.n_epochs = 1 # number of epochs
#         self.target_update = 1 # interval for updating the target network
#         self.buffer_size = 1000000 # replay buffer size
class SAC:
    def __init__(self, input_size, hidden_size, output_size, parameter):
        self.params = parameter
        self.n_states = input_size
        self.n_actions = output_size
        # self.action_space = cfg.action_space
        self.sample_count = 0
        self.update_count = 0
        self.gamma = 0.99
        self.tau = 0.005
        self.alpha = 0.1
        self.n_epochs = 1
        self.target_update = 1
        self.automatic_entropy_tuning = False
        self.batch_size = 64
        self.memory = ReplayBuffer(1000000)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
        self.critic = QNetwork(input_size,output_size, hidden_size).to(device=self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=1e-3)
        self.critic_target = QNetwork(input_size, output_size, hidden_size).to(self.device)
        
        self.target_entropy = 0.98 * (-np.log(1 / self.n_actions))
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha = self.log_alpha.exp()
        self.alpha_optim = Adam([self.log_alpha], lr=1e-3)

        self.epsilon = 0.95
        self.epsilon_start = 0.95
        self.epsilon_end = 0.01
        self.epsilon_decay = 500

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)
        
        self.policy = PolicyNet(input_size, output_size, hidden_size).to(self.device)
        self.policy_optim = Adam(self.policy.parameters(), lr=1e-3)

    # def sample_action(self,state):
    #     self.sample_count+=1
    #     self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
    #         math.exp(-1. * self.sample_count / self.epsilon_decay) 
    #     if random.random() < self.epsilon:
    #         action = random.randrange(self.n_actions)
    #     else:
    #         state = torch.tensor(state, device=self.device, dtype=torch.float32).unsqueeze(0)
    #         q_values, _ = self.policy(state)
    #         action = q_values.max(1)[1].item() # choose action corresponding to the maximum q value
    #     return action
        
    def predict(self,state):
        state = torch.tensor(state, device=self.device, dtype=torch.float32).unsqueeze(0)
        q_values, _ = self.policy(state)
        action = q_values.max(1)[1].item() # choose action corresponding to the maximum q value
        return action.detach().cpu().numpy()[0]
    
    def singleBatch(self):
        if len(self.memory) < self.batch_size: # when transitions in memory donot meet a batch, not update
            return
        
        self.update_count += 1
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.memory.sample(batch_size=self.batch_size)

        state_batch = torch.tensor(state_batch, device=self.device,  dtype=torch.float)
        action_batch = torch.tensor(action_batch, device=self.device).unsqueeze(1)
        reward_batch = torch.tensor(reward_batch, device=self.device, dtype=torch.float).unsqueeze(1)
        next_state_batch = torch.tensor(next_state_batch, device=self.device, dtype=torch.float)
        done_batch = torch.tensor(done_batch, device=self.device, dtype=torch.float).unsqueeze(1)

        with torch.no_grad():
            next_state_action, next_probs = self.policy(next_state_batch)
            next_log_probs = torch.log(next_probs)

            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch)
            min_qf_next_target = (next_probs * (torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_log_probs)).sum(-1).unsqueeze(-1)
            next_q_value = reward_batch + (1 - done_batch) * self.gamma * (min_qf_next_target)

        qf1, qf2 = self.critic(state_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1 = qf1.gather(1, action_batch) ; qf2 = qf2.gather(1, action_batch)

        qf1_loss = F.mse_loss(qf1, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(qf2, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf_loss = qf1_loss + qf2_loss

        self.critic_optim.zero_grad()
        qf_loss.backward()
        for param in self.critic.parameters():  
            param.grad.data.clamp_(-1, 1)
        self.critic_optim.step()


        pi, probs = self.policy(state_batch)
        log_probs = torch.log(probs)
        with torch.no_grad():
            qf1_pi, qf2_pi = self.critic(state_batch)
            min_qf_pi = torch.min(qf1_pi, qf2_pi)
        policy_loss = (probs * ((self.alpha * log_probs) - min_qf_pi)).mean() # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]
        
        self.policy_optim.zero_grad()
        policy_loss.backward()
        for param in self.policy.parameters():  
            param.grad.data.clamp_(-1, 1)            
        self.policy_optim.step()

        log_probs = (probs * log_probs).sum(-1)
        alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()

        self.alpha = self.log_alpha.exp()

        
        return {"loss":policy_loss.item()}
            

    def save_model(self, model_performance,episodes_index, checkpoint_path):
        
        if os.path.isdir(checkpoint_path) == False:
            # os.mkdir(checkpoint_path)
            #print(os.getcwd())
            os.makedirs(checkpoint_path)
        agent_id = self.params.get("agent_id").lower()
        disease_number = self.params.get("disease_number")
        success_rate = model_performance["success_rate"]
        average_reward = model_performance["average_reward"]
        average_turn = model_performance["average_turn"]
        average_match_rate = model_performance["average_match_rate"]
        average_match_rate2 = model_performance["average_match_rate2"]
        model_file_name = os.path.join(checkpoint_path, "model_d" + str(disease_number) +  str(agent_id) + "_s" + str(success_rate) + "_r" + str(average_reward) + "_t" + str(average_turn)\
                          + "_mr" + str(average_match_rate) + "_mr2-" + str(average_match_rate2) + "_e-" + str(episodes_index) + ".pkl")

        torch.save({'policy_state_dict': self.policy.state_dict(),
                    'critic_state_dict': self.critic.state_dict(),
                    'critic_target_state_dict': self.critic_target.state_dict(),
                    'critic_optimizer_state_dict': self.critic_optim.state_dict(),
                    'policy_optimizer_state_dict': self.policy_optim.state_dict()}, model_file_name)
        
    
    def load_model(self, saved_model):
        checkpoint = torch.load(saved_model, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
        self.critic_optim.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        self.policy_optim.load_state_dict(checkpoint['policy_optimizer_state_dict'])

    def update_target_network(self):
        """
        Updating the target network with the parameters copyed from the current networks.
        """
        # hard update
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
                target_param.data.copy_( param.data )
        