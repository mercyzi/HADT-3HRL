'''
Author: John
Email: johnjim0816@gmail.com
Date: 2020-09-11 23:03:00
LastEditor: John
LastEditTime: 2022-12-17 19:56:24
Discription: 
Environment: 
'''
import time
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical,Normal
import torch.utils.data as Data
import numpy as np
from collections import deque
import operator
import math
import random

from src.dialogue_system.intrinsic_rewards.icm import ICM
from src.dialogue_system.intrinsic_rewards.rnd import RND
from src.dialogue_system.intrinsic_rewards.ride import RIDE


class Critic(nn.Module):
    def __init__(self,input_dim,output_dim,hidden_dim=256):
        super(Critic,self).__init__()
        assert output_dim == 1 # critic must output a single value
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        value = self.fc3(x)
        return value

class ActorSoftmax(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256):
        super(ActorSoftmax, self).__init__()
        # self.policy_layer = torch.nn.Sequential(
        #     torch.nn.Linear(input_dim, hidden_dim),
            
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(hidden_dim, hidden_dim),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(hidden_dim, output_dim)
        # )
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
    def forward(self,x):
        if torch.cuda.is_available():
            x.cuda()
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # probs = F.softmax(self.fc3(x),dim=1)
        probs = self.fc3(x)
        # print(self.fc3(x))
        # print(probs)
        # assert 0
        # probs =  self.policy_layer(x)
        return probs


class ReplayBufferQue:
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.buffer = deque(maxlen=self.capacity)
    def push(self,transitions):
        '''_summary_
        Args:
            trainsitions (tuple): _description_
        '''
        self.buffer.append(transitions)
    def sample(self, batch_size: int, sequential: bool = False):
        if batch_size > len(self.buffer):
            batch_size = len(self.buffer)
        if sequential: # sequential sampling
            rand = random.randint(0, len(self.buffer) - batch_size)
            batch = [self.buffer[i] for i in range(rand, rand + batch_size)]
            return zip(*batch)
        else:
            batch = random.sample(self.buffer, batch_size)
            return zip(*batch)
    def clear(self):
        self.buffer.clear()
    def __len__(self):
        return len(self.buffer)

class PGReplay(ReplayBufferQue):
    '''replay buffer for policy gradient based methods, each time these methods will sample all transitions
    Args:
        ReplayBufferQue (_type_): _description_
    '''
    def __init__(self):
        self.buffer = deque()
    def sample(self):
        ''' sample all the transitions
        '''
        batch = list(self.buffer)
        return zip(*batch)

class PPO(object):
    def __init__(self, input_size, hidden_size, output_size, parameter):
        self.ppo_type = 'clip' # clip or kl
        self.params = parameter
        if self.ppo_type == 'kl':
            self.kl_target = 0.1 # target KL divergence
            self.kl_lambda = 0.5 # lambda for KL penalty, 0.5 is the default value in the paper
            self.kl_beta = 1.5 # beta for KL penalty, 1.5 is the default value in the paper
            self.kl_alpha = 2 # alpha for KL penalty, 2 is the default value in the paper
        self.gamma = 0.99
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor = ActorSoftmax(input_size,output_size, hidden_dim = hidden_size).cuda(device=self.device)
        self.critic = Critic(input_size,1,hidden_dim=hidden_size).cuda(device=self.device)
        print(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=0.0003)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=0.001)
        self.memory = PGReplay()
        self.icm = ICM(input_size,output_size)
        self.rnd = RND(input_size,output_size)
        self.ride = RIDE(input_size,output_size)
        self.k_epochs = 4 # update policy for K epochs
        self.sample_count = 0
        self.table_count = 0
        self.eps_clip = 0.1 # clip parameter for PPO
        self.entropy_coef = 0.1 # entropy coefficient
        self.train_batch_size = 512 # ppo train batch size
        self.sgd_batch_size = 32 # sgd batch size
        self.goal_amend = 0.95
        self.imc_amend = 0

    def predict(self,state,mask_human):
        self.sample_count += 1
        end1 = time.time()
        state = torch.tensor(state, device=self.device, dtype=torch.float32).unsqueeze(dim=0)
        end2 = time.time()
        probs = self.actor(state)
        end3 = time.time()
        mask = torch.ByteTensor([[0,mask_human, 0]]).to(device=self.device)
        probs = probs.masked_fill(mask==1, value=torch.tensor(-1e9))
        probs = F.softmax(probs,dim=1)
        dist = Categorical(probs)
        action = dist.sample()
        self.probs = probs.detach()
        self.log_probs = dist.log_prob(action).detach()
        
        # print('程序运行时间为: %s Seconds'%(end3-end2))
        # print('程序运行时间为: %s Seconds'%(end2-end1))
        # assert 0
        # print(self.probs)
        # print(self.log_probs)
        # assert 0
        return action.detach().cpu().numpy().item()
    @torch.no_grad()
    def predict_eval(self,state,mask_human):
        state = torch.tensor(state, device=self.device, dtype=torch.float32).unsqueeze(dim=0)
        probs = self.actor(state)

        mask = torch.ByteTensor([[0,mask_human, 0]]).to(device=self.device)
        probs = probs.masked_fill(mask==1, value=torch.tensor(-1e9))
        
        probs = F.softmax(probs,dim=1)
        action = torch.argmax(probs,dim=1)
        # print(action.detach().cpu().item())
        # print(action)
        # print(probs)
        # assert 0
        
        return action.detach().cpu().item()
    
    def update(self, share_agent=None):
        # update policy every train_batch_size steps
        if self.sample_count <= self.train_batch_size :
            return
        # print("update policy")
        states, actions, rewards, dones, probs, log_probs, mask_batch, next_states = self.memory.sample()
        # convert to tensor
        states = torch.tensor(np.array(states), device=self.device, dtype=torch.float32) # shape:[train_batch_size,n_states]
        next_states = torch.tensor(np.array(next_states), device=self.device, dtype=torch.float32) # shape:[train_batch_size,n_states]
        actions_cpu = torch.tensor(np.array(actions), device='cpu', dtype=torch.float32) 
        actions = torch.tensor(np.array(actions), device=self.device, dtype=torch.float32).unsqueeze(dim=1) # shape:[train_batch_size,1]
        rewards = torch.tensor(np.array(rewards), device=self.device, dtype=torch.float32).unsqueeze(dim=1) # shape:[train_batch_size,1]
        dones_cpu = torch.tensor(np.array(dones), device='cpu', dtype=torch.float32)
        dones = torch.tensor(np.array(dones), device=self.device, dtype=torch.float32).unsqueeze(dim=1) # shape:[train_batch_size,1]
        
        probs = torch.cat(probs).to(self.device) # shape:[train_batch_size,n_actions]
        log_probs = torch.tensor(log_probs, device=self.device, dtype=torch.float32).unsqueeze(dim=1) # shape:[train_batch_size,1]  
        masks = torch.tensor(np.array(mask_batch), device=self.device, dtype=torch.uint8) # shape:[train_batch_size,1]  
        # intrinsic_reward
        int_dict = self.icm.compute_intrinsic_reward(states, actions_cpu, next_states)
        # rnd_dict = self.rnd.compute_intrinsic_reward(states, actions_cpu, next_states)
        # ride_dict = self.ride.compute_intrinsic_reward(states, actions_cpu, next_states)
        # self.ride.episode_reset()
        in_reward =  int_dict['intrinsic_reward']
        # print(ride_dict['control_rewards'])
        # print("count",ride_dict['count_rewards'])


        rewards_all = rewards + in_reward.unsqueeze(dim=1)
        

        returns, ave_returns, num_eps = self._compute_returns(rewards_all, dones, in_reward.unsqueeze(dim=1)) # shape:[train_batch_size,1]    
        torch_dataset = Data.TensorDataset(states, actions, probs, log_probs,returns, masks)
        train_loader = Data.DataLoader(dataset=torch_dataset, batch_size=self.sgd_batch_size, shuffle=True,drop_last=False)
        
        
        
        loss_mean = []
        for _ in range(self.k_epochs):
            for batch_idx, (old_states, old_actions, old_probs, old_log_probs, returns, masks) in enumerate(train_loader):

                
                # compute advantages
                values = self.critic(old_states) # detach to avoid backprop through the critic
                advantages = returns - values.detach() # shape:[train_batch_size,1]
                # get action probabilities
                new_probs = self.actor(old_states) # shape:[train_batch_size,n_actions]
                new_probs = new_probs.masked_fill(masks==1, value=torch.tensor(-1e9))
        
                new_probs = F.softmax(new_probs,dim=1)
                dist = Categorical(new_probs)
                
                # get new action probabilities
                new_log_probs = dist.log_prob(old_actions.squeeze(dim=1)) # shape:[train_batch_size]
                
                # compute ratio (pi_theta / pi_theta__old):
                ratio = torch.exp(new_log_probs.unsqueeze(dim=1) - old_log_probs) # shape: [train_batch_size, 1]
                # compute surrogate loss
                surr1 = ratio * advantages # shape: [train_batch_size, 1]

                if self.ppo_type == 'clip':
                    surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
                    # compute actor loss
                    actor_loss = - (torch.min(surr1, surr2).mean() + self.entropy_coef * dist.entropy().mean())
                elif self.ppo_type == 'kl':
                    kl_mean = F.kl_div(torch.log(new_probs.detach()), old_probs.unsqueeze(1),reduction='mean') # KL(input|target),new_probs.shape: [train_batch_size, n_actions]
                    # kl_div = torch.mean(new_probs * (torch.log(new_probs) - torch.log(old_probs)), dim=1) # KL(new|old),new_probs.shape: [train_batch_size, n_actions]
                    surr2 = self.kl_lambda * kl_mean
                    # surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
                    # compute actor loss
                    actor_loss = - (surr1.mean() + surr2 + self.entropy_coef * dist.entropy().mean())
                    if kl_mean > self.kl_beta * self.kl_target:
                        self.kl_lambda *= self.kl_alpha
                    elif kl_mean < 1/self.kl_beta * self.kl_target:
                        self.kl_lambda /= self.kl_alpha
                else:
                    raise NameError
                # compute critic loss
                critic_loss = nn.MSELoss()(returns, values) # shape: [train_batch_size, 1]

                
                # tot_loss = actor_loss + 0.5 * critic_loss
                # take gradient step
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()  
                actor_loss.backward()
                critic_loss.backward()
                # tot_loss.backward()
                self.actor_optimizer.step()
                self.critic_optimizer.step()

                loss_mean.append(actor_loss.detach().item()) 
        memory_len = len(self.memory)            
        self.memory.clear()
        self.sample_count = 0
        return {"loss":np.mean(loss_mean), "reward":ave_returns ,"reward_imc":np.mean(in_reward.cpu().numpy()),'len':memory_len}
    def _compute_returns(self, rewards, dones, in_rewards):
        # monte carlo estimate of state rewards
        returns = []
        real_return = []
        discounted_sum = 0
        discounted_sum_in = 0
        for reward, done, in_reward in zip(reversed(rewards), reversed(dones), reversed(in_rewards)):
            if done:
                discounted_sum = 0
                discounted_sum_in = 0
            discounted_sum = reward + (self.gamma * discounted_sum)
            discounted_sum_in = in_reward + (self.gamma * discounted_sum_in)
            if done:
                real_return.append(discounted_sum.item() - discounted_sum_in.item())
            returns.insert(0, discounted_sum)
        # Normalizing the rewards:
        # assert 0
        returns = torch.tensor(returns, device=self.device, dtype=torch.float32).unsqueeze(dim=1)
        returns = (returns - returns.mean()) / (returns.std() + 1e-5) # 1e-5 to avoid division by zero
        return returns, np.mean(real_return), len(real_return)

    def _compute_imc_rewards(self, state_rep, next_state_rep,imc_value):
        
        a = []
        for i in range(state_rep.size()[0]):
            if imc_value[i][0] != 0:
                a.append(imc_value[i][0].item())
                imc_value[i][0] = torch.tensor(self.imc_amend ,device=self.device, dtype=torch.float32)

        if self.table_count % 30 == 29:
            self.imc_amend += np.mean([i for i in a])
            
        return imc_value
    
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
        average_activate_human = model_performance["average_activate_human"]
        model_file_name = os.path.join(checkpoint_path, "s" + str(success_rate) + "_r" + str(average_reward) + "_t" + str(average_turn)\
                          + "_aah" + str(average_activate_human) + "_mr2-" + str(average_match_rate2) + "_e-" + str(episodes_index) + ".pkl")

        torch.save({'actor_state_dict': self.actor.state_dict(),
                    'critic_state_dict': self.critic.state_dict()}, model_file_name)
    
    def load_model(self, saved_model):
        checkpoint = torch.load(saved_model, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
    
    def save_traj(self, traj, fpath):
        from pathlib import Path
        import pickle
        # create path
        Path(fpath).mkdir(parents=True, exist_ok=True)
        traj_pkl = os.path.join(fpath, 'traj.pkl')
        with open(traj_pkl, 'wb') as f:
            pickle.dump(traj, f)