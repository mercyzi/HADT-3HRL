import numpy as np
import copy
import sys, os
import random
import re
import pickle
import math
import torch
import torch.nn.functional
from collections import namedtuple

sys.path.append(os.getcwd().replace("src/dialogue_system/reward_shaping",""))


class NNModel(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, parameter):
        super(NNModel, self).__init__()
        self.params = parameter
        # different layers. Two layers.
        self.policy_layer = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_size, bias=True),
            torch.nn.Dropout(0.5),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hidden_size, output_size, bias=True)
        )

        # one layer.
        #self.policy_layer = torch.nn.Linear(input_size, output_size, bias=True)

    def forward(self, x):
        if torch.cuda.is_available():
            x.cuda()
        q_values = self.policy_layer(x)
        return q_values


class RewardModel(object):
    def __init__(self):
        
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.criterion = torch.nn.MSELoss()
        named_tuple = ("human_rate","reward")
        self.Transition = namedtuple('Transition', named_tuple)

        
        ## simple structure of net
        input_size = 1
        hidden_size = 5
        output_size = 1
        
        self.net = NNModel(input_size=input_size, hidden_size=hidden_size,output_size=output_size, parameter=None).to(self.device)
        self.total_batch = []
        weight_p, bias_p = [], []
        for name, p in self.net.named_parameters():
            if 'bias' in name:
                bias_p.append(p)
            else:
                weight_p.append(p)
        self.optimizer = torch.optim.Adam([
            {'params': weight_p, 'weight_decay': 0.001},  # with L2 regularization
            {'params': bias_p, 'weight_decay': 0}  # no L2 regularization.
        ], lr=0.0004)


    def train(self, batch):
        batch = self.Transition(*zip(*batch))
        features = torch.Tensor(batch.human_rate).view(-1,1).to(torch.float32).to(self.device)
        
        tag = torch.Tensor(batch.reward).to(torch.float32).to(self.device)
        
        out = self.net.forward(features)
        # print("features",features.tolist())
        # print("rew",batch.reward)
        # print("tag",tag.tolist())
        # print("out",out.tolist())
        loss = self.criterion(out.view(-1), tag.view(-1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {"loss": loss.item()}
    
    def predict(self, features):
        self.net.eval()
        features = torch.Tensor(features).to(torch.float32).view(-1,1).to(self.device)
        Ys = self.net.forward(features)
        self.net.train()
        return Ys

    def test(self, test_batch):
        
        batch = self.Transition(*zip(*test_batch))
        features = torch.Tensor(batch.human_rate).to(torch.float32).to(self.device)
        # print(features)
        tag = batch.reward
        Ys = self.predict(features.cpu()).view(-1)
        # print("tag",tag)
        # print("pred",Ys.tolist())
        num_correct = len([1 for i in range(len(tag)) if abs(tag[i]-Ys[i]) < 0.01])
        
        test_acc = num_correct / len(test_batch)
        return test_acc

    def train_reward_model(self, epochs, name):
        batch_size = 100
        if len(self.total_batch) < 50:
            return 
        ### train epochs
        for iter in range(epochs):
            if len(self.total_batch) < batch_size:
                batch = self.total_batch
            else:
                batch = random.sample(self.total_batch, batch_size)
            loss = self.train(batch)
        
        test_batch = self.total_batch
        acc = self.test(test_batch)
        print('[Reward Model {}] size:{},loss:{:.4f},acc:{:.4f}'.format(name ,len(self.total_batch), loss["loss"], acc))
            
            

    def build_train_set(self, human_rate, reward):
        ### 总大小为200
        batch = []
        temp = []
        temp.append(human_rate)
        if len(self.total_batch) > 200 - 1 :
            self.total_batch = self.total_batch[1:]
        self.total_batch.append((temp, reward))
           

if __name__ == "__main__":
    
    model = RewardModel()
    for iter in range(210):
        model.build_train_set(4*0.1,iter,iter*-0.1)
    model.train_reward_model(10000000)
