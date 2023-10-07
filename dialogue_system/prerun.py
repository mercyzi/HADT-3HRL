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

sys.path.append(os.getcwd().replace("src/dialogue_system",""))


from src.dialogue_system.policy_learning.dqn_torch import DQNModel



class multi_worker_model(object):
    def __init__(self, worknet):
        
        self.input_size_dqn_all = {1: 374, 4: 494, 5: 389, 6: 339, 7: 279, 12: 304, 13: 359, 14: 394, 19: 414}
        self.id2lowerAgent = {}
        self.id2optimizer = {}
        self.label_all_path = './../../data/synthetic_dataset'
        
        self.id2symptom = {}
        self.symptom2id = {}
        self.symptom_set = pickle.load(open(os.path.join(self.label_all_path, 'slot_set.p'), 'rb'))
        try:
            self.symptom_set.pop("disease")
        except:
            pass
        for symptom, v in self.symptom_set.items():
            self.symptom2id[symptom] = v
            self.id2symptom[v] = symptom
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.criterion = torch.nn.CrossEntropyLoss()
        named_tuple = ("features","tag")
        self.Transition = namedtuple('Transition', named_tuple)

        self.save_model = {}
        for key, value in self.input_size_dqn_all.items():
            label = str(key)
            label_new_path = os.path.join(self.label_all_path, 'label' + str(label))
            #disease_symptom = pickle.load(open(os.path.join(label_new_path, 'disease_symptom.p'), 'rb'))
            slot_set = pickle.load(open(os.path.join(label_new_path, 'slot_set.p'), 'rb'))
            slot_set.pop("disease")
            ## address for saving model 
            path_list = './../model/pre_woker/'.split('/')
            path_list.insert(-1, str(label))
            self.save_model[label] = '/'.join(path_list)
            ## same structure with DQN
            input_size = len(slot_set) * 3
            hidden_size = 512
            output_size = len(slot_set)
            self.id2lowerAgent[label] = worknet[label]
            #self.id2lowerAgent[label] = DQNModel(input_size=input_size, hidden_size=hidden_size,output_size=output_size, parameter=None).to(self.device)

            weight_p, bias_p = [], []
            for name, p in self.id2lowerAgent[label].named_parameters():
                if 'bias' in name:
                    bias_p.append(p)
                else:
                    weight_p.append(p)
            self.id2optimizer[label] = torch.optim.Adam([
                {'params': weight_p, 'weight_decay': 0.001},  # with L2 regularization
                {'params': bias_p, 'weight_decay': 0}  # no L2 regularization.
            ], lr=0.0004)


    def train(self, batch, label):
        batch = self.Transition(*zip(*batch))
        features = torch.Tensor(batch.features).to(self.device)
        tag = torch.LongTensor(batch.tag).to(self.device)
        out = self.id2lowerAgent[label].forward(features)
        loss = self.criterion(out, tag)
        self.id2optimizer[label].zero_grad()
        loss.backward()
        self.id2optimizer[label].step()
        return {"loss": loss.item()}
    
    def predict(self, features, label):
        self.id2lowerAgent[label].eval()
        features = torch.Tensor(features).to(self.device)
        Ys = self.id2lowerAgent[label].forward(features)
        max_index = np.argmax(Ys.detach().cpu().numpy(), axis=1)
        self.id2lowerAgent[label].train()
        return Ys, max_index

    def test(self, test_batch, label):
        
        batch = self.Transition(*zip(*test_batch))
        features = torch.Tensor(batch.features).to(self.device)
        tag = batch.tag
        Ys, pred = self.predict(features.cpu(), label)
        num_correct = len([1 for i in range(len(tag)) if tag[i]==pred[i]])
        test_acc = num_correct / len(test_batch)
        return test_acc

    def train_workers(self, epochs, group_id):
        batch_size = 100
        group_id = str(group_id)
        for key, value in self.input_size_dqn_all.items():
            ### load dataset
            label = str(key)
            if label == group_id:
                label_new_path = os.path.join(self.label_all_path, 'label' + str(label))
                #disease_symptom = pickle.load(open(os.path.join(label_new_path, 'disease_symptom.p'), 'rb'))
                slot_set = pickle.load(open(os.path.join(label_new_path, 'slot_set.p'), 'rb'))
                slot_set.pop("disease")
                goal_set = pickle.load(open(os.path.join(label_new_path, 'goal_set.p'), 'rb'))
                ### build dataset
                total_batch = self.build_woker_pretraining_dataset(slot_set, goal_set)
                print('worker_id = '+label+' :',len(total_batch))
                ### train epochs
                for iter in range(epochs):
                    batch = random.sample(total_batch, batch_size)
                    loss = self.train(batch, label)
                    if iter % 1000 == 0:
                        test_batch = total_batch
                        acc = self.test(test_batch, label)
                        print('epoch:{},loss:{:.4f},acc:{}'.format(iter, loss["loss"], acc))
            
            

    def build_woker_pretraining_dataset(self, slot_set, goal_set):
        ### symptom into model input
        def define_symptom_input(explicit_symptom, slot_set):
            state_input = np.zeros((len(slot_set.keys()),3))
            for slot in slot_set:
                if slot == explicit_symptom:
                    temp_slot = [1,0,0]
                else:
                    temp_slot = [0,0,1]
                state_input[slot_set[slot], :] = temp_slot
            state_rep = state_input.reshape(1,len(slot_set.keys())*3)[0]
            return state_rep
        ### consider only train dataset
        total_set = copy.deepcopy(goal_set["train"])
        total_batch = []

        for i, dialogue in enumerate(total_set):
            explicit = dialogue['goal']['explicit_inform_slots']
            implicit = dialogue['goal']['implicit_inform_slots']
            for imp_slot, value in implicit.items():
                if value == True:
                    temp = list(explicit.keys())[0]
                    input_state = define_symptom_input(temp,slot_set)
                    tag_id = slot_set[imp_slot]
                    total_batch.append((input_state, tag_id))
        return total_batch    

    def saved_model(self, label):
        if os.path.isdir(self.save_model[label]) == False:
            os.makedirs(self.save_model[label])
        model_file_name = os.path.join(self.save_model[label], "wokerId" + label + ".pkl")

        torch.save(self.id2lowerAgent[label].state_dict(), model_file_name)   






if __name__ == "__main__":
    
    wokrer = multi_worker_model()
    wokrer.train_workers(3000)
