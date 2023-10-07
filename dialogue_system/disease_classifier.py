import torch
import torch.nn.functional
import os
import numpy as np
from collections import namedtuple
import pickle
import copy
import random
import torch.nn as nn
import torch.nn.functional as F
import argparse
import matplotlib.pyplot as plt
from torchmetrics.classification import MulticlassAccuracy
import torch.nn as nn
# import torchvision
from torchvision import models
from collections import namedtuple
import timm
from timm.models import create_model
from timm.models import swin_small_patch4_window7_224
from fvcore.nn import sigmoid_focal_loss_jit
from torch import Tensor
import matplotlib.pyplot as plt

import matplotlib as mpl
import seaborn as sns
import pickle
import os
import time
from scipy import interpolate
import numpy as np
from scipy.ndimage import gaussian_filter1d


class classifer(nn.Module):
	def __init__(self,in_ch,num_classes):
		super().__init__()
		self.avgpool = nn.AdaptiveAvgPool1d(1)
		self.fc = nn.Linear(in_ch,num_classes)
 
	def forward(self, x):
		x = self.avgpool(x.transpose(1, 2))  # B C 1
		x = torch.flatten(x, 1)
		x = self.fc(x)
		return x
 
class Swin(nn.Module):
    def __init__(self):
        super().__init__() 
        #创建模型，并且加载预训练参数
        self.swin= create_model('swin_large_patch4_window7_224_in22k',pretrained=True)
        #整体模型的结构
        pretrained_dict = self.swin.state_dict()
        #去除模型的分类层
        self.backbone = nn.Sequential(*list(self.swin.children())[:-2])
        #去除分类层的模型架构
        model_dict = self.backbone.state_dict()
 
        # 将pretrained_dict里不属于model_dict的键剔除掉
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # 更新现有的model_dict
        model_dict.update(pretrained_dict)
        # 加载我们真正需要的state_dict
        self.backbone.load_state_dict(model_dict)
        #屏蔽除分类层所有的参数
        for p in self.backbone.parameters():
            p.requires_grad = False
        #构建新的分类层
        self.head = classifer(1536, 90)
 
    def forward(self, x):
        x = self.backbone(x)
        x=self.head(x)
        return x

class Resnet18Backbone(nn.Module):
    def __init__(self):
        super(Resnet18Backbone, self).__init__()

        # self.model = torchvision.models.resnet18(pretrained=False)
        self.model.load_state_dict(torch.load('resnet18-5c106cde.pth'))
        self.model.fc = nn.Sequential(
            torch.nn.Linear(512, 256, bias=True),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(256, 90, bias=True)
        )

    def forward(self, x):
        x = x.reshape(-1,3, 2, 133)
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        x = x.squeeze(2).squeeze(2)
        x = self.model.fc(x)
        # print(x.size())
        # assert 0

        return x



class Model(torch.nn.Module):
    """
    DQN model with one fully connected layer, written in pytorch.
    """
    def __init__(self, input_size, hidden_size, output_size):
        super(Model, self).__init__()
        # different layers. Two layers.
        # self.vgg19 = models.vgg19_bn(pretrained=False)
        # self.vgg19.load_state_dict(torch.load('/home/zxh/srcformz4/src/dialogue_system/vgg19_bn-c79401a0.pth'))
        # self.vgg19.classifier._modules['6'] = nn.Sequential(nn.Linear(4096, 230*3))

        # self.m1 = timm.create_model('vgg19', pretrained=True, num_classes=0, global_pool='')
        # # self.m1.load_state_dict(torch.load("/home/zxh/srcformz4/src/dialogue_system/resnet50_a1_0-14fe96d1.pth"), strict=True)
        # self.m1.reset_classifier(num_classes=230*3, global_pool='fast')
        # self.m1 = swin_small_patch4_window7_224(pretrained=True)
        # num_ftrs = self.m1.head.in_features
        # self.m1.head = nn.Linear(num_ftrs, 4)
        
        self.policy_layer = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_size, bias=True),
            torch.nn.Dropout(0.3),
            torch.nn.LeakyReLU(),
            #torch.nn.Linear(hidden_size,hidden_size),
            #torch.nn.Dropout(0.5),
            #torch.nn.LeakyReLU(),
            torch.nn.Linear(hidden_size, output_size, bias=True)
        )

        # one layer.
        #self.policy_layer = torch.nn.Linear(input_size, output_size, bias=True)

    def forward(self, x):
        if torch.cuda.is_available():
            x.cuda()
        # x = x.reshape(-1,3, 115, 2)
        # y = self.m1(x.float())
          
        # # print(y)
        # # assert 0
        q_values = self.policy_layer(x.float())
        # q_values = self.m1(x)

        return q_values

class FocalLoss(torch.nn.Module):

    def __init__(self, weight=None, reduction='mean', gamma=0, eps=1e-7):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.ce = torch.nn.CrossEntropyLoss(weight=weight, reduction=reduction)

    def forward(self, input, target):
        logp = self.ce(input, target)
        p = torch.exp(-logp)
        loss = (1 - p) ** self.gamma * logp
        return loss.mean()

class MultiCEFocalLoss(torch.nn.Module):
    def __init__(self, class_num, gamma=2, alpha=None, reduction='mean'):
        super(MultiCEFocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Tensor(torch.ones(class_num, 1)).cuda(device='cuda:0')
        else:
            self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.class_num =  class_num

    def forward(self, predict, target):
        pt = F.softmax(predict, dim=1) # softmmax获取预测概率
        class_mask = F.one_hot(target, self.class_num) #获取target的one hot编码
        ids = target.view(-1, 1) 
        # print(ids)
        # print(self.alpha)
        alpha = self.alpha[ids.data.view(-1)].view(-1,1) # 注意，这里的alpha是给定的一个list(tensor
#),里面的元素分别是每一个类的权重因子
        probs = (pt * class_mask).sum(1).view(-1, 1) # 利用onehot作为mask，提取对应的pt
        log_p = probs.log()
# 同样，原始ce上增加一个动态权重衰减因子
        loss = -alpha * (torch.pow((1 - probs), self.gamma)) * log_p

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss

class dl_classifier(object):
    def __init__(self, input_size, hidden_size, output_size,  parameter):
        self.parameter = parameter
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = Model(input_size=input_size, hidden_size=hidden_size, output_size=output_size).to(self.device)
        # self.model = Swin().cuda(device=self.device)
        # self.model = Resnet18Backbone().cuda(device=self.device)
        print(self.model)
        # assert 0
        self.acc = []
        weight_p, bias_p = [], []
        for name, p in self.model.named_parameters():
            if 'bias' in name:
                bias_p.append(p)
            else:
                weight_p.append(p)
        self.weight_decay = 0.001
        self.lr = 0.0001#0.01-0.001
        self.loss_mode = 'focloss' ##ce foc 
        self.optimizer = torch.optim.Adam([
            {'params': weight_p, 'weight_decay': self.weight_decay},  # with L2 regularization
            {'params': bias_p, 'weight_decay': 0}  # no L2 regularization.
        ], lr=self.lr)
        # print(self.optimizer)
        # assert 0
        #], lr=parameter.get("dqn_learning_rate"))
        self.weight=None#torch.from_numpy(np.array([0.1, 0.8, 1.0, 1.0])).float()
        self.criterion = torch.nn.CrossEntropyLoss()
        self.focalloss_2= FocalLoss(gamma=0.5, reduction='none',weight = self.weight)#2,3,4,5
        self.focalloss_m = MultiCEFocalLoss(class_num = output_size,gamma=2,reduction='sum')
        named_tuple = ("slot","disease")
        self.Transition = namedtuple('Transition', named_tuple)
        self.metric = MulticlassAccuracy(num_classes=output_size,average=None)
        self.values = 0
        self.count = 0
        #self.test_batch = self.create_data(train_mode=False)

        #if self.params.get("train_mode") is False and self.params.get("agent_id").lower() == 'agentdqn':
        #    self.restore_model(self.params.get("saved_model"))

    def train(self, batch):
        batch = self.Transition(*zip(*batch))
        #print(batch.slot.shape)
        slot = torch.Tensor(batch.slot).to(self.device)
        disease = torch.LongTensor(batch.disease).to(self.device)
        out = self.model.forward(slot)
        # print(disease)
        # print(out)
        #print(out.shape, disease)
        
        # loss = self.criterion(out, disease)
        # loss = sigmoid_focal_loss_jit(
        #     out,
        #     disease.reshape(100,1),
        #     alpha=0.25,
        #     gamma=2.0,
        #     reduction="sum",
        # )
        # loss = self.focalloss_2(out, disease)
        if self.loss_mode == 'focloss':
            loss = self.focalloss_m(out ,disease)
        else:
            loss = self.criterion(out, disease)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {"loss": loss.item()}

    def predict(self, slots,ruled_candidate_disease = [0,1,2,3]):
        self.model.eval()
        # print(batch.slot.shape)
        slots = torch.Tensor(slots).to(self.device)
        Ys = self.model.forward(slots)
        y_cpu = []
        y_cpu.append([Ys.detach().cpu().numpy()[0][i] if i in ruled_candidate_disease else -99999 for i in range(0,4) ]) 
        y_cpu = np.asarray(y_cpu)
        # y_cpu.reshape((1,4))
        if ruled_candidate_disease == [0,1,2,3]:
            max_index = np.argmax(Ys.detach().cpu().numpy(),axis = 1)
        else:
            max_index = np.argmax(y_cpu,axis = 1)
        self.model.train()
        return Ys, max_index


    def train_dl_classifier(self, epochs):
        batch_size = self.parameter.get("batch_size")
        #print(batch_size)
        #print(self.total_batch[0])
        total_batch = self.create_data(train_mode=True)
        for iter in range(epochs):
            batch = random.sample(total_batch, batch_size)
            #print(batch[0][0].shape)
            loss = self.train(batch)
            if iter%100==0:
                print('epoch:{},loss:{:.4f}'.format(iter, loss["loss"]))
                self.acc.append(self.test_dl_classifier())

    def test_dl_classifier(self):
        self.model.eval()
        self.test_batch = self.create_data(train_mode=False)
        batch = self.Transition(*zip(*self.test_batch))
        slot = torch.Tensor(batch.slot).to(self.device)
        #disease = torch.LongTensor(batch.disease).to(self.device)
        disease = batch.disease
        Ys, pred = self.predict(slot.cpu())
        # print(Ys)
        # print(type(pred))
        disease = np.asarray(disease)
        self.count += 1
        # print(type(disease))
        # print(self.values)
        # print(self.metric(torch.from_numpy(pred), torch.from_numpy(disease)))
        self.values += self.metric(torch.from_numpy(pred), torch.from_numpy(disease))
        
        print(self.values/self.count)
        
        # print(pred)
        # print(disease)
        num_correct = len([1 for i in range(len(disease)) if disease[i]==pred[i]])
        print(len([1 for i in range(len(disease)) if disease[i]==pred[i] and disease[i] == 80] ))
        print("the test accuracy is %f", num_correct / len(self.test_batch))
        # print(dir(self.metric.to('cpu')))
        # fig_, ax_ = self.metric.plot()
        # fig_.show()
        self.model.train()
        return float("%.3f" % (num_correct / len(self.test_batch)))

    def test(self, test_batch):
        #self.model.eval()

        batch = self.Transition(*zip(*test_batch))
        slot = torch.LongTensor(batch.slot).to(self.device)
        #disease = torch.LongTensor(batch.disease).to(self.device)
        disease = batch.disease
        Ys, pred = self.predict(slot.cpu())
        #print(pred)
        disease = np.asarray(disease)
        self.count += 1
        # print(type(disease))
        # print(self.values)
        # print(self.metric(torch.from_numpy(pred), torch.from_numpy(disease)))
        self.values = self.metric(torch.from_numpy(pred), torch.from_numpy(disease))
        # if self.count % 10 == 9:
        #     print(self.values)

        num_correct = len([1 for i in range(len(disease)) if disease[i]==pred[i]])
        #print("the test accuracy is %f", num_correct / len(self.test_batch))
        test_acc = num_correct / len(test_batch)
        #self.model.train()
        return test_acc



    def create_data(self, train_mode):
        goal_set = pickle.load(open(self.parameter.get("goal_set"), 'rb'))
        self.slot_set = pickle.load(open(self.parameter.get("slot_set"), 'rb'))
        disease_symptom = pickle.load(open(self.parameter.get("disease_symptom"),'rb'))
        # print(disease_symptom.keys())
        # assert 0
        
        self.disease2id = {}
        for disease, v in disease_symptom.items():
            self.disease2id[disease] = v['index']
        self.slot_set.pop('disease')
        disease_y = []
        # total_set = random.sample(goal_set['train'], 10000)
        if train_mode==True:
            total_set = copy.deepcopy(goal_set["train"])
        else:
            total_set = copy.deepcopy(goal_set["test"])
        total_batch = []


        for i, dialogue in enumerate(total_set):
            # slots_exp = [0] * len(self.slot_set)
            tag = dialogue['disease_tag']
            # tag_group=disease_symptom1[tag]['symptom']
            disease_y.append(tag)
            

            goal = dialogue['goal']

            symp = {}
            explicit = goal['explicit_inform_slots']
            implicit = goal['implicit_inform_slots']
            symp.update(explicit)
            symp.update(implicit)
            # print(implicit)
            # print(symp)
            # print(self.slot_set)
            slots_exp = np.zeros((len(self.slot_set.keys()),3))

            for slot in self.slot_set:
                if slot in symp.keys():
                    if symp[slot] is True:
                        temp_slot = [1,0,0]
                    elif symp[slot] == dialogue_configuration.I_DO_NOT_KNOW:
                        # assert 0
                        temp_slot = [0,1,0]
                    else:
                        temp_slot = [0,0,1]
                    #print(symp[slot], temp_slot)
                else:
                    temp_slot = [0,0,1]
                slots_exp[self.slot_set[slot], :] = temp_slot
            slots_exp = slots_exp.reshape(1,len(self.slot_set.keys())*3)[0]
            # print(type(slots_exp))
            if sum(slots_exp) == 0:
                print("############################")
            total_batch.append((slots_exp, self.disease2id[tag]))
        #print("the disease data creation is over")
        return total_batch

    def save_model(self,  model_performance, episodes_index, checkpoint_path):
        if os.path.isdir(checkpoint_path) == False:
            os.makedirs(checkpoint_path)
        agent_id = self.parameter.get("agent_id").lower()
        disease_number = self.parameter.get("disease_number")
        success_rate = model_performance["success_rate"]
        average_reward = model_performance["average_reward"]
        average_turn = model_performance["average_turn"]
        average_match_rate = model_performance["average_match_rate"]
        average_match_rate2 = model_performance["average_match_rate2"]
        average_activate_human = model_performance["average_activate_human"]
        model_file_name = os.path.join(checkpoint_path, "s" + str(success_rate) + "_r" + str(average_reward) + "_t" + str(average_turn)\
                          + "_aah" + str(average_activate_human) + "_mr2-" + str(average_match_rate2) + "_e-" + str(episodes_index) + ".pkl")

        torch.save(self.model.state_dict(), model_file_name)

    def train_deep_learning_classifier(self, epochs):
        #self.model.train_dl_classifier(epochs=5000)
        #print("############   the deep learning model is training over  ###########")
        self.disease_replay = pickle.load(file=open("/home/zxh/new_for_mr2/src/dialogue_system/run/dia_dataset.p", "rb"))
        self.disease_replay_test = pickle.load(file=open("/home/zxh/new_for_mr2/src/dialogue_system/run/dia_dataset_test.p", "rb"))
        for iter in range(epochs):
            batch = random.sample(self.disease_replay, min(100,len(self.disease_replay)))
            loss = self.train(batch=batch)
            if iter % 10 == 9:
                test_batch = random.sample(self.disease_replay_test, min(1000,len(self.disease_replay)))
                test_acc = self.test(test_batch=test_batch)
                print('disease_replay:{},loss:{:.4f}, test_acc:{:.4f}'.format(len(self.disease_replay), loss["loss"], test_acc))
                if test_acc > 0.7:
                    name_file = str(self.loss_mode)+'_wd'+str(self.weight_decay)+'_lr'+str(self.lr)+".pkl"
                    torch.save(self.model.state_dict(), name_file)
                    print(name_file)
                    assert 0
    
    def restore_model(self, saved_model):
        """
        Restoring the trained parameters for the model. Both current and target net are restored from the same parameter.

        Args:
            saved_model (str): the file name which is the trained model.
        """
        print("loading trained model", saved_model)
        if torch.cuda.is_available() is False:
            map_location = 'cpu'
        else:
            map_location = None
        self.model.load_state_dict(torch.load(saved_model,map_location=map_location))

    def eval_mode(self):
        self.model.eval()

    def plot(self, save_name, epochs):
        self.values = [0.986514686,0.9400200080000001,0.9435138670000001,0.9421745810000001,0.990144058,0.96521046,0.941420529,0.9641949929999999,0.9774068675]
        # self.values = self.values.data.numpy()
        # epochs = int(epochs / 100)
        epoch_index = np.arange(9)
        y_mean = np.mean(self.values)
        bar_width = 0.25
        tick_label = ['1', '4', '5', '6', '7','12', '13', '14', '19' ]
        plt.bar(epoch_index, self.values, bar_width,label = "Mean={:.3f}".format(y_mean) ,align="center", color="deepskyblue", alpha=0.5)
        # plt.plot(epoch_index,self.values,label="Low Layer", color = "deepskyblue",linewidth=1)
        plt.axhline(y_mean ,ls = '--', lw = 1,\
           color = 'royalblue', label = 'Average')
        plt.xticks(epoch_index, tick_label)
        
        ax = plt.gca()
        # ax.tick_params(bottom=False, top=False, left=False, right=False)
        # ax.spines['top'].set_visible(False) #去掉上边框
        # ax.spines['bottom'].set_visible(False) #去掉下边框
        # ax.spines['left'].set_visible(False) #去掉左边框
        # ax.spines['right'].set_visible(False)
        # ax.patch.set_facecolor("lightcyan")    # 设置 ax1 区域背景颜色               
        # ax.patch.set_alpha(0.1)
        # ax.spines['right'].set_color('lightcyan')
        # ax.spines['left'].set_color('lightcyan')
        # ax.spines['bottom'].set_color('lightcyan')
        # ax.spines['top'].set_color('lightcyan')
        # # ax.spines['right'].set_linewidth(2)
        # # ax.spines['left'].set_linewidth(2)
        # # ax.spines['bottom'].set_linewidth(2)
        # # ax.spines['top'].set_linewidth(2)
        # plt.grid(True)
        # ax.grid(color='lightcyan',
        # linestyle='-',
        # linewidth=2,
        # alpha=0.3)
        
        plt.xlabel("Group Id")
        plt.ylabel("Symptom recall")
        
        # plt.title("Learning Curve")
    
        plt.legend(loc="lower right")
        # plt.grid(True)
        plt.savefig(save_name,dpi=400)

        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    file0='/home/zxh/HRL/src/data/synthetic_dataset'
    parser.add_argument("--slot_set", dest="slot_set", type=str, default=file0+'/slot_set.p',help='path and filename of the slots set')
    parser.add_argument("--goal_set", dest="goal_set", type=str, default=file0+'/goal_set.p',help='path and filename of user goal')
    parser.add_argument("--disease_symptom", dest="disease_symptom", type=str,default=file0+"/disease_symptom.p",help="path and filename of the disease_symptom file")
    parser.add_argument("--batch_size", dest="batch_size", type=int, default=100, help="the batch size when training.")
    args = parser.parse_args()
    parameter = vars(args)
    epochs = 5000000
    model = dl_classifier(input_size=266*3, hidden_size=256,
                                   output_size=90,
                                   parameter=parameter)
    # model.train_dl_classifier(epochs=epochs)
    model.train_deep_learning_classifier(200000)
    model.plot("1",epochs=epochs)