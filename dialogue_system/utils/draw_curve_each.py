# -*- coding:utf-8 -*-

"""
For parameters. Drwaring the learning curve for each combination of parameters.
"""

import matplotlib.pyplot as plt

import matplotlib as mpl
import seaborn as sns
import pickle
import os
import time
from scipy import interpolate
import scipy.stats as st
import numpy as np
from scipy.ndimage import gaussian_filter1d


class Ploter(object):
    def __init__(self, performance_file):
        self.performance_file = performance_file
        self.epoch_size = 0
        self.success_rate = {}
        self.average_reward = {}
        self.average_match_rate = {}
        self.average_turn = {}
        self.average_activate_human = {}
        self.average_inquire_human = {}
        self.average_ban = {}
        self.human_prob = {}
        self.machine_prob = {}
        self.group_data = {}
        self.weight_prob = []
        self.aah = []
        self.aih = []

    def load_data(self, performance_file, label):
        performance = pickle.load(file=open(performance_file, "rb"))
        self.epoch_size = max(self.epoch_size, len(performance.keys()))
        sr, ar, awd,at, aah,aih,ab = self.__load_data(performance=performance)
        self.success_rate[label] = sr
        self.average_reward[label] = ar
        self.average_match_rate[label] = awd
        self.average_turn[label] = at
        self.average_activate_human[label] = aah
        self.average_inquire_human[label] = aih
        self.average_ban[label] = ab

    def __load_data(self, performance):
        success_rate = []
        average_reward = []
        average_match_rate = []
        average_turn = []
        aah = []
        aih = []
        ab = []
        for index in range(0, len(performance.keys()),1):
            #print(performance[index].keys())
            success_rate.append(performance[index]["success_rate"])
            average_reward.append(performance[index]["average_reward"])
            average_match_rate.append(performance[index]["average_match_rate2"])
            average_turn.append((performance[index]["average_turn"]-1)/2)
            aah.append(performance[index]["average_activate_human"])
            aih.append(performance[index]["average_inquire_human"])
            ab.append(performance[index]["average_ban"])
        return success_rate, average_reward, average_match_rate,average_turn, aah, aih, ab


    def plot(self, save_name, label_list):
        # epoch_index = [i for i in range(0, 500, 1)]

        ax2, ax1 = plt.subplots()
        for label in self.average_turn.keys():
            
            epoch_index = [i for i in range(0, len(self.average_turn[label]))]
            #epoch_index = [i for i in range(0, 1000)]
            # func1 = interpolate.interp1d(epoch_index,self.success_rate[label][0:max(epoch_index)+1],kind='cubic')
            # func2 = interpolate.interp1d(epoch_index,self.average_turn[label][0:max(epoch_index)+1],kind='cubic')


            # len_x = len(epoch_index)
            # new_x = numpy.arange(0.0, len_x-1, 10)
            # new_y1 = func1(new_x)
            # new_y2 = func2(new_x)


            y_max = max(self.success_rate[label][0:max(epoch_index)+1])
            
            y_mean = np.mean(self.success_rate[label][0:max(epoch_index)+1])
            y_var = np.var(self.success_rate[label][0:max(epoch_index)+1], ddof=1)
            
            y2_mean = np.mean(self.average_turn[label][0:max(epoch_index)+1])
            
            y3_mean = np.mean(self.average_activate_human[label][0:max(epoch_index)+1])

            y4_mean = np.mean(self.average_match_rate[label][0:max(epoch_index)+1])

            y5_mean = np.mean(self.average_inquire_human[label][0:max(epoch_index)+1])

            y6_mean = np.mean(self.average_ban[label][0:max(epoch_index)+1])
            y_smoothed = gaussian_filter1d(self.success_rate[label], sigma=10)
            # sns.lineplot(epoch_index,y_smoothed,ci=95)

            low_CI_bound, high_CI_bound = st.t.interval(0.95, len(self.success_rate[label])-1,
                                                        loc=np.mean(self.success_rate[label],0),
                                                        scale=st.sem(self.success_rate[label]))
            print(low_CI_bound)
            print(high_CI_bound)
            print(y_var)
            # assert 0
            plt.plot(epoch_index,y_smoothed,label="Mean={:.3f}±{:.4f}".format(y_mean,y_var), linewidth=1)
            # plt.fill_between(epoch_index,y1 = y_smoothed + bound, y2 = y_smoothed - bound,  alpha=0.3)
            # ax = sns.lineplot(x = epoch_index,y= y_smoothed,errorbar=('ci', 0.95))
            # plt.plot(epoch_index,self.average_turn[label][0:max(epoch_index)+1],label="Mean={:.3f}".format(y2_mean), linewidth=1)
            # plt.plot(epoch_index,self.average_activate_human[label][0:max(epoch_index)+1],label="Mean={:.3f}".format(y3_mean), linewidth=1)
            # plt.plot(epoch_index,self.average_match_rate[label][0:max(epoch_index)+1],label="Mean={:.3f}".format(y4_mean), linewidth=1)
            # plt.plot(epoch_index,self.average_inquire_human[label][0:max(epoch_index)+1],label="Mean={:.3f}".format(y5_mean), linewidth=1)
            # plt.plot(epoch_index,self.average_ban[label][0:max(epoch_index)+1],label="Mean={:.3f}".format(y6_mean), linewidth=1)
            # ax1.plot(new_x, new_y1,color='red',label='Success Rate', linewidth=1)
            # ax1.set_ylabel('Success Rate')
            # ax2 = ax1.twinx() 
            # #ax2.plot(epoch_index,self.average_turn[label][0:max(epoch_index)+1], label='Average Turn', linewidth=1)
            
            # ax2.plot(new_x, new_y2,label='Average Turn', linewidth=1)
            # ax2.set_ylabel('Average Turn')

        # plt.hlines(0.11,0,epoch_index,label="Random Agent", linewidth=1, colors="r")
        # plt.hlines(0.38,0,epoch_index,label="Rule Agent", linewidth=1, colors="purple")

        plt.xlabel("Simulation Epoch")
        plt.ylabel("Success Rate")
        # plt.ylabel('Average Turn')
        # plt.ylabel('Average Activate Human')
        #Implicit symptom Recall
        # plt.ylabel('Implicit symptom Recall')
        # plt.ylabel('Average Inquire Human')
        # plt.ylabel('Average Ban')
        plt.title("Learning Curve")
        # ax1.legend()
        # ax2.legend()
        # if len(label_list) >= 2:
        #     plt.legend()
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.savefig(save_name,dpi=400)

        plt.show()

    def load_group_data(self, performance_file, label):
        performance = pickle.load(file=open(performance_file, "rb"))
        #self.epoch_size = max(self.epoch_size, len(performance.keys()))
        mp = []
        #print(performance)
        #assert 0
        for index in ['1','4','5','6','7','12','13','14','19']:
            #print(performance[index].keys())
            mp.append(performance[index]["true_number_prob"])
        
        
        self.group_data[label] = mp
        
        

    def plot_group(self, save_name, label_list):
        # epoch_index = [i for i in range(0, 500, 1)]

        ax2, ax1 = plt.subplots()
        for label in self.group_data.keys():
            
            
            

            #柱高信息
            Y = self.group_data[label]
            y_mean = np.mean(self.group_data[label])
            
            # print(self.weight_prob)
            # print(y_mean)
            # print(y_11)
            # assert 0
            #print(self.human_prob[label])
            X = np.arange(len(Y))

            bar_width = 0.25
            #"1": 0, "4": 0, "5": 0, "6": 0, "7": 0, "12": 0, "13": 0, "14": 0, "19": 0}
            tick_label = ['1','4','5','6','7','12','13','14','19']

            #显示每个柱的具体高度
            # for x,y in zip(X,Y):
            #     plt.text(x+0.005,y+0.005,'%.3f' %y, ha='center',va='bottom')

            # for x,y1 in zip(X,Y1):
            #     plt.text(x+0.24,y1+0.005,'%.3f' %y1, ha='center',va='bottom')
            
            #绘制柱状图    
            plt.bar(X, Y, bar_width, align="center", color="red", alpha=0.5)
            plt.axhline(y_mean ,ls = '--', lw = 2,\
           color = 'royalblue', label = 'Average')
            plt.xticks(X+bar_width/2, tick_label)


        plt.xlabel("Group Name")
        # plt.ylabel("Average Inquire Rate")
        # plt.ylabel("Average Activate Rate")
        plt.ylabel("Success Rate")
        # plt.ylabel("Average Activate Human")
        plt.title('Pilot Process' + "(Average:{:.1f})".format(y_mean*100))

        
        
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.savefig(save_name,dpi=400)

        plt.show()

    def load_mid_data(self, performance_file, label):
        performance = pickle.load(file=open(performance_file, "rb"))
        #self.epoch_size = max(self.epoch_size, len(performance.keys()))
        hp = []
        mp = []
        human_num = []
        #print(performance)
        #assert 0
        for index in performance.keys():
            #print(performance[index].keys())
            hp.append(performance[index]["human_prob"])
            mp.append(performance[index]["machine_prob"])
            human_num.append(performance[index]["all_number"])
        sum_num = np.sum(human_num)
        
        self.weight_prob = human_num/sum_num
        self.aih = hp
        self.human_prob[label] = hp
        self.machine_prob[label] = mp

    def load_mid_ture_data(self, performance_file, label):
        performance = pickle.load(file=open(performance_file, "rb"))
        #self.epoch_size = max(self.epoch_size, len(performance.keys()))
        hp = []
        mp = []
        human_num = []
        # print(performance)
        # print(len(performance))
        # assert 0
        for index in performance.keys():
            #print(performance[index].keys())
            hp.append(performance[index]["human_ture_prob"])
            mp.append(performance[index]["machine_ture_prob"])
            human_num.append(performance[index]["all_number"])
        sum_num = np.sum(human_num)
        self.weight_prob = human_num/sum_num
        self.human_prob[label] = hp
        print(hp)
        assert 0
        self.machine_prob[label] = mp



    def load_mid_activator_data(self, performance_file, label):
        performance = pickle.load(file=open(performance_file, "rb"))
        #self.epoch_size = max(self.epoch_size, len(performance.keys()))
        hp = []
        mp = []
        human_num = []
        # print(performance)
        # print(len(performance))
        # assert 0
        for index in performance.keys():
            #print(performance[index].keys())
            if performance[index]["activator_all_number"] != 0:
                hp.append(performance[index]["activator_human_prob"])
                mp.append(performance[index]["activator_machine_prob"])
                human_num.append(performance[index]["activator_all_number"])
            else:
                hp.append(0)
                mp.append(0)
                human_num.append(0)
        sum_num = np.sum(human_num)
        self.weight_prob = human_num/sum_num
        self.human_prob[label] = hp
        self.aah = hp
        self.machine_prob[label] = mp

    def plot_mid(self, save_name, label_list):
        # epoch_index = [i for i in range(0, 500, 1)]

        ax2, ax1 = plt.subplots()
        for label in label_list:
            # self.human_prob[label] = np.array([0.7,0.69,0.63,0.59,0.55,0.57,0.53,0.49,0.45,0.41,0.39,0.4,0.37,0.33,0.3,0.27,0.3,0.36,0.39,0.43])
            # self.machine_prob[label] = 1 - self.human_prob[label]
            # self.weight_prob = np.ones(20,dtype=float)
            epoch_index = [i for i in range(0, len(self.machine_prob[label]))]
            # sns.set(color_codes=True)
            # mpl.rcParams["font.sans-serif"] = ["SimHei"]
            # mpl.rcParams["axes.unicode_minus"] = False

            #柱高信息
            Y = self.human_prob[label]
            y_mean = np.average(Y,weights = self.weight_prob)
            print(self.weight_prob)
            print(y_mean)
            # assert 0
            y_11 = np.mean(Y)
            # print(self.weight_prob)
            # print(y_mean)
            # print(y_11)
            # assert 0
            Y1 = self.machine_prob[label]
            #print(self.human_prob[label])
            X = np.arange(len(Y))

            bar_width = 0.25
            tick_label = ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20']

            #显示每个柱的具体高度
            # for x,y in zip(X,Y):
            #     plt.text(x+0.005,y+0.005,'%.3f' %y, ha='center',va='bottom')

            # for x,y1 in zip(X,Y1):
            #     plt.text(x+0.24,y1+0.005,'%.3f' %y1, ha='center',va='bottom')
            
            #绘制柱状图    
            plt.plot(X, self.aih,  color="red", label="Human Inquire ", linewidth=1)
            plt.plot(X, self.aah,  color="green", label="Human Activate ", linewidth=1)
            # plt.bar(X+bar_width, Y1, bar_width, color="green", align="center", \
            #         label="Machine", alpha=0.5)
            plt.xticks(X+bar_width/2, tick_label)


        plt.xlabel("Turn Index")
        plt.ylabel("Relative Rate")
        # plt.ylabel("Average Activate Rate")
        # plt.ylabel("Average Inquire Accurary")
        plt.title('Pilot Process')
        # plt.title('Pilot Process' + "(human:{:.1f}%)".format(y_mean*100))

        
        
        plt.legend(loc="best")
        # plt.grid(True)
        plt.savefig(save_name,dpi=400)

        plt.show()

    @staticmethod
    def get_dirlist(path, key_word_list=None, no_key_word_list=None):
        file_name_list = os.listdir(path)  # 获得原始json文件所在目录里面的所有文件名称
        if key_word_list == None and no_key_word_list == None:
            temp_file_list = file_name_list
        elif key_word_list != None and no_key_word_list == None:
            temp_file_list = []
            for file_name in file_name_list:
                have_key_words = True
                for key_word in key_word_list:
                    if key_word not in file_name:
                        have_key_words = False
                        break
                    else:
                        pass
                if have_key_words == True:
                    temp_file_list.append(file_name)
        elif key_word_list == None and no_key_word_list != None:
            temp_file_list = []
            for file_name in file_name_list:
                have_no_key_word = False
                for no_key_word in no_key_word_list:
                    if no_key_word in file_name:
                        have_no_key_word = True
                        break
                if have_no_key_word == False:
                    temp_file_list.append(file_name)
        elif key_word_list != None and no_key_word_list != None:
            temp_file_list = []
            for file_name in file_name_list:
                have_key_words = True
                for key_word in key_word_list:
                    if key_word not in file_name:
                        have_key_words = False
                        break
                    else:
                        pass
                have_no_key_word = False
                for no_key_word in no_key_word_list:
                    if no_key_word in file_name:
                        have_no_key_word = True
                        break
                    else:
                        pass
                if have_key_words == True and have_no_key_word == False:
                    temp_file_list.append(file_name)

        return temp_file_list


if __name__ == "__main__":
    # file_name = "./../model/dqn/learning_rate/learning_rate_d4_e999_agent1_dqn1.p"
    # file_name = "/Users/qianlong/Desktop/learning_rate_d4_e_agent1_dqn1_T22_lr0.001_SR44_mls0_gamma0.95_epsilon0.1_1499.p"
    # save_name = file_name + ".png"
    # ploter = Ploter(file_name)
    # ploter.load_data(performance_file=file_name, label="DQN Agent")
    # ploter.plot(save_name, label_list=["DQN Agent"])

    # ploter.load_data("./../model/dqn/learning_rate/learning_rate_d7_e999_agent1_dqn1.p",label="d7a1q1")
    # ploter.load_data("./../model/dqn/learning_rate/learning_rate_d10_e999_agent1_dqn0.p",label="d10a1q0")
    # ploter.load_data("./../model/dqn/learning_rate/learning_rate_d10_e999_agent1_dqn1.p",label="d10a1q1")
    # ploter.plot(save_name, label_list=["d7a1q0", "d7a1q1", "d10a1q0", "d10a1q1"])


    # Draw learning curve from directory.
    path = "/home/zxh/HRL_ppo/src/model/DQN/performance_new/"
    save_path = "/home/zxh/HRL_ppo/src/model/DQN/performance_new/"
    no_key_word_list = ["_99.", "_199.", "_299.", "_399."]
    performance_file_list = Ploter.get_dirlist(path=path,key_word_list=["_process","51613520"],no_key_word_list=["group"])
    print("file_number:", len(performance_file_list))
    time.sleep(8)

    for file_name in performance_file_list:
        print(file_name)
        performance_file = path + file_name
        save_name = save_path + file_name[0:25] + ".png"
        ploter = Ploter(performance_file=performance_file)
        # ploter.load_data(performance_file=performance_file,label="DQN Agent")
        # ploter.plot(save_name=save_name,label_list=["DQN Agent"])
        # ploter.load_group_data(performance_file=performance_file,label="DQN Agent")
        # ploter.plot_group(save_name=save_name,label_list=["DQN Agent"])
        ploter.load_mid_data(performance_file=performance_file,label="DQN Agent")
        ploter.load_mid_activator_data(performance_file=performance_file,label="DQN Agent")
        # ploter.load_mid_ture_data(performance_file=performance_file,label="DQN Agent")
        ploter.plot_mid(save_name=save_name,label_list=["DQN Agent"])