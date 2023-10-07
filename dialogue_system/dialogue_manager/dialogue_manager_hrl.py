# -*- coding:utf-8 -*-
"""
dialogue manager for hierarchical reinforcement learning policy
"""

import copy
import random
from collections import deque
import sys, os
import time

sys.path.append(os.getcwd().replace("src/dialogue_system/dialogue_manager",""))

from src.dialogue_system.state_tracker import StateTracker as StateTracker
from src.dialogue_system import dialogue_configuration
from src.dialogue_system.disease_classifier import dl_classifier
from src.dialogue_system.agent.utils import state_to_representation_last, reduced_state_to_representation_last
import numpy as np
from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, classification_report, accuracy_score
import pickle
import torch
import torch.nn.functional as F

class DialogueManager_HRL(object):
    """
    Dialogue manager of this dialogue system.
    """
    def __init__(self, user, agent, parameter):
        
        self.state_tracker = StateTracker(user=user, agent=agent, parameter=parameter)
        
        self.parameter = parameter
        self.experience_replay_pool = deque(maxlen=self.parameter.get("experience_replay_pool_size"))
        self.inform_wrong_disease_count = 0
        self.dialogue_output_file = parameter.get("dialogue_file")
        self.save_dialogue = parameter.get("save_dialogue")
        self.action_history = []
        self.master_action_history = []
        self.now_imp_recall = 0
        self.lower_action_history = []
        self.group_id_match = 0
        self.repeated_action_count = 0
        self.reward_for_activate_doctor_amend = 0
        self.slot_set = pickle.load(open(self.parameter.get("slot_set"), 'rb'))
        self.disease_symptom = pickle.load(open(self.parameter.get("disease_symptom"),'rb'))
        if 'disease' in self.slot_set.keys():
            self.slot_set.pop('disease')
        self.id2disease = {}
        self.disease2id = {}
        for disease, v in self.disease_symptom.items():
            self.id2disease[v['index']] = disease
            self.disease2id[disease] = v['index']
        self.disease_replay = {'train': deque(maxlen=50000), 'test': deque(maxlen=500)}
        self.worker_right_inform_num = 0
        self.acc_by_group = {x:[0,0,0] for x in ['12', '13', '14', '19', '1', '4', '5', '6', '7',
        'h12', 'h13', 'h14', 'h19', 'h1', 'h4', 'h5', 'h6', 'h7']}
        # 这里的三维向量分别表示inform的症状正确的个数、group匹配正确的个数、隶属于某个group的个数


        if self.parameter.get("train_mode")==False:
            self.test_by_group = {x:[0,0,0] for x in ['12', '13', '14', '19', '1', '4', '5', '6', '7','h12', 'h13', 'h14', 'h19', 'h1', 'h4', 'h5', 'h6', 'h7']}
            #这里的三维向量分别表示成功次数、group匹配正确的个数、隶属于某个group的个数
            self.disease_record = []
            self.lower_reward_by_group = {x: [] for x in ['12', '13', '14', '19', '1', '4', '5', '6', '7','h12', 'h13', 'h14', 'h19', 'h1', 'h4', 'h5', 'h6', 'h7']}
            # self.master_index_by_group = {x:[] for x in ['12', '13', '14', '19', '1', '4', '5', '6', '7']}
            self.master_index_by_group = []
            self.symptom_by_group = {x: [0,0] for x in ['12', '13', '14', '19', '1', '4', '5', '6', '7','h12', 'h13', 'h14', 'h19', 'h1', 'h4', 'h5', 'h6', 'h7']}

    def next(self, greedy_strategy, save_record, index,mode='train'):
        """
        The next two turn of this dialogue session. The agent will take action first and then followed by user simulator.
        :param save_record: bool, save record?
        :param train_mode: bool, True: the purpose of simulation is to train the model, False: just for simulation and the
                           parameters of the model will not be updated.
        :return: immediate reward for taking this agent action.
        """
        # Agent takes action.
        def delete_item_from_dict(item, value):
            new_item = {}
            for k, v in item.items():
                if v != value: new_item[k] = v
            return new_item
        # end1 = time.time()
        lower_reward = 0
        state = self.state_tracker.get_state()
        group_id = self.state_tracker.user.goal["group_id"]
        self.master_action_space = self.state_tracker.agent.master_action_space
        state_slots = self.make_ml_features(state["current_slots"]["inform_slots"])
        state_cv_rep = self.cv.transform([state_slots]).toarray()
        all_Y, pre_disease = self.model.predict(state_cv_rep)
        disease_conf = F.softmax(all_Y).max(-1)[0].item()
        early_stop = False
        if disease_conf > 0.99:
            early_stop = True
        agent_action, master_action_index, activator_action_index, lower_action_index = self.state_tracker.agent.next(state=state, turn=self.state_tracker.turn,
                                                                                              greedy_strategy=greedy_strategy, index=index,
                                                                                              disease_tag = self.state_tracker.user.goal)
        # end2 = time.time()
        # print(self.state_tracker.user.goal["disease_tag"])
        if agent_action == -1 :
            lower_action = "ban activate human"
            action_type = 'ban'
        elif len(agent_action["request_slots"]) > 0:
            lower_action = list(agent_action["request_slots"].keys())[0]
            assert len(list(agent_action["request_slots"].keys())) == 1
            action_type = "symptom"
        elif len(agent_action["inform_slots"]) > 0:
            lower_action = list(agent_action["inform_slots"].keys())[0]
            assert len(list(agent_action["inform_slots"].keys()))==1
            action_type = "disease"
            #print("#########")
        else:
            lower_action = "return to master"
            assert agent_action["action"] == 'return'
            action_type = 'return'
            #print("***************************************************************************************")
        # print(state['turn'])
        # print("self.state_tracker.turn",self.state_tracker.turn)
        # print("['turn']",self.state_tracker.get_state()['turn'])
        # if len(self.state_tracker.user.goal['goal']['implicit_inform_slots']) == 0 or len(delete_item_from_dict(self.state_tracker.get_state()["current_slots"]["inform_slots"],dialogue_configuration.I_DO_NOT_KNOW)) == len(self.state_tracker.user.goal['goal']['explicit_inform_slots']) + len(self.state_tracker.user.goal['goal']['implicit_inform_slots']):
        #     self.auto_diagnose =True
        if self.parameter.get("disease_as_action")==False:
            if save_record == True:
                condition = False
            else:
                condition = (state['turn']==self.parameter.get("max_turn")*2 or action_type == 'ban')
            if action_type == "disease" or condition or early_stop:# or lower_action in self.lower_action_history:
                #once the action is repeated or the dialogue reach the max turn, then the classifier will output the predicted disease
                state_rep = self.current_state_representation(state)
                disease = self.state_tracker.user.goal["disease_tag"]
                state_slots = self.make_ml_features(state["current_slots"]["inform_slots"])
                state_cv_rep = self.cv.transform([state_slots]).toarray()
                # print(state_slots)
                # print(state_cv_rep)
                # assert 0
                # state_slots1 = self.make_ml_features(self.state_tracker.user.goal['goal']['explicit_inform_slots'])
                # state_cv_rep_exp = self.cv.transform([state_slots1]).toarray()
                # state_slots2 = self.make_ml_features(self.state_tracker.user.goal['goal']['implicit_inform_slots'])
                
                # state_cv_rep_all = self.cv.transform([state_slots1+' '+state_slots2]).toarray()
                
                # master_state_rep = reduced_state_to_representation_last(state=state,
                #                                                 slot_set=self.state_tracker.agent.slot_set,
                #                                                 parameter=self.parameter)
                
                
                
                all_Y, pre_disease = self.model.predict(state_cv_rep)
                
                if mode == 'train':
                    self.disease_replay['train'].append((state_cv_rep[0], self.disease2id[disease]))
                    # self.disease_replay['train'].append((state_cv_rep_exp[0], self.disease2id[disease]))
                    # self.disease_replay['train'].append((state_cv_rep_all[0], self.disease2id[disease]))
                    # self.disease_replay['train'].append((master_state_rep_hide_imp, self.disease2id[disease]))
                else:
                    self.disease_replay['test'].append((state_cv_rep[0], self.disease2id[disease]))
                lower_action_index = -1
                master_action_index = 2
                agent_action = {'action': 'inform', 'inform_slots': {"disease":self.id2disease[pre_disease[0]]}, 'request_slots': {},"explicit_inform_slots":{}, "implicit_inform_slots":{}}
                #print(agent_action)
                if self.parameter.get("train_mode") == False:
                    self.disease_record.append([disease,self.id2disease[pre_disease[0]]])
        # end3 = time.time()
        if action_type == 'ban':
            self.state_tracker.state["turn"] = self.state_tracker.turn
            self.state_tracker.turn += 1
        else:
            self.state_tracker.state_updater(agent_action=agent_action)
        # print("turn:%2d, state for agent:\n" % (state["turn"]) , json.dumps(state))

        # User takes action.

        user_action, reward, episode_over, dialogue_status = self.state_tracker.user.next(agent_action=agent_action,turn=self.state_tracker.turn)
        # end4 = time.time()
        # print(reward)
        inquire_state = False
        if user_action == -1:
            self.state_tracker.state["turn"] = self.state_tracker.turn 
            self.state_tracker.turn += 1
        else:
            if user_action["action"] == "inform":
                inquire_state = True
            self.state_tracker.state_updater(user_action=user_action)
            # if self.state_tracker.turn >40 :
            #     print(user_action["action"])
            #     print(self.state_tracker.turn)
        # print("turn:%2d, update after user :\n" % (state["turn"]), json.dumps(state))
        #print('status', dialogue_status)
        
        

        # if self.state_tracker.turn == self.state_tracker.max_turn:
        #     episode_over = True

        if master_action_index < 2:
            self.master_action_history.append(self.master_action_space[activator_action_index])
        #print(self.state_tracker.get_state()["turn"])
        #print(master_action_index)

        

        if self.parameter.get('agent_id').lower()=='agenthrljoint2' and lower_action in self.lower_action_history:
            lower_reward = self.parameter.get("reward_for_repeated_action")
            print("################################")
        self.reward_for_activate_doctor_amend = 0
        if self.parameter.get("agent_id") == "agenthrljoint2" and self.parameter.get("disease_as_action")==False and action_type != "disease":
            #print(master_action_index)
            #print('lower_action',lower_action)
            reward, lower_reward = self.next_by_hrl_joint2(dialogue_status, lower_action, state, activator_action_index, group_id, episode_over, reward)
            if self.state_tracker.agent.subtask_terminal is True and master_action_index == 1:
                another_state = self.state_tracker.agent.parallel_next(self.state_tracker.user.goal)
                # print(self.state_tracker.agent.master_state["current_slots"]["inform_slots"])
                # print(another_state)
                # print(self.state_tracker.get_state()["current_slots"]["inform_slots"])
                # print("goal", self.state_tracker.user.goal['goal']['implicit_inform_slots'])
                disease = self.state_tracker.user.goal["disease_tag"]
                disease_id = self.disease2id[disease]
                cur_state = copy.deepcopy(self.state_tracker.get_state()["current_slots"]["inform_slots"])
                another_state_rep = self.cv.transform([self.make_ml_features(another_state)]).toarray()
                all_Y1, pre_disease1 = self.model.predict(another_state_rep)
                Y1 = F.softmax(all_Y1, dim = 1).detach().cpu().numpy()[0]
                cur_state_rep = self.cv.transform([self.make_ml_features(cur_state)]).toarray()
                all_Y2, pre_disease2 = self.model.predict(cur_state_rep)
                Y2 = F.softmax(all_Y2, dim = 1).detach().cpu().numpy()[0]
                # print(disease_id)
                # print(Y1)
                # print(Y2)
                if disease_id != pre_disease1[0] and disease_id == pre_disease2[0]:
                    # print('*')
                    self.reward_for_activate_doctor_amend = 10 * 2
                elif disease_id != pre_disease1[0] and disease_id != pre_disease2[0] and Y2[disease_id] > Y1[disease_id]:
                    # print('#')
                    self.reward_for_activate_doctor_amend = 10 
                else:
                    # print('@')
                    self.reward_for_activate_doctor_amend = 0
                
        else:
            #print(agent_action)
            #print('aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa')
            if lower_action in self.action_history:
                reward = self.parameter.get("reward_for_repeated_action")
                print("************************")
                self.repeated_action_count += 1
                episode_over = True
            else:
                self.action_history.append(lower_action)
                # print('lower_action',lower_action)
                # print('self.action_history',self.action_history)
                # lower_action disease
                # self.action_history ['disease']



        if dialogue_status == dialogue_configuration.DIALOGUE_STATUS_INFORM_WRONG_DISEASE:
            self.inform_wrong_disease_count += 1

        # if dialogue_status == dialogue_configuration.DIALOGUE_STATUS_SUCCESS:
        #     print("success:", self.state_tracker.user.state)
        # elif dialogue_status == dialogue_configuration.DIALOGUE_STATUS_NOT_COME_YET:
        #     print("not come:", self.state_tracker.user.state)
        # else:
        #     print("failed:", self.state_tracker.user.state)
        # if len(self.state_tracker.user.state["rest_slots"].keys()) ==0:
        #     print(self.state_tracker.user.goal)
        #     print(dialogue_status,self.state_tracker.user.state)
        #print(reward)
        if len(self.state_tracker.user.goal['goal']['implicit_inform_slots']) == 0:
            self.now_imp_recall = 1
        else:
            self.now_imp_recall = float(float(len(delete_item_from_dict(self.state_tracker.get_state()["current_slots"]["inform_slots"],dialogue_configuration.I_DO_NOT_KNOW)) - len(self.state_tracker.user.goal['goal']['explicit_inform_slots'])) / len(self.state_tracker.user.goal['goal']['implicit_inform_slots']))
        if save_record is True and self.auto_diagnose is False:
            
            #print(self.action_history)
            if self.parameter.get("initial_symptom") is False or self.state_tracker.get_state()["turn"]==2:
                self.record_training_sample(
                    state=state,
                    agent_action=lower_action_index,
                    next_state=self.state_tracker.get_state(),
                    reward=reward,
                    episode_over=episode_over,
                    lower_reward = lower_reward,
                    master_action_index = master_action_index,
                    activator_action_index = activator_action_index,
                    imp_recall = self.now_imp_recall,
                    state_turn = self.state_tracker.turn - 2
                    )
        # end5 = time.time()
        # Record machine inquire acc
        disease = self.state_tracker.user.goal["disease_tag"]
        disease_id = self.disease2id[disease]
        cur_state = copy.deepcopy(self.state_tracker.get_state()["current_slots"]["inform_slots"])
        cur_state_rep = self.cv.transform([self.make_ml_features(cur_state)]).toarray()
        all_Y2, pre_disease2 = self.model.predict(cur_state_rep)
        if disease_id == pre_disease2[0]:
            return_disease = 1
            # print('1')
        else:
            return_disease = 0
        # Output the dialogue.
        slots_proportion_list = []
        if episode_over == True:
            self.action_history = []
            self.lower_action_history = []
            current_slots = copy.deepcopy(state["current_slots"]["inform_slots"])
            imp_acc = 0
            real_implicit_slots = len(self.state_tracker.user.goal['goal']['implicit_inform_slots'])
            real_explicit_slots = len(self.state_tracker.user.goal['goal']['explicit_inform_slots'])
            for values in current_slots.values():
                if values != dialogue_configuration.I_DO_NOT_KNOW:
                    imp_acc += 1
            num_of_all_slots = len(current_slots)
            slots_proportion_list.append(imp_acc - real_explicit_slots)  #only imp patient have
            slots_proportion_list.append(self.state_tracker.agent.ban_human)   #all ban
            slots_proportion_list.append(real_implicit_slots)    #all imp
            try:
                last_master_action = self.master_action_history[-1]
            except:
                last_master_action = -1
            if self.save_dialogue == True :
                state = self.state_tracker.get_state()
                goal = self.state_tracker.user.get_goal()
                # self.__output_dialogue(state=state, goal=goal, master_history=self.master_action_history)
            if last_master_action == group_id:
                self.group_id_match += 1
            self.master_action_history = []
            # if self.parameter.get("train_mode") == False and self.parameter.get("agent_id").lower()=="agenthrlnew2":
            #     #self.test_by_group[group_id][2] += 1
            #     #if last_master_action == group_id:
            #     #    self.test_by_group[group_id][1] += 1
            #     #    if reward == self.parameter.get("reward_for_success"):
            #     #        self.test_by_group[group_id][0] += 1
            #     if action_type == "disease":
            #         self.test_by_group[last_master_action][2] += 1
            #         if last_master_action == group_id:
            #             self.test_by_group[last_master_action][1] += 1
            #             if reward == self.parameter.get("reward_for_success"):
            #                 self.test_by_group[last_master_action][0] += 1
        # end = time.time()
        
        # print('程序运行时间为: %s Seconds'%(end-end5))
        # print('程序运行时间为: %s Seconds'%(end5-end4))
        # print('程序运行时间为: %s Seconds'%(end4-end3))
        # print('程序运行时间为: %s Seconds'%(end3-end2))
        # print('程序运行时间为: %s Seconds'%(end2-end1))
        # assert 0
        return reward, episode_over, dialogue_status, slots_proportion_list, inquire_state, return_disease

    def next_by_hrl_joint2(self, dialogue_status, lower_action, state, master_action_index, group_id, episode_over, reward):
        '''
                    self.acc_by_group[group_id][2] += 1
                    if self.master_action_space[master_action_index] == group_id:
                        self.acc_by_group[group_id][1] += 1
                        if self.lower_reward_function(state=state, next_state=self.state_tracker.get_state()) > 0:
                            self.acc_by_group[group_id][0] += 1
                    '''
        alpha = self.parameter.get("weight_for_reward_shaping")
        if dialogue_status == dialogue_configuration.DIALOGUE_STATUS_REACH_MAX_TURN:
            self.repeated_action_count += 1

        # if action_type == "return":
        #    lower_reward = alpha/2 * self.worker_right_inform_num
        # if action_type == "symptom" and self.state_tracker.agent.subtask_terminal:  #reach max turn
        #    lower_reward = -alpha
        

        lower_reward = alpha * self.lower_reward_function(state=state, next_state=self.state_tracker.get_state())
        if lower_action in self.lower_action_history:  # repeated action
            print(lower_action)
            assert 0
            lower_reward = -alpha
            self.state_tracker.agent.subtask_terminal = True
            # episode_over = True
            # self.repeated_action_count += 1
            reward = self.parameter.get("reward_for_repeated_action")
        # elif self.state_tracker.agent.subtask_terminal is False or lower_reward>0:
        else:
            ## 要么是44，要么是0
            lower_reward = max(0, lower_reward)
            if lower_reward > 0:
                self.state_tracker.agent.subtask_terminal = True
                self.worker_right_inform_num += 1
                #print(alpha,lower_reward)
                lower_reward = alpha
            self.lower_action_history.append(lower_action)
            #print("###############################",self.lower_action_history)
            # if self.parameter.get("train_mode")==False:
            #    self.lower_reward_by_group[self.master_action_space[master_action_index]].append(lower_reward)
        # else:
        #    lower_reward = -alpha
        # if self.lower_reward_function(state=state, next_state=self.state_tracker.get_state())>0:
        #   self.worker_right_inform_num += 1
        # if  dialogue_status == dialogue_configuration.DIALOGUE_STATUS_REACH_MAX_TURN:
        

        if episode_over == True:
            self.state_tracker.agent.subtask_terminal = True
            self.state_tracker.agent.subtask_turn = 0
            '''
            #reward = alpha/2 * self.worker_right_inform_num - 2
            if dialogue_status != dialogue_configuration.DIALOGUE_STATUS_REACH_MAX_TURN:
                if self.master_action_space[master_action_index] == group_id:
                    reward = 10
                else:
                    reward = -1
            self.worker_right_inform_num = 0
            #self.lower_action_history = []
            '''
        elif self.state_tracker.agent.subtask_terminal:
            # reward = alpha / 2 * self.worker_right_inform_num - 2
            '''
            if self.master_action_space[master_action_index] == group_id:
                reward = 10
            else:
                reward = -1
            '''
            self.worker_right_inform_num = 0
            if self.parameter.get("train_mode") == False:
                self.master_index_by_group.append([group_id, master_action_index])
            # self.lower_action_history = []
            #print(reward,lower_reward)
        return reward, lower_reward

    def initialize(self, dataset, goal_index=None):
        self.state_tracker.initialize()
        self.inform_wrong_disease_count = 0
        user_action = self.state_tracker.user.initialize(dataset=dataset, goal_index=goal_index)
        self.state_tracker.state_updater(user_action=user_action)
        self.state_tracker.agent.initialize()
        self.auto_diagnose = False
        # self.group_id_match = 0
        # print("#"*30 + "\n" + "user goal:\n", json.dumps(self.state_tracker.user.goal))
        # state = self.state_tracker.get_state()
        # print("turn:%2d, initialized state:\n" % (state["turn"]), json.dumps(state))

    def record_training_sample(self, state, agent_action, reward, next_state, episode_over, **kwargs):
        if self.parameter.get("agent_id").lower() in ["agenthrljoint",'agenthrljoint2']:
            lower_reward = kwargs.get("lower_reward")
            master_action_index = kwargs.get("master_action_index")
            activator_action_index = kwargs.get("activator_action_index")
            imp_recall = kwargs.get("imp_recall")
            state_turn = kwargs.get("state_turn")
            self.state_tracker.agent.record_training_sample(state, agent_action, reward, next_state, episode_over, lower_reward, master_action_index, activator_action_index,imp_recall, state_turn, self.reward_for_activate_doctor_amend)
        else:
            self.state_tracker.agent.record_training_sample(state, agent_action, reward, next_state, episode_over)

    def record_prioritized_training_sample(self, state, agent_action, reward, next_state, episode_over, TD_error, **kwargs):
        if self.parameter.get("agent_id").lower() in ["agenthrljoint",'agenthrljoint2']:
            lower_reward = kwargs.get("lower_reward")
            self.state_tracker.agent.record_prioritized_training_sample(state, agent_action, reward, next_state,
                                                                        episode_over, TD_error, lower_reward)
        else:
            self.state_tracker.agent.record_prioritized_training_sample(state, agent_action, reward, next_state, episode_over, TD_error)

    def set_agent(self,agent):
        self.state_tracker.set_agent(agent=agent)

    def train(self):
        self.state_tracker.agent.train_dqn()
        self.state_tracker.agent.update_target_network()

    def __output_dialogue(self,state, goal, master_history):
        history = state["history"]
        file = open(file=self.dialogue_output_file,mode="a+",encoding="utf-8")
        file.write("User goal: " + str(goal)+"\n")
        for turn in history:
            #print(turn)
            try:
                speaker = turn["speaker"]
            except:
                speaker = 'agent'
            action = turn["action"]
            inform_slots = turn["inform_slots"]
            request_slots = turn["request_slots"]
            if speaker == "agent":
                try:
                    master_action = master_history.pop(0)
                    file.write(speaker + ": master+ " + str(master_action) +' +'+ action + "; inform_slots:"
                               + str(inform_slots) + "; request_slots:" + str(request_slots) + "\n")
                except:
                    file.write(speaker + ": master+ " + ' +' + action + "; inform_slots:" + str(inform_slots)
                               + "; request_slots:" + str(request_slots) + "\n")
            else:
                file.write(speaker + ": " + action + "; inform_slots:" + str(inform_slots) + "; request_slots:" + str(request_slots) + "\n")
        file.write("\n\n")
        assert len(master_history) == 0
        file.close()

    def lower_reward_function(self, state, next_state):
        """
        The reward for lower agent
        :param state:
        :param next_state:
        :return:
        """
        def delete_item_from_dict(item, value):
            new_item = {}
            for k, v in item.items():
                if v != value: new_item[k] = v
            return new_item

        # slot number in state.
        slot_dict = copy.deepcopy(state["current_slots"]["inform_slots"])
        slot_dict = delete_item_from_dict(slot_dict, dialogue_configuration.I_DO_NOT_KNOW)

        next_slot_dict = copy.deepcopy(next_state["current_slots"]["inform_slots"])
        next_slot_dict = delete_item_from_dict(next_slot_dict, dialogue_configuration.I_DO_NOT_KNOW)
        gamma = self.parameter.get("gamma")
        return gamma * len(next_slot_dict) - len(slot_dict)
        #return max(0, gamma * len(next_slot_dict) - len(slot_dict))

    def current_state_representation(self, state):
        """
        The state representation for the input of disease classifier.
        :param state: the last dialogue state before fed into disease classifier.
        :return: a vector that has equal length with slot set.
        """
        assert 'disease' not in self.slot_set.keys()
        state_rep = [0]*len(self.slot_set)
        current_slots = copy.deepcopy(state['current_slots'])
        if self.parameter.get('data_type') == 'simulated':
            for slot, value in current_slots['inform_slots'].items():
                if value == True:
                    state_rep[self.slot_set[slot]] = 1
                #elif value == "I don't know.":
                #    state_rep[self.slot_set[slot]] = -1
                #else:
                #    print(value)
                #    raise ValueError("the slot value of inform slot is not among True and I don't know")
            #print(state_rep)
        elif self.parameter.get('data_type') == 'real':
            for slot, value in current_slots['inform_slots'].items():
                if value == True:
                    state_rep[self.slot_set[slot]] = 1
                elif value == False:
                    state_rep[self.slot_set[slot]] = -1
        return state_rep

    def current_state_representation_both(self, state):
        assert 'disease' not in self.slot_set.keys()
        state_rep = np.zeros((len(self.slot_set.keys()), 3))
        current_slots = copy.deepcopy(state['current_slots'])
        #for slot, value in current_slots['inform_slots'].items():
        for slot in self.slot_set:
            if slot in current_slots['inform_slots']:
                if current_slots['inform_slots'][slot] == True:
                    state_rep[self.slot_set[slot],:] = [1,0,0]
                elif current_slots['inform_slots'][slot] == dialogue_configuration.I_DO_NOT_KNOW:
                    state_rep[self.slot_set[slot],:] = [0,1,0]
                else:
                    state_rep[self.slot_set[slot],:] = [0,0,1]
            else:
                state_rep[self.slot_set[slot],:] = [0,0,1]
                #raise ValueError("the slot value of inform slot is not among True and I don't know")
        print(state_rep)
        state_rep1 = state_rep.reshape(1, len(self.slot_set.keys()) * 3)[0]
        return state_rep1
    
    def make_ml_features(self, state_slots):
        feature = []
        for k,v in state_slots.items():
            if v == True:
                feature.append(k+'-Positive')
            elif v == False:
                feature.append(k+'-Negative')
        return ' '.join(feature)
    
    def train_ml_classifier(self):
        goal_set = pickle.load(open(self.parameter.get("goal_set"),'rb'))
        disease_y = []
        total_set = random.sample(goal_set['train'], 5000)

        slots_exp = np.zeros((len(total_set), len(self.slot_set)))
        for i, dialogue in enumerate(total_set):
            tag = dialogue['disease_tag']
            # tag_group=disease_symptom1[tag]['symptom']
            disease_y.append(tag)
            goal = dialogue['goal']
            explicit = goal['explicit_inform_slots']
            for exp_slot, value in explicit.items():
                try:
                    slot_id = self.slot_set[exp_slot]
                    if value == True:
                        slots_exp[i, slot_id] = '1'
                except:
                    pass

        self.model = svm.SVC(kernel='linear', C=1)
        self.model.fit(slots_exp, disease_y)

    def build_deep_learning_classifier(self):
        # for index in ['1', '4', '5', '6', '7','12', '13', '14', '19']:
        #     self.dl_model[index] = dl_classifier(input_size=len(self.slot_set), hidden_size=256,
        #                             output_size=len(self.disease_symptom),
        #                             parameter=self.parameter)
        # print(self.dl_model.values())
        # assert 0
        # if self.parameter.get("fix_9_classifier") is True:
        #     self.dl_model['1'].restore_model("/home/zxh/new_for_mr2/src/model/DQN/checkpoint/0309185708_agenthrljoint2_T22_ss100_lr0.0005_cp20_G1_beta-0.0_smt3_RFS66_RFF0_RFNCY0.0_RFIRS0_RFRA-44_RFRMT-66_mls0_gamma0.95_gammaW0.9_epsilon0.1_awd0_crs0_hwg0_wc0_var0_sdai0_wfrs44.0_dtft0_ird0_ubc0.985_lbc1e-10_dataSynthetic_Dataset_RID0/classifier/model_d10agenthrljoint2_s0.822_r54.252_t41.0_mr0.0_mr2-0.624_e-24.pkl")
        #     self.dl_model['4'].restore_model("/home/zxh/new_for_mr2/src/model/DQN/checkpoint/0309195246_agenthrljoint2_T22_ss100_lr0.0005_cp20_G4_beta-0.0_smt3_RFS66_RFF0_RFNCY0.0_RFIRS0_RFRA-44_RFRMT-66_mls0_gamma0.95_gammaW0.9_epsilon0.1_awd0_crs0_hwg0_wc0_var0_sdai0_wfrs44.0_dtft0_ird0_ubc0.985_lbc1e-10_dataSynthetic_Dataset_RID0/classifier/model_d10agenthrljoint2_s0.956_r63.096_t41.0_mr0.0_mr2-0.653_e-28.pkl")
        #     self.dl_model['5'].restore_model("/home/zxh/new_for_mr2/src/model/DQN/checkpoint/0309200038_agenthrljoint2_T22_ss100_lr0.0005_cp20_G5_beta-0.0_smt3_RFS66_RFF0_RFNCY0.0_RFIRS0_RFRA-44_RFRMT-66_mls0_gamma0.95_gammaW0.9_epsilon0.1_awd0_crs0_hwg0_wc0_var0_sdai0_wfrs44.0_dtft0_ird0_ubc0.985_lbc1e-10_dataSynthetic_Dataset_RID0/classifier/model_d10agenthrljoint2_s0.732_r48.312_t41.0_mr0.0_mr2-0.73_e-30.pkl")
        #     self.dl_model['6'].restore_model("/home/zxh/new_for_mr2/src/model/DQN/checkpoint/0309201124_agenthrljoint2_T22_ss100_lr0.0005_cp20_G6_beta-0.0_smt3_RFS66_RFF0_RFNCY0.0_RFIRS0_RFRA-44_RFRMT-66_mls0_gamma0.95_gammaW0.9_epsilon0.1_awd0_crs0_hwg0_wc0_var0_sdai0_wfrs44.0_dtft0_ird0_ubc0.985_lbc1e-10_dataSynthetic_Dataset_RID0/classifier/model_d10agenthrljoint2_s0.86_r56.76_t41.0_mr0.0_mr2-0.747_e-22.pkl")
        #     self.dl_model['7'].restore_model("/home/zxh/new_for_mr2/src/model/DQN/checkpoint/0309205257_agenthrljoint2_T22_ss100_lr0.0005_cp20_G7_beta-0.0_smt3_RFS66_RFF0_RFNCY0.0_RFIRS0_RFRA-44_RFRMT-66_mls0_gamma0.95_gammaW0.9_epsilon0.1_awd0_crs0_hwg0_wc0_var0_sdai0_wfrs44.0_dtft0_ird0_ubc0.985_lbc1e-10_dataSynthetic_Dataset_RID0/classifier/model_d10agenthrljoint2_s0.687_r45.342_t41.0_mr0.0_mr2-0.866_e-20.pkl")
        #     self.dl_model['12'].restore_model("/home/zxh/new_for_mr2/src/model/DQN/checkpoint/0309205854_agenthrljoint2_T22_ss100_lr0.0005_cp20_G12_beta-0.0_smt3_RFS66_RFF0_RFNCY0.0_RFIRS0_RFRA-44_RFRMT-66_mls0_gamma0.95_gammaW0.9_epsilon0.1_awd0_crs0_hwg0_wc0_var0_sdai0_wfrs44.0_dtft0_ird0_ubc0.985_lbc1e-10_dataSynthetic_Dataset_RID0/classifier/model_d10agenthrljoint2_s0.769_r50.754_t41.0_mr0.0_mr2-0.857_e-35.pkl")
        #     self.dl_model['13'].restore_model("/home/zxh/new_for_mr2/src/model/DQN/checkpoint/0309214328_agenthrljoint2_T22_ss100_lr0.0005_cp20_G13_beta-0.0_smt3_RFS66_RFF0_RFNCY0.0_RFIRS0_RFRA-44_RFRMT-66_mls0_gamma0.95_gammaW0.9_epsilon0.1_awd0_crs0_hwg0_wc0_var0_sdai0_wfrs44.0_dtft0_ird0_ubc0.985_lbc1e-10_dataSynthetic_Dataset_RID0/classifier/model_d10agenthrljoint2_s0.888_r58.608_t41.0_mr0.0_mr2-0.717_e-41.pkl")
        #     self.dl_model['14'].restore_model("/home/zxh/new_for_mr2/src/model/DQN/checkpoint/0309214337_agenthrljoint2_T22_ss100_lr0.0005_cp20_G14_beta-0.0_smt3_RFS66_RFF0_RFNCY0.0_RFIRS0_RFRA-44_RFRMT-66_mls0_gamma0.95_gammaW0.9_epsilon0.1_awd0_crs0_hwg0_wc0_var0_sdai0_wfrs44.0_dtft0_ird0_ubc0.985_lbc1e-10_dataSynthetic_Dataset_RID0/classifier/model_d10agenthrljoint2_s0.907_r59.862_t41.0_mr0.0_mr2-0.733_e-50.pkl")
        #     self.dl_model['19'].restore_model("/home/zxh/new_for_mr2/src/model/DQN/checkpoint/0309215543_agenthrljoint2_T22_ss100_lr0.0005_cp20_G19_beta-0.0_smt3_RFS66_RFF0_RFNCY0.0_RFIRS0_RFRA-44_RFRMT-66_mls0_gamma0.95_gammaW0.9_epsilon0.1_awd0_crs0_hwg0_wc0_var0_sdai0_wfrs44.0_dtft0_ird0_ubc0.985_lbc1e-10_dataSynthetic_Dataset_RID0/classifier/model_d10agenthrljoint2_s0.835_r55.11_t41.0_mr0.0_mr2-0.671_e-52.pkl")
        self.model = dl_classifier(input_size=62 , hidden_size=256,
                                    output_size=len(self.disease_symptom),
                                    parameter=self.parameter)
        self.cv = CountVectorizer(tokenizer=tokenizer)
        features = []
        #'哭闹-Positive 脱水-Positive 稀便-Positive 腹泻-Positive'
        for goal in self.state_tracker.user.goal_set['train']:
            state_solts = copy.deepcopy(goal['goal']['explicit_inform_slots'])
            state_solts.update(goal['goal']['implicit_inform_slots'])
            feature = self.make_ml_features(state_solts)
            features.append(feature)
        _x_train = self.cv.fit_transform(features)
        # ex = self.cv.transform(['哭闹-Positive 脱水-Negative 稀便-Positive' ]).toarray()
        # print(len(ex[0]))
        # assert 0
        # self.model_classifier_group = dl_classifier(input_size=len(self.slot_set) * 3, hidden_size=256,
        #                             output_size=9,
        #                             parameter=self.parameter)
        # saved_model = '/home/zxh/HRL_ppo/src/model/DQN/checkpoint/0516214511_agenthrljoint2_DA0.7_RFH-14_ST3_T20_ss200_lr0.0005_wfrs10.0_RFS15_RFF0_RFNCY0.0_RFIRS0_RFRA-44_RFRMT-44_mls0_gamma1_gammaW0.9_epsilon0.1_awd0_crs0_hwg0_wc0_var0_sdai0_dtft0_ird0_ubc0.985_lbc1e-10_dataDxy_Dataset_RID0/classifier/s0.875_r13.125_t2.587_aah1.856_mr2-1.0_e-87.pkl'
        saved_model = '/home/zxh/HRL_ppo/src/dialogue_system/run/disease_classifier0.894231.pkl'
        self.model.restore_model(saved_model)
        if self.parameter.get("train_mode") == False:
            temp_path = self.parameter.get("saved_model")
            path_list = temp_path.split('/')
            path_list.insert(-1, 'classifier')
            saved_model = '/'.join(path_list)
            # saved_model = '/home/zxh/HRL/src/model/DQN/checkpoint/0502170050_agenthrljoint2_DA0.7_RFH-44_ST3_T20_ss100_lr0.0005_RFS66_RFF0_RFNCY0.0_RFIRS0_RFRA-44_RFRMT-44_mls0_gamma1_gammaW0.9_epsilon0.1_awd0_crs0_hwg0_wc0_var0_sdai0_wfrs44.0_dtft0_ird0_ubc0.985_lbc1e-10_dataMuzhi_Dataset_RID0/classifier/s0.768_r50.662_t16.965_aah0.732_mr2-0.976_e-45.pkl'
            self.model.restore_model(saved_model)
            self.model.eval_mode()
    def train_deep_learning_classifier(self, epochs):
        best_test_acc = self.model.test(test_batch=self.disease_replay['test'])
        print('best_test_acc',best_test_acc)
        loss = {}
        loss["loss"] = 0.0
        test_acc = 0
        best_model = copy.deepcopy(self.model.model)
        for iter in range(epochs):
            
            try:
                batch = random.sample(self.disease_replay['train'], min(self.parameter.get("batch_size"),len(self.disease_replay['train'])))
                
                loss = self.model.train(batch=batch)
                
            except:
                pass
            # print(loss)
            
            try:
                test_batch = self.disease_replay['test']
                test_acc = self.model.test(test_batch=test_batch)
                
                if test_acc > best_test_acc:
                    best_test_acc = test_acc
                    print("update__",test_acc)
                    best_model.load_state_dict(self.model.model.state_dict())
                    if test_acc> 0.90:
                        torch.save(self.model.model.state_dict(), 'disease_classifier'+format(test_acc,'5f')+'.pkl')
            except:
                pass
            # print(test_acc)
        # test_batch = random.sample(self.disease_replay['train'], min(500,len(self.disease_replay['train'])))
        # test_acc = self.model.test(test_batch=test_batch)

        self.model.model.load_state_dict(best_model.state_dict())
        
        
        # self.disease_replay = {'train': deque(maxlen=50000), 'test': deque(maxlen=1000)}
        print('disease_replay:{},loss:{:.4f}, test_acc:{:.4f}'.format(len(self.disease_replay['train']), loss["loss"], best_test_acc))

    def save_dl_model(self, model_performance, episodes_index, checkpoint_path=None):
        # Saving master agent
        temp_checkpoint_path = os.path.join(checkpoint_path, 'classifier/')
        self.model.save_model(model_performance=model_performance, episodes_index=episodes_index, checkpoint_path=temp_checkpoint_path)

    def exp_transform(self, x):
        exp_sum = 0
        for i in x:
            exp_sum += np.exp(i)
        return [np.exp(i)/exp_sum for i in x]

def tokenizer(x):
    return x.split()