import numpy as np
import copy
import sys, os
import random
import re
import pickle
import math
import time
from collections import deque, Counter

sys.path.append(os.getcwd().replace("src/dialogue_system/agent", ""))
from src.dialogue_system.agent.agent_dqn import AgentDQN as LowerAgent
from src.dialogue_system.agent.agent_rule import AgentRule as LowerAgent2
from src.dialogue_system.policy_learning.dqn_torch import DQN, DQN2
from src.dialogue_system.policy_learning.ppo_torch import PPO
from src.dialogue_system.agent.utils import state_to_representation_last, reduced_state_to_representation_last
from src.dialogue_system import dialogue_configuration

from src.dialogue_system.agent.prioritized_new import PrioritizedReplayBuffer


class AgentHRL_joint2(object):
    def __init__(self, action_set, slot_set, disease_symptom, parameter):
        self.parameter = parameter
        self.action_set = action_set
        self.slot_set = slot_set
        if "disease" in self.slot_set.keys():
            self.slot_set.pop("disease")
        self.disease_symptom = disease_symptom
        self.master_experience_replay_size = 10000
        self.experience_replay_pool = deque(maxlen=self.master_experience_replay_size)
        self.activator_m_experience_replay_pool = deque(maxlen=self.master_experience_replay_size)
        self.activator_h_experience_replay_pool = deque(maxlen=self.master_experience_replay_size)
        if self.parameter.get("data_type")=='simulated':
            self.input_size_dqn_all = {1: 374, 4: 494, 5: 389, 6: 339, 7: 279, 12: 304, 13: 359, 14: 394, 19: 414}
        elif self.parameter.get("data_type") == 'real':
            self.input_size_dqn_all = {99:1}
        else:
            raise ValueError

        self.id2disease = {}
        self.id2lowerAgent = {}
        self.id2workernet= {}
        self.pretrained_lowerAgent = {}
        self.master_action_space = []
        self.lower_agent_is_human = False
        self.reward_record = []
        temp_parameter = {}
        # Worker human
        label_all_path = self.parameter.get("file_all")
        disease_symptom = pickle.load(open(os.path.join(label_all_path, 'disease_symptom.p'), 'rb'))
        slot_set = pickle.load(open(os.path.join(label_all_path, 'slot_set.p'), 'rb'))
        action_set = pickle.load(open(os.path.join(label_all_path, 'action_set.p'), 'rb'))
        self.lowerHuman = LowerAgent2(action_set=action_set, slot_set=slot_set,
                                            disease_symptom=disease_symptom, parameter=copy.deepcopy(parameter),
                                            disease_as_action=False)
        # Worker machine
        for key, value in self.input_size_dqn_all.items():
            label = str(key)
            self.master_action_space.append(label)
            label_all_path = self.parameter.get("file_all")
            label_new_path = os.path.join(label_all_path, 'label' + str(label))
            disease_symptom = pickle.load(open(os.path.join(label_new_path, 'disease_symptom.p'), 'rb'))
            slot_set = pickle.load(open(os.path.join(label_new_path, 'slot_set.p'), 'rb'))
            action_set = pickle.load(open(os.path.join(label_new_path, 'action_set.p'), 'rb'))

            temp_parameter[label] = copy.deepcopy(parameter)
            path_list = parameter['saved_model'].split('/')
            path_list.insert(-1, 'lower')
            path_list.insert(-1, str(label))
            temp_parameter[label]['saved_model'] = '/'.join(path_list)
            temp_parameter[label]['gamma'] = temp_parameter[label]['gamma_worker']  # discount factor for the lower agent.

            temp_parameter[label]["input_size_dqn"] = self.input_size_dqn_all[int(label)]
            
            self.id2lowerAgent[label] = LowerAgent(action_set=action_set, slot_set=slot_set,
                                                disease_symptom=disease_symptom, parameter=temp_parameter[label],
                                                disease_as_action=False)
        
            

        # self.mulitworker = multi_worker_model(self.id2workernet)
        # Master / Activator policy.
        input_size = (len(self.slot_set)) * 3    
        hidden_size = parameter.get("hidden_size_dqn", 300)
        self.output_size = len(self.input_size_dqn_all)  
        #print("input_size",input_size)
        self.activator_m = DQN2(input_size=198,
                           hidden_size=hidden_size,
                           output_size=self.output_size,
                           parameter=parameter,
                           named_tuple=('state', 'agent_action', 'reward', 'next_state', 'episode_over'))
        
        self.master = PPO(input_size=input_size,
                           hidden_size=300,
                           output_size=3,
                           parameter=parameter)
        self.parameter = parameter
        
        self.current_lower_agent_id = -1
        self.behave_prob = 1
        print("inquire DQN:", self.master_action_space)
        self.count = 0
        self.subtask_terminal = True
        self.subtask_turn = 0
        self.subtask_max_turn = parameter.get("subtask_max_turn")
        self.past_lower_agent_pool = {key: 0 for key in self.id2lowerAgent.keys()}

        if parameter.get("classifier_training") is True and parameter.get("train_mode") is True:
            print("########## Lower machine model is restore now ##########")
            for label, agent in self.id2lowerAgent.items():
                #print(temp_parameter[label])
                
                machine_trained_path = parameter['saved_master_trained'].split('/')
                machine_trained_path.insert(-1, 'lower')
                machine_trained_path.insert(-1, str(label))
                machine_trained_path = '/'.join(machine_trained_path)
                # rootdir = os.path.join(parameter['saved_machine_trained'], str(label))
                # list_r = os.listdir(rootdir)
                # machine_trained_path = os.path.join(rootdir, list_r[0])
                # print(machine_trained_path)
                # assert 0
                self.id2lowerAgent[label].dqn.restore_model(machine_trained_path)
                self.id2lowerAgent[label].dqn.current_net.eval()
                self.id2lowerAgent[label].dqn.target_net.eval()
            #  activator trained
            machine_trained_path = parameter['saved_master_trained'].split('/')
            machine_trained_path.insert(-1, 'lower')
            machine_trained_path.insert(-1, 'activator_m')
            machine_trained_path = '/'.join(machine_trained_path)
            self.activator_m.restore_model(machine_trained_path)
            self.activator_m.current_net.eval()
            self.activator_m.target_net.eval()
            self.master.load_model(parameter.get("saved_master_trained"))
            # self.master.actor.eval()
            # self.master.critic.eval()

            
        if parameter.get("machine_trained") is True and parameter.get("train_mode") is True:
            print("########## Lower machine model is restore now ##########")
            for label, agent in self.id2lowerAgent.items():
                #print(temp_parameter[label])
                
                machine_trained_path = parameter['saved_machine_trained'].split('/')
                machine_trained_path.insert(-1, 'lower')
                machine_trained_path.insert(-1, str(label))
                machine_trained_path = '/'.join(machine_trained_path)
                # rootdir = os.path.join(parameter['saved_machine_trained'], str(label))
                # list_r = os.listdir(rootdir)
                # machine_trained_path = os.path.join(rootdir, list_r[0])
                # print(machine_trained_path)
                # assert 0
                self.id2lowerAgent[label].dqn.restore_model(machine_trained_path)
                self.id2lowerAgent[label].dqn.current_net.eval()
                self.id2lowerAgent[label].dqn.target_net.eval()
            #  activator trained
            # machine_trained_path = parameter['saved_machine_trained'].split('/')
            # machine_trained_path.insert(-1, 'lower')
            # machine_trained_path.insert(-1, 'activator_m')
            # machine_trained_path = '/'.join(machine_trained_path)
            self.activator_m.restore_model(parameter['saved_machine_trained'])
            self.activator_m.current_net.eval()
            self.activator_m.target_net.eval()
            #
        
        if parameter.get("train_mode") is False:
            print("########## master/activator model is restore now ##########")
            self.master.load_model(parameter.get("saved_model"))
            self.master.actor.eval()
            self.master.critic.eval()
            temp_save_model_path = parameter.get("saved_model")# 'lower/activator_m'
            path_ac_list = temp_save_model_path.split('/')
            path_ac_list.insert(-1, 'lower')
            path_ac_list.insert(-1, 'activator_m')
            temp_save_model_path = '/'.join(path_ac_list)
            self.activator_m.restore_model(temp_save_model_path)
            self.activator_m.current_net.eval()
            self.activator_m.target_net.eval()
            
            for label, agent in self.id2lowerAgent.items():
                #print(temp_parameter[label])
                self.id2lowerAgent[label].dqn.restore_model(temp_parameter[label]['saved_model'])
                self.id2lowerAgent[label].dqn.current_net.eval()
                self.id2lowerAgent[label].dqn.target_net.eval()

        self.agent_action = {
            "turn": 1,
            "action": None,
            "request_slots": {},
            "inform_slots": {},
            "explicit_inform_slots": {},
            "implicit_inform_slots": {},
            "speaker": "agent"
        }

    def initialize(self):
        """
        Initializing an dialogue session.
        :return: nothing to return.
        """
        self.candidate_disease_list = []
        self.candidate_symptom_list = []
        self.agent_action = {
            "turn": None,
            "action": None,
            "request_slots": {},
            "inform_slots": {},
            "explicit_inform_slots": {},
            "implicit_inform_slots": {},
            "speaker": "agent"
        }
        self.subtask_terminal = True
        self.subtask_turn = 0
        self.master_reward = 0
        self.real_master_reward = 0
        self.ban_human = False

    def next(self, state, turn, greedy_strategy, **kwargs):
        """
        Taking action based on different methods, e.g., DQN-based AgentDQN, rule-based AgentRule.
        Detail codes will be implemented in different sub-class of this class.
        :param state: a vector, the representation of current dialogue state.
        :param turn: int, the time step of current dialogue session.
        :return: the agent action, a tuple consists of the selected agent action and action index.
        """
        # for agent_rule(doctor)
        disease_tag = kwargs.get("disease_tag")
        
        # end1 = time.time()
        # print(disease_tag)
        # assert 0
        # represent the master state into a vector first

        # print(state["turn"])
        if self.parameter.get("state_reduced"):
            try:
               self.slot_set.pop("disease")
            except:
               pass
            state_rep = reduced_state_to_representation_last(state=state,
                                                             slot_set=self.slot_set,
                                                             parameter=self.parameter)  # sequence representation.
        else:
            state_rep = state_to_representation_last(state=state,
                                                     action_set=self.action_set,
                                                     slot_set=self.slot_set,
                                                     disease_symptom=self.disease_symptom,
                                                     max_turn=self.parameter["max_turn"])  # sequence representation.
        # print(len(state_rep))
        # end2 = time.time()
        # Only when the subtask is terminal, master agent takes an action.
        if self.subtask_terminal == True:
            self.master_state = copy.deepcopy(state)
            #print(len(state_rep))
            self.__master_next(state_rep=state_rep, greedy_strategy=greedy_strategy,turn = turn)
            # end3 = time.time()
            self.subtask_terminal = False
            self.subtask_turn = 0
        
        # The selected lower agent takes an agent.
        # symptom_dist = self.disease_to_symptom_dist[self.id2disease[self.current_lower_agent_id]]
        # 在state_to_representation_last的步骤中，可以自动将不属于slot set中的slot去除掉
        activator_action_index = -1
        self.activator_h_action_index = -1
        self.activator_m_action_index = -1
        
        if self.master_action_index > 1:  # The disease classifier is activated.
            agent_action = {'action': 'inform', 'inform_slots': {"disease": 'UNK'}, 'request_slots': {},
                            "explicit_inform_slots": {}, "implicit_inform_slots": {}}
            agent_action["turn"] = turn
            agent_action["inform_slots"] = {"disease": None}
            agent_action["speaker"] = 'agent'
            agent_action["action_index"] = None
            lower_action_index = -1
            self.subtask_terminal = True
            #print("********************************************************************")
        else:
            #print("**",self.master_action_index)
            if self.master_action_index == 0:
                if self.parameter.get("machine_trained") is True:
                    greedy_strategy = False
                self.__activator_m_next(state_rep=state_rep, greedy_strategy=greedy_strategy)
                activator_action_index = self.activator_m_action_index
                self.current_lower_agent_id = self.master_action_space[self.activator_m_action_index]
                self.lower_agent_is_human = False
                self.subtask_turn += 1
                agent_action, lower_action_index = self.id2lowerAgent[str(self.current_lower_agent_id)].next(state, self.subtask_turn, greedy_strategy=greedy_strategy,disease_tag = disease_tag)
            else:
                self.__activator_h_next(state_rep=state_rep, greedy_strategy=greedy_strategy)
                # 人类
                self.lower_agent_is_human = True
                self.subtask_turn += 1
                agent_action, lower_action_index, self.ban_human = self.lowerHuman.next(state, self.subtask_turn, greedy_strategy=greedy_strategy,disease_tag = disease_tag)
                if self.ban_human is True:
                    self.subtask_terminal = True
                    self.subtask_turn = 0


            if self.subtask_turn >= self.subtask_max_turn:
                self.subtask_terminal = True
                self.subtask_turn = 0
        # end4 = time.time()
        # print('程序运行时间为: %s Seconds'%(end-end5))
        # print('程序运行时间为: %s Seconds'%(end5-end4))
        # print('程序运行时间为: %s Seconds'%(end4-end3))
        # print('程序运行时间为: %s Seconds'%(end3-end2))
        # print('程序运行时间为: %s Seconds'%(end2-end1))
        # assert 0
        return agent_action, self.master_action_index, activator_action_index, lower_action_index
    def parallel_next(self, goal):
        state_rep = reduced_state_to_representation_last(state=self.master_state,
                                                             slot_set=self.slot_set,
                                                             parameter=self.parameter)
        turn = 0
        subtask = True
        expect_state = copy.deepcopy(self.master_state)
        # self.__activator_m_next(state_rep=state_rep, greedy_strategy=greedy_strategy)
        # self.current_lower_agent_id = self.master_action_space[self.activator_m_action_index]
        while turn < self.subtask_max_turn and subtask :
            agent_action, lower_action_index = self.id2lowerAgent[str(99)].next(expect_state, turn, greedy_strategy=False)
            action = list(agent_action['request_slots'].keys())[0]
            action_value = "I don't know."
            if goal['goal']['implicit_inform_slots'] is not None and action in goal['goal']['implicit_inform_slots'].keys():
                action_value = goal['goal']['implicit_inform_slots'][action]
                subtask = False
                expect_state["current_slots"]["inform_slots"][action] = action_value
                # print(expect_state["current_slots"]["inform_slots"])
                # print(action)
                # print(goal['goal']['implicit_inform_slots'])
                # assert 0
            else:
                expect_state["current_slots"]["inform_slots"][action] = action_value
            
            turn += 1

        return expect_state["current_slots"]["inform_slots"]


    def __master_next(self, state_rep, greedy_strategy, turn):
        # Master agent takes an action.
        epsilon = self.parameter.get("epsilon")
        #print(greedy_strategy)
        mask_human = False
        if turn <= 7:
            mask_human = True
        if greedy_strategy == True:
            self.master_action_index = self.master.predict(state=state_rep, mask_human = mask_human)
        # Evaluating mode.
        else:
            self.master_action_index = self.master.predict_eval(state=state_rep, mask_human = mask_human)
        # if self.master_action_index == 1 :
        #     self.master_action_index = 0
        self.master_action_index = 0
        if self.ban_human is True and self.master_action_index == 1:
            self.master_action_index = 2
        # print('程序运行时间为: %s Seconds'%(end3-end2))
        # print('程序运行时间为: %s Seconds'%(end2-end1))
        # assert 0
        
    def __activator_m_next(self, state_rep, greedy_strategy):
        # Master agent takes an action.
        epsilon = self.parameter.get("epsilon")
        #print(greedy_strategy)
        self.activator_m_action_index = 0
        # if greedy_strategy == True:
        #     greedy = random.random()
        #     if greedy < epsilon:
        #         self.activator_m_action_index = random.randint(0, 1)
        #     else:
        #         self.activator_m_action_index = self.activator_m.predict(Xs=[state_rep])[1]
        # # Evaluating mode.
        # else:
        #     self.activator_m_action_index = self.activator_m.predict(Xs=[state_rep])[1]

    def __activator_h_next(self, state_rep, greedy_strategy):
        
        self.activator_h_action_index = 0
        


    def train(self):
        """
        Training the agent.
        Args:
            batch: the sample used to training.
        Return:
             dict with a key `loss` whose value it a float.
        """
        ave_master_reward = 0
        cur_bellman_err = 0
        mlen = 0
        icm_reward = 0
        ret = self.master.update()
        if ret is not None:
            ave_master_reward = ret["reward"]
            self.reward_record.append(ave_master_reward)
            # if len(self.reward_record) % 1000 == 999:
            #     pickle.dump(file=open(os.path.join('/home/zxh/HRL_ppo/src/dialogue_system/res/'+'ride/'+'ride_9_'+str(len(self.reward_record))+".p"), "wb"), obj=self.reward_record)
            cur_bellman_err = ret["loss"]
            mlen = ret['len']
            icm_reward = ret['reward_imc']
        if cur_bellman_err != 0:
            print("[Master agent] cur bellman err %.4f, experience replay pool %s, ave return %.4f, ave intrinsic reward %.4f" % (
                float(cur_bellman_err) , mlen,float(ave_master_reward),float(icm_reward)))

    def update_target_network(self):
        self.activator_m.update_target_network()
        
        for key in self.id2lowerAgent.keys():
            if 'h' not in key:
                self.id2lowerAgent[key].update_target_network()

    def save_model(self, model_performance, episodes_index, checkpoint_path=None):
        # Saving master agent
        self.master.save_model(model_performance=model_performance, episodes_index=episodes_index,
                               checkpoint_path=checkpoint_path)
        # Saving activator agent
        temp_checkpoint_path = os.path.join(checkpoint_path, 'lower/activator_m')
        self.activator_m.save_model(model_performance=model_performance, episodes_index=episodes_index,
                               checkpoint_path=temp_checkpoint_path)
        
        # Saving lower agent
        for key, lower_agent in self.id2lowerAgent.items():
            if 'h' not in key:
                temp_checkpoint_path = os.path.join(checkpoint_path, 'lower/' + str(key))
                lower_agent.dqn.save_model(model_performance=model_performance, episodes_index=episodes_index,
                                        checkpoint_path=temp_checkpoint_path)

    def train_dqn(self):
        """
        Train dqn.
        :return:
        """
        # ('state', 'agent_action', 'reward', 'next_state', 'episode_over')
        # Training of master agent
        cur_bellman_err = 0.0
        ave_master_reward = 0.0 
        batch_size = self.parameter.get("batch_size", 16)
        # train master
        # if self.parameter.get("classifier_training") is False:
        #     ret = self.master.update()
        #     if ret is not None:
        #         ave_master_reward = ret["reward"]
        #         cur_bellman_err = ret["loss"]

        #     print("[Master agent] cur bellman err %.4f, experience replay pool %s, ave reward %.4f" % (
        #         float(cur_bellman_err) , len(self.master.memory),float(ave_master_reward)))
        if self.count % 10 == 6 and self.parameter.get("machine_trained") is False:
            # train activator m
            cur_bellman_err = 0.0
            for iter in range(math.ceil(len(self.activator_m_experience_replay_pool) / batch_size)):
                batch = random.sample(self.activator_m_experience_replay_pool, min(batch_size, len(self.activator_m_experience_replay_pool)))
                loss = self.activator_m.singleBatch(batch=batch, params=self.parameter,
                                        weight_correction=self.parameter.get("weight_correction"))
                cur_bellman_err += loss["loss"]
            print("[Activator m agent] cur bellman err %.4f, experience replay pool %s" % (
                float(cur_bellman_err) / (len(self.activator_m_experience_replay_pool) + 1e-10), len(self.activator_m_experience_replay_pool)))
        
        # train machine
        if self.count % 10 == 9 and self.parameter.get("machine_trained") is False:
        # if self.count % 25 == 0 and self.count > 1000:
            #print(len(self.id2lowerAgent))
            for group_id, lower_agent in self.id2lowerAgent.items():
                
                if 'h' not in group_id:
                    if len(lower_agent.experience_replay_pool) > 150:
                        #多任务学习
                        # self.mulitworker.train_workers(20, group_id)
                        lower_agent.train_dqn(label=group_id)
                        self.past_lower_agent_pool[group_id] = len(lower_agent.experience_replay_pool)

        self.count += 1
        # Training of lower agents.
        # for disease_id, lower_agent in self.id2lowerAgent.items():
        #    lower_agent.train_dqn()

    def reward_shaping(self, state, next_state):
        def delete_item_from_dict(item, value):
            new_item = {}
            for k, v in item.items():
                if v != value: new_item[k] = v
            return new_item

        # slot number in state.
        slot_dict = copy.deepcopy(state["current_slots"]["inform_slots"])
        slot_dict.update(state["current_slots"]["explicit_inform_slots"])
        slot_dict.update(state["current_slots"]["implicit_inform_slots"])
        slot_dict.update(state["current_slots"]["proposed_slots"])
        slot_dict.update(state["current_slots"]["agent_request_slots"])
        slot_dict = delete_item_from_dict(slot_dict, dialogue_configuration.I_DO_NOT_KNOW)

        next_slot_dict = copy.deepcopy(next_state["current_slots"]["inform_slots"])
        next_slot_dict.update(next_state["current_slots"]["explicit_inform_slots"])
        next_slot_dict.update(next_state["current_slots"]["implicit_inform_slots"])
        next_slot_dict.update(next_state["current_slots"]["proposed_slots"])
        next_slot_dict.update(next_state["current_slots"]["agent_request_slots"])
        next_slot_dict = delete_item_from_dict(next_slot_dict, dialogue_configuration.I_DO_NOT_KNOW)
        gamma = self.parameter.get("gamma")
        return gamma * len(next_slot_dict) - len(slot_dict)

    def record_training_sample(self, state, agent_action, reward, next_state, episode_over, lower_reward, master_action_index, activator_action_index,imp_recall,state_turn,reward_for_activate_doctor_amend):
        # samples of master agent.
        
        
        
        shaping = self.reward_shaping(state, next_state)
        alpha = self.parameter.get("weight_for_reward_shaping")
        if episode_over is True:
            
            
            pass
        else:
            
            
            reward =  alpha * shaping
            

        # samples of lower agent.
        
        if int(agent_action) >= 0 :
            
            if -1 != self.current_lower_agent_id:
                self.id2lowerAgent[self.current_lower_agent_id].record_training_sample(state, agent_action, lower_reward,
                                                                                    next_state, episode_over)

        if self.parameter.get("state_reduced"):
            state_rep = reduced_state_to_representation_last(state=state,
                                                             slot_set=self.slot_set, parameter=self.parameter)  # sequence representation.
            next_state_rep = reduced_state_to_representation_last(state=next_state, slot_set=self.slot_set, parameter=self.parameter)
            master_state_rep = reduced_state_to_representation_last(state=self.master_state, slot_set=self.slot_set, parameter=self.parameter)
        # print(self.master_state["current_slots"]["inform_slots"])
        # assert 0
        self.master_reward += reward
        self.real_master_reward += reward
        
        # samples of second agent.
        if self.subtask_terminal and int(agent_action) >= 0:
            if self.master_action_index == 2:
                assert 0
            
            if self.master_reward >-60 and self.master_reward <=0:
                self.master_reward = self.master_reward /4
            
            if self.master_action_index == 0:
                self.activator_m_experience_replay_pool.append((master_state_rep, activator_action_index, self.master_reward, next_state_rep, episode_over, 0))
            elif self.master_action_index == 1:
                self.activator_h_experience_replay_pool.append((master_state_rep, activator_action_index - 9, self.master_reward, next_state_rep, episode_over, 0))
        # samples of frist agent.
        if self.subtask_terminal or episode_over is True:
            mask_human = False
            all_human = False
            if state_turn <= 7:
                mask_human = True
            # else:
            #     all_human = True
            if episode_over is True:# 达到最大轮次或激活了疾病分类器
                if master_action_index !=2:
                    self.real_master_reward = 0
                self.real_master_reward = max(0, self.real_master_reward )
                # else:
                if imp_recall >= 0.99 :
                    self.real_master_reward += self.parameter.get("reward_for_success")  
                self.master.memory.push((master_state_rep, master_action_index, self.real_master_reward, episode_over, self.master.probs, self.master.log_probs, [all_human,mask_human,0], next_state_rep))
                # end1 = time.time()
                self.train()
                # end2 = time.time()
                # print('程序运行时间为: %s Seconds'%(end2-end1))
            else:# 未结束表示，激活的是activator
                if self.master_action_index == 1:
                    # doctor_cost = self.parameter.get("reward_for_activate_doctor")
                    self.real_master_reward = self.real_master_reward + self.parameter.get("reward_for_activate_doctor") + reward_for_activate_doctor_amend
                self.master.memory.push((master_state_rep, master_action_index, self.real_master_reward, episode_over, self.master.probs, self.master.log_probs, [all_human,mask_human,0], next_state_rep))
            
            
            self.master_reward = 0
            self.real_master_reward = 0


    def flush_pool(self):
        self.experience_replay_pool = deque(maxlen=self.master_experience_replay_size)
        self.activator_m_experience_replay_pool = deque(maxlen=self.master_experience_replay_size)
        self.activator_h_experience_replay_pool = deque(maxlen=self.master_experience_replay_size)
    
    def flush_lower_pool(self):
        self.activator_m_experience_replay_pool = deque(maxlen=self.master_experience_replay_size)
        self.activator_h_experience_replay_pool = deque(maxlen=self.master_experience_replay_size)

    def train_mode(self):
        # self.master.actor.train()
        self.activator_m.current_net.train()
        

    def eval_mode(self):
        # self.master.actor.eval()
        self.activator_m.current_net.eval()
        





