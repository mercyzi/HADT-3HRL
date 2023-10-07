# -*-coding: utf-8 -*-

import sys
import os
import pickle
import time
import json
from collections import deque
import copy
import numpy as np

sys.path.append(os.getcwd().replace("src/dialogue_system/run",""))

from src.dialogue_system.agent import AgentRule
from src.dialogue_system.agent import AgentDQN
from src.dialogue_system.user_simulator import UserRule as User
from src.dialogue_system.dialogue_manager import DialogueManager
from src.dialogue_system.dialogue_manager import DialogueManager_HRL
from src.dialogue_system.reward_shaping import RewardModel
from src.dialogue_system import dialogue_configuration
from src.dialogue_system.run.preprocess_classifier_data import enrich_data
#from src.dialogue_system.dialogue_manager import dl_classifier
import random

class RunningSteward(object):
    """
    The steward of running the dialogue system.
    """
    def __init__(self, parameter, checkpoint_path):
        self.epoch_size = parameter.get("simulation_size",100)
        self.parameter = parameter
        self.checkpoint_path = checkpoint_path
        self.learning_curve = {}
        self.mid_process = {}
        self.each_group = {}
        self.activator_is_h = False 
        self.rec_human_arr = {1: [], 3: [], 5: [], 7: [], 9: [], 11: [], 13: [], 15: [], 17: [], 19: [],
            21: [], 23: [], 25: [], 27: [], 29: [], 31: [], 33: [], 35: [], 37: [], 39: [], 41: []}
        # self.reward_model = {}
        # self.reward_penalty = {}
        # for iter in range(1,43,2):
        #     self.reward_penalty[iter] = self.parameter["reward_for_activate_doctor"]
        #     self.reward_model[iter] = RewardModel()
        #     print(iter,self.reward_penalty[iter])
        # print(self.reward_penalty)
        # assert 0
        self.cur_mr2_group = {'1':[],'2':[], '4':[], '5':[], '6':[], '7':[],'12':[], '13':[], '14':[], '19':[],'99':[] }
        slot_set = pickle.load(file=open(parameter["slot_set"], "rb"))
        action_set = pickle.load(file=open(parameter["action_set"], "rb"))
        goal_set = pickle.load(file=open(parameter["goal_set"], "rb"))
        disease_symptom = pickle.load(file=open(parameter["disease_symptom"], "rb"))
        re_train_set = copy.deepcopy(goal_set['validate'])
        # re_train_set2 = copy.deepcopy(goal_set['train'])
        # a = 0
        # b = 0
        # ttt = json.loads(open('/home/zxh/srcformz10/src/data/mz10/train_set.json', 'r').read())
        # for goal in ttt:
        #     for k,v in goal['imp_sxs'].items():
        #         a = a + 1
        #         if v != '2':
        #             b = b + 1

        # print(b/a)
        # assert 0

        for goal in goal_set['train']:
            goal['group_id'] = '99'
            goal['goal']['request_slots'] = {}
            goal['goal']['request_slots']['disease'] = 'UNK'
            for k, v in goal['goal']['explicit_inform_slots'].items():
                if k in goal['goal']['implicit_inform_slots'].keys():
                    goal['goal']['implicit_inform_slots'].pop(k)
        # print(a)
        for goal in goal_set['test']:
            goal['group_id'] = '99'
            goal['goal']['request_slots'] = {}
            goal['goal']['request_slots']['disease'] = 'UNK'
            for k, v in goal['goal']['explicit_inform_slots'].items():
                if k in goal['goal']['implicit_inform_slots'].keys():
                    goal['goal']['implicit_inform_slots'].pop(k)  
        
        for item in re_train_set:
            
            
            goal_set['train'].append(item)

        # re_train_set2 = copy.deepcopy(goal_set['train'])
        # for item in re_train_set2:
        #     imp = {}
        #     # print(item['goal']['explicit_inform_slots'])
        #     for key,v in item['goal']['implicit_inform_slots'].items():
        #         if v is True:
        #             imp[key] = v
            
            
        #     item['goal']['implicit_inform_slots'] = item['goal']['explicit_inform_slots']
        #     item['goal']['explicit_inform_slots'] = imp
            
        #     goal_set['train'].append(item)
        # for item in re_train_set2:
        #     imp = {}
        #     # print(item['goal']['explicit_inform_slots'])
        #     for key,v in item['goal']['implicit_inform_slots'].items():
        #         if v is False:
        #             imp[key] = v
            
            
        #     item['goal']['implicit_inform_slots'] = item['goal']['explicit_inform_slots']
        #     item['goal']['explicit_inform_slots'] = imp
            
        #     goal_set['train'].append(item)
        # goal_set['train'] = enrich_data(goal_set['train'])
        # print(len(goal_set['train']))
        # assert 0

        user = User(goal_set=goal_set, disease_syptom=disease_symptom,parameter=parameter)
        agent = AgentRule(action_set=action_set, slot_set=slot_set, disease_symptom=disease_symptom, parameter=parameter)
        if parameter.get("use_all_labels"):
            self.dialogue_manager = DialogueManager_HRL(user=user, agent=agent, parameter=parameter)
        else:
            self.dialogue_manager = DialogueManager(user=user, agent=agent, parameter=parameter)
        if self.parameter.get("disease_as_action") == False:
            if self.parameter.get("classifier_type") == "machine_learning":
                self.dialogue_manager.train_ml_classifier()
                print("############   the machine learning model is training  ###########")
            elif self.parameter.get("classifier_type") == "deep_learning":
                self.dialogue_manager.build_deep_learning_classifier()
            else:
                raise ValueError("the classifier type is not among machine_learning and deep_learning")


        self.best_result = {"success_rate":0.0, "average_reward": 0.0, "average_turn": 0,"average_wrong_disease":10,"average_match_rate2":0.0,'average_inquire_human':2.0,'average_activate_human':10.0,'average_ban':1}
        self.best_lower_result = {"average_match_rate":0.0,"average_match_rate2":0.0}

    def simulate(self, epoch_number, train_mode=False):
        """
        Simulating the dialogue session between agent and user simulator.
        :param agent: the agent used to simulate, an instance of class Agent.
        :param epoch_number: the epoch number of simulation.
        :param train_mode: bool, True: the purpose of simulation is to train the model, False: just for simulation and the
                           parameters of the model will not be updated.
        :return: nothing to return.
        """
        # initializing the count matrix for AgentWithGoal
        # print('Initializing the count matrix for AgentWithGoal')
        # self.simulation_epoch(epoch_size=500, train_mode=train_mode)
        save_model = self.parameter.get("save_model")
        save_performance = self.parameter.get("save_performance")
        # self.dialogue_manager.state_tracker.user.set_max_turn(max_turn=self.parameter.get('max_turn'))
        # 预训练
        # for group_id, lower_agent in self.dialogue_manager.state_tracker.agent.id2lowerAgent.items():
        #     self.dialogue_manager.state_tracker.agent.mulitworker.train_workers(5000, group_id)
        
        best_acc = 0
        for index in range(0, epoch_number,1):
            # Training AgentDQN with experience replay
            if train_mode is True:
                # start = time.time()
                self.dialogue_manager.train()
                
                # end1 = time.time()
                # Simulating and filling experience replay pool.
                self.simulation_epoch(epoch_size=self.epoch_size, index=index)
                # end2 = time.time()
                # print('程序运行时间为: %s Seconds'%(end2-end1))
                # print('程序运行时间为: %s Seconds'%(end1-start))
                # assert 0

            # Evaluating the model.
            #print(index)
            if index > -1:
                print("activator_human_prob",[float(format(np.mean(i), '.3f')) for i in self.rec_human_arr.values()])
                # self.dialogue_manager.disease_replay ={'train': deque(maxlen=50000), 'test': deque(maxlen=1000)}
                # result = self.evaluate_model(dataset="test", index=index)#validate
                # # print(len(self.dialogue_manager.disease_replay['test']))
                # for i in range(2000):
                #     self.evaluate_model(dataset="train", index=index,mode = 'train')
                    
                #     self.dialogue_manager.train_deep_learning_classifier(epochs=500)
                result = self.evaluate_model(dataset="test", index=index)#validate
                # if result["success_rate"]> best_acc and result["average_activate_human"]  < 1.2:
                #     best_acc = result["success_rate"]
                print('best_result',self.best_result)
                # print("mr2_by_group:",[float(format(np.mean(i), '.3f')) for i in self.cur_mr2_group.values()])
                if result["average_activate_human"] < self.best_result["average_activate_human"] and \
                    result["average_turn"]  < 16 and\
                    result["success_rate"] > 0.79 and\
                        train_mode==True:
                        #result["average_wrong_disease"] <= self.best_result["average_wrong_disease"] and \
                    best_acc = result["average_ban"]
                    
                    self.dialogue_manager.state_tracker.agent.flush_pool()
                    # self.simulation_epoch(epoch_size=self.epoch_size, index=index)
                    if save_model is True:
                        self.dialogue_manager.state_tracker.agent.save_model(model_performance=result, episodes_index = index, checkpoint_path=self.checkpoint_path)
                        if self.parameter.get("agent_id").lower() in ["agenthrljoint", "agenthrljoint2",'agentdqn']:
                            self.dialogue_manager.save_dl_model(model_performance=result, episodes_index=index,
                                                                checkpoint_path=self.checkpoint_path)
                        print("###########################The model was saved.###################################")
                    else:
                        pass
                    self.best_result = copy.deepcopy(result)
        # The training is over and save the model of the last training epoch.
        if save_model is True and train_mode is True and epoch_number > 0:
            self.dialogue_manager.state_tracker.agent.save_model(model_performance=result, episodes_index=index, checkpoint_path=self.checkpoint_path)
            if self.parameter.get("agent_id").lower() in ["agenthrljoint","agenthrljoint2"]:
                self.dialogue_manager.save_dl_model(model_performance=result, episodes_index=index, checkpoint_path=self.checkpoint_path)
        if save_performance is True and train_mode is True and epoch_number > 0:
            self.__dump_performance__(epoch_index=index)

    def simulation_epoch(self, epoch_size, index):
        """
        Simulating one epoch when training model.
        :param epoch_size: the size of each epoch, i.e., the number of dialogue sessions of each epoch.
        :return: a dict of simulation results including success rate, average reward, average number of wrong diseases.
        """
        success_count = 0
        absolute_success_count = 0
        total_reward = 0
        total_turns = 0
        self.dialogue_manager.state_tracker.agent.eval_mode() # for testing
        inform_wrong_disease_count = 0
        for epoch_index in range(0,epoch_size, 1):
            # start = time.time()
            self.dialogue_manager.initialize(dataset="train")
            # start1 = time.time()
            episode_over = False
            # print(self.dialogue_manager.state_tracker.user.goal)
            while episode_over is False:
                cur_turn = self.dialogue_manager.state_tracker.turn
                # if index > 100 and epoch_index == epoch_size - 1 :
                #     if self.rec_human_arr[cur_turn]:
                #         self.reward_model[cur_turn].build_train_set(np.mean(self.rec_human_arr[cur_turn]), self.reward_penalty[cur_turn])
                # self.parameter["reward_for_activate_doctor"] = self.reward_penalty[cur_turn]
                reward, episode_over, dialogue_status,slots_proportion_list,inquire_state, return_disease= self.dialogue_manager.next(greedy_strategy=True, save_record=True, index=index)
                # print(self.dialogue_manager.state_tracker.get_state()["current_slots"]["inform_slots"])
                # print(self.dialogue_manager.state_tracker.turn)
                total_reward += reward
                
            total_turns += self.dialogue_manager.state_tracker.turn
            inform_wrong_disease_count += self.dialogue_manager.inform_wrong_disease_count
            if dialogue_status == dialogue_configuration.DIALOGUE_STATUS_SUCCESS:
                success_count += 1
                if self.dialogue_manager.inform_wrong_disease_count == 0:
                    absolute_success_count += 1
            # start2 = time.time()
            # print('1程序运行时间为: %s Seconds'%(start1-start))
            # print('2程序运行时间为: %s Seconds'%(start2-start1))
            
        success_rate = float("%.3f" % (float(success_count) / epoch_size))
        absolute_success_rate = float("%.3f" % (float(absolute_success_count) / epoch_size))
        average_reward = float("%.3f" % (float(total_reward) / epoch_size))
        average_turn = float("%.3f" % (float(total_turns) / epoch_size))
        average_wrong_disease = float("%.3f" % (float(inform_wrong_disease_count) / epoch_size))
        
        res = {"success_rate":success_rate, "average_reward": average_reward, "average_turn": average_turn,
               "average_wrong_disease":average_wrong_disease,"ab_success_rate":absolute_success_rate}
        # print("%3d simulation success rate %s, ave reward %s, ave turns %s, ave wrong disease %s" % (index,res['success_rate'], res['average_reward'], res['average_turn'], res["average_wrong_disease"]))
        self.dialogue_manager.state_tracker.agent.train_mode() # for training
        return res

    def evaluate_model(self, dataset, index, mode='test'):
        """
        Evaluating model during training.
        :param index: int, the simulation index.
        :return: a dict of evaluation results including success rate, average reward, average number of wrong diseases.
        """
        if self.parameter.get("use_all_labels"):
            self.dialogue_manager.repeated_action_count = 0
            self.dialogue_manager.group_id_match = 0
        if self.parameter.get("initial_symptom"):
            self.dialogue_manager.group_id_match = 0
        self.dialogue_manager.repeated_action_count = 0
        save_performance = self.parameter.get("save_performance")
        self.dialogue_manager.state_tracker.agent.eval_mode() # for testing
        # if self.parameter.get("classifier_type")=="deep_learning" and self.parameter.get("disease_as_action") == False:
        #     self.dialogue_manager.train_deep_learning_classifier(epochs=20)
        success_count = 0
        absolute_success_count = 0
        total_reward = 0
        total_turns = 0
        #evaluate_session_number = len(self.dialogue_manager.state_tracker.user.goal_set[dataset])
        dataset_len=len(self.dialogue_manager.state_tracker.user.goal_set[dataset])
        evaluate_session_number=self.parameter.get("evaluate_session_number")
        random.seed(12354)
        evaluate_session_index = random.sample(range(dataset_len), evaluate_session_number)
        inform_wrong_disease_count = 0
        num_of_true_slots = 0
        num_of_implicit_slots = 0
        
        real_implicit_slots = 0
        
        num_of_session = 0
        num_of_activate_human = 0
        num_of_inquire_human = 0
        num_of_ban = 0
        # evaluate_session_index += evaluate_session_index
        # evaluate_session_index += evaluate_session_index
        #for goal_index in range(0,evaluate_session_number, 1):
        for goal_index in evaluate_session_index:
            self.dialogue_manager.initialize(dataset=dataset, goal_index=goal_index)
            episode_over = False
            num_of_activate_human_for_group = 0
            while episode_over == False:
                subtask_state = self.dialogue_manager.state_tracker.agent.subtask_terminal
                reward, episode_over, dialogue_status,slots_proportion_list,inquire_state, return_disease = self.dialogue_manager.next(
                    save_record=False,greedy_strategy=False, index=index,mode=mode)
                
                # print("master",self.dialogue_manager.state_tracker.agent.master_action_index)
                # print("turn",self.dialogue_manager.state_tracker.turn)
                if episode_over == False:
                    if self.dialogue_manager.state_tracker.agent.lower_agent_is_human is True:
                        num_of_inquire_human += 1
                    if subtask_state == True and self.dialogue_manager.state_tracker.agent.lower_agent_is_human is True:
                        num_of_activate_human += 1
                        # print("99")
                        
                        num_of_activate_human_for_group +=1
                    self.record_mid_process(self.dialogue_manager.state_tracker.turn, self.dialogue_manager.state_tracker.agent.lower_agent_is_human, subtask_state, inquire_state, return_disease)
                total_reward += reward
            assert len(slots_proportion_list)>0
            
            # print(self.dialogue_manager.state_tracker.user.goal)
            num_of_session += 1
            group_id = self.dialogue_manager.state_tracker.user.goal['group_id']
            # 最新的1000个mr2
            if len(self.cur_mr2_group[group_id]) >1000 - 1:
                self.cur_mr2_group[group_id] = self.cur_mr2_group[group_id][1:]
            if slots_proportion_list[2] == 0 or slots_proportion_list[0] == slots_proportion_list[2]:
                self.cur_mr2_group[group_id].append(1.0)
            else:
                self.cur_mr2_group[group_id].append(float("%.5f" % (float(slots_proportion_list[0]) / float(slots_proportion_list[2]))))
            num_of_true_slots += slots_proportion_list[0]
            num_of_ban  += slots_proportion_list[1]
            real_implicit_slots += slots_proportion_list[2]
            total_turns += (self.dialogue_manager.state_tracker.turn-3)/2
            inform_wrong_disease_count += self.dialogue_manager.inform_wrong_disease_count
            self.record_each_group(self.dialogue_manager.state_tracker.user.goal['disease_id'], dialogue_status, num_of_activate_human_for_group)
            if dialogue_status == dialogue_configuration.DIALOGUE_STATUS_SUCCESS:
                success_count += 1
                if self.dialogue_manager.inform_wrong_disease_count == 0:
                    absolute_success_count += 1
        # print([self.each_group[iter]['true_number_prob'] for iter in range(90) if iter in self.each_group.keys()])
        
        #activator_human_prob
        # print([self.mid_process[iter]['activator_human_prob'] for iter in  self.mid_process.keys()])
        # #turn_disease_prob
        # print([self.mid_process[iter]['turn_disease_true'] for iter in  self.mid_process.keys()])
        # print([self.mid_process[iter]['all_number'] for iter in  self.mid_process.keys()])
        # print([self.mid_process[iter]['activator_human_number'] for iter in  self.mid_process.keys()])
        random.seed(None)
        # assert 0
        success_rate = float("%.3f" % (float(success_count) / num_of_session))
        absolute_success_rate = float("%.3f" % (float(absolute_success_count) / num_of_session))
        average_reward = float("%.3f" % (float(total_reward) / num_of_session))
        average_turn = float("%.3f" % (float(total_turns) / num_of_session))
        average_wrong_disease = float("%.3f" % (float(inform_wrong_disease_count) / num_of_session))
        # mr表示隐式症状询问准确率， mr2表示隐式症状召回率
        match_rate = 0#float("%.3f" % (float(num_of_true_slots) / float(num_of_implicit_slots)))
        match_rate2 = float("%.3f" % (float(num_of_true_slots) / float(real_implicit_slots)))
        average_activate_human = float("%.3f" % (float(num_of_activate_human) / float(num_of_session)))
        average_inquire_human = float("%.3f" % (float(num_of_inquire_human) / float(num_of_session)))
        average_ban = float("%.3f" % (float(num_of_ban) / float(num_of_session)))
        average_repeated_action = float("%.4f" % (float(self.dialogue_manager.repeated_action_count) / num_of_session))

        self.dialogue_manager.state_tracker.agent.train_mode() # for training.
        res = {
            "success_rate":success_rate,
            "average_reward": average_reward,
            "average_turn": average_turn,
            "average_repeated_action": average_repeated_action,
            "average_match_rate2": match_rate2,
            "ab_success_rate":absolute_success_rate,
            "average_match_rate":match_rate,
            "average_activate_human":average_activate_human,
            "average_inquire_human":average_inquire_human,
            "average_ban":average_ban
        }
        self.learning_curve.setdefault(index, dict())
        self.learning_curve[index]["success_rate"]=success_rate
        self.learning_curve[index]["average_reward"]=average_reward
        self.learning_curve[index]["average_turn"] = average_turn
        #self.learning_curve[index]["average_wrong_disease"]=average_wrong_disease
        self.learning_curve[index]["average_match_rate"]=match_rate
        self.learning_curve[index]["average_match_rate2"] = match_rate2
        self.learning_curve[index]["average_activate_human"] = average_activate_human
        self.learning_curve[index]["average_inquire_human"] = average_inquire_human
        self.learning_curve[index]["average_ban"] = average_ban
        self.learning_curve[index]["average_repeated_action"] = average_repeated_action
        if index % 10 ==9:
            print('[INFO]', self.parameter["run_info"])
            
            # data_pro = json.dumps(self.mid_process, indent=4,ensure_ascii=False, sort_keys=False,separators=(',', ':'))
            # print(data_pro)
            # data_pro = json.dumps(self.each_group, indent=4,ensure_ascii=False, sort_keys=False,separators=(',', ':'))
            # print(data_pro)
            # pickle.dump(file=open("_test.p", "wb"), obj=self.mid_process)
            # assert 0
        if index % 100 ==99 and save_performance == True:
            self.__dump_performance__(epoch_index=index)
        print("%3d simulation SR [%s], ave reward %s, ave turns %s, ave match rate %s, ave match rate2 %s, ave inquire human %s, ave activate human %s, ave ban %s" % (index,res['success_rate'],res['average_reward'], res['average_turn'], res["average_match_rate"],res[ "average_match_rate2"],res["average_inquire_human"],res["average_activate_human"],res["average_ban"]))
        self.dialogue_manager.state_tracker.agent.master.goal_amend = res["average_match_rate2"]
        # print(self.dialogue_manager.ban_turn)
        if self.parameter.get("use_all_labels") == True and self.parameter.get("disease_as_action") == False:
            #self.dialogue_manager.train_deep_learning_classifier(epochs=100)

            if self.parameter.get("agent_id").lower() == "agenthrljoint":
                temp_by_group = {}
                for key,value in self.dialogue_manager.acc_by_group.items():
                    temp_by_group[key] = [0.0, 0.0]
                    if value[1] > 0:
                        temp_by_group[key][0] = float("%.3f" % (value[0]/value[1]))
                        temp_by_group[key][1] = float("%.3f" % (value[1]/value[2]))
                if index % 10 == 9:
                    #self.dialogue_manager.train_deep_learning_classifier(epochs=20)
                    print(self.dialogue_manager.acc_by_group)
                    print(temp_by_group)
                self.dialogue_manager.acc_by_group = {x: [0, 0, 0] for x in self.dialogue_manager.state_tracker.agent.master_action_space}

        if self.parameter.get("use_all_labels") == True and self.parameter.get("agent_id").lower() in ["agenthrljoint", "agenthrljoint2"] and self.parameter.get('train_mode') == False:
            pickle.dump(self.dialogue_manager.disease_record, open('./../../data/disease_record.p', 'wb'))
            pickle.dump(self.dialogue_manager.lower_reward_by_group, open('./../../data/lower_reward_by_group.p', 'wb'))
            pickle.dump(self.dialogue_manager.master_index_by_group, open('./../../data/master_index_by_group.p', 'wb'))
            pickle.dump(self.dialogue_manager.symptom_by_group, open('./../../data/symptom_by_group.p', 'wb'))
            print("##################   the disease record is saved   #####################")

        if self.parameter.get("use_all_labels") and self.parameter.get("agent_id").lower()=="agenthrlnew2" and self.parameter.get("disease_as_action"):
            print("the group id match is %f"%(int(self.dialogue_manager.group_id_match) / int(evaluate_session_number)))
            self.dialogue_manager.group_id_match = 0
            if self.parameter.get("train_mode")==False:
                test_by_group = {key:float(x[0])/float(x[1]) for key,x in self.dialogue_manager.test_by_group.items()}
                print(self.dialogue_manager.test_by_group)
                print(test_by_group)
                self.dialogue_manager.test_by_group = {x:[0,0,0] for x in self.dialogue_manager.state_tracker.agent.master_action_space}
        return res
    def record_each_group(self, group_id, dialogue_status,num_of_activate_human):
        self.each_group.setdefault(group_id, dict())
        if "all_number" not in self.each_group[group_id].keys():
            self.each_group[group_id]["all_number"] = 0
            self.each_group[group_id]["true_number"] = 0
            self.each_group[group_id]["true_number_prob"] = 0
            self.each_group[group_id]["activate_human_number"] = 0
        self.each_group[group_id]["all_number"] += 1
        self.each_group[group_id]["activate_human_number"] += num_of_activate_human
        self.each_group[group_id]["ave_activate_human"] = float("%.3f" % (self.each_group[group_id]["activate_human_number"] / self.each_group[group_id]["all_number"]))
        if dialogue_status == dialogue_configuration.DIALOGUE_STATUS_SUCCESS:
            self.each_group[group_id]["true_number"] += 1
        self.each_group[group_id]["true_number_prob"] = float("%.3f" % (self.each_group[group_id]["true_number"] / self.each_group[group_id]["all_number"]))
    
    def record_mid_process(self, turn_num, is_human, subtask_over, inquire_state, return_disease):
        self.mid_process.setdefault(turn_num, dict())
        
        if "all_number" not in self.mid_process[turn_num].keys():
            self.mid_process[turn_num]["all_number"] = 0
            self.mid_process[turn_num]["human_number"] = 0
            self.mid_process[turn_num]["machine_number"] = 0
            self.mid_process[turn_num]["activator_all_number"] = 0
            self.mid_process[turn_num]["activator_human_number"] = 0
            self.mid_process[turn_num]["activator_machine_number"] = 0
            self.mid_process[turn_num]["activator_human_prob"] = 0
            self.mid_process[turn_num]["human_number_true"] = 0
            self.mid_process[turn_num]["machine_number_true"] = 0
            self.mid_process[turn_num]["turn_disease_true"] = 0
            self.mid_process[turn_num]["turn_disease_prob"] = 0
        
        if is_human is False:
            human = 0
            machine = 1
            if turn_num == 3 or subtask_over is True:
                self.rec_human_arr[turn_num-2].append(0)
                self.mid_process[turn_num]["activator_machine_number"] = self.mid_process[turn_num]["activator_machine_number"] + 1
                self.mid_process[turn_num]["activator_all_number"] = self.mid_process[turn_num]["activator_all_number"] + 1

        else:
            human = 1
            machine = 0
            if turn_num == 3 or subtask_over is True:

                self.rec_human_arr[turn_num-2].append(1)
                self.mid_process[turn_num]["activator_human_number"] = self.mid_process[turn_num]["activator_human_number"] + 1
                self.mid_process[turn_num]["activator_all_number"] = self.mid_process[turn_num]["activator_all_number"] + 1   
            
        self.mid_process[turn_num]["all_number"] = self.mid_process[turn_num]["all_number"] + 1
        self.mid_process[turn_num]["human_number"] = self.mid_process[turn_num]["human_number"] + human
        self.mid_process[turn_num]["machine_number"] = self.mid_process[turn_num]["machine_number"] + machine
        if len(self.rec_human_arr[turn_num-2]) >1000 - 1:
            self.rec_human_arr[turn_num-2] = self.rec_human_arr[turn_num-2][1:]
        if machine == 1 :
            if inquire_state == True:
                self.mid_process[turn_num]["machine_number_true"] = self.mid_process[turn_num]["machine_number_true"] + 1
        if human == 1:
            if inquire_state == True:
                self.mid_process[turn_num]["human_number_true"] = self.mid_process[turn_num]["human_number_true"] + 1
        self.mid_process[turn_num]["human_prob"] = float("%.2f" % (self.mid_process[turn_num]["human_number"] / self.mid_process[turn_num]["all_number"]))
        self.mid_process[turn_num]["machine_prob"] = float("%.2f" % (self.mid_process[turn_num]["machine_number"] / self.mid_process[turn_num]["all_number"]))
        if self.mid_process[turn_num]["human_number"] != 0:
            self.mid_process[turn_num]["human_ture_prob"] = float("%.2f" % (self.mid_process[turn_num]["human_number_true"] / self.mid_process[turn_num]["human_number"]))
        if self.mid_process[turn_num]["machine_number"] != 0:
            self.mid_process[turn_num]["machine_ture_prob"] = float("%.2f" % (self.mid_process[turn_num]["machine_number_true"] / self.mid_process[turn_num]["machine_number"]))
        if self.mid_process[turn_num]["activator_all_number"] != 0:
            self.mid_process[turn_num]["activator_human_prob"] = float("%.2f" % (self.mid_process[turn_num]["activator_human_number"] / self.mid_process[turn_num]["activator_all_number"]))
            self.mid_process[turn_num]["activator_machine_prob"] = float("%.2f" % (self.mid_process[turn_num]["activator_machine_number"] / self.mid_process[turn_num]["activator_all_number"]))

        self.mid_process[turn_num]["turn_disease_true"] += return_disease
        self.mid_process[turn_num]["turn_disease_prob"] = float("%.2f" % (self.mid_process[turn_num]["turn_disease_true"] / self.mid_process[turn_num]["all_number"]))
    
    def warm_start(self, epoch_number):
        """
        Warm-starting the dialogue, using the sample from rule-based agent to fill the experience replay pool for DQN.
        :param agent: the agent used to warm start dialogue system.
        :param epoch_number: the number of epoch when warm starting, and the number of dialogue sessions of each epoch
                             equals to the simulation epoch.
        :return: nothing to return.
        """
        for index in range(0,epoch_number,1):
            res = self.simulation_epoch(epoch_size=self.epoch_size, index=index)
            print("%3d simulation SR %s, ABSR %s,ave reward %s, ave turns %s, ave wrong disease %s" % (
            index, res['success_rate'], res["ab_success_rate"], res['average_reward'], res['average_turn'], res["average_wrong_disease"]))
            # if len(self.dialogue_manager.experience_replay_pool)==self.parameter.get("experience_replay_pool_size"):
            #     break

    def __dump_performance__(self, epoch_index):
        """
        Saving the performance of model.

        Args:
            epoch_index: int, indicating the current epoch.
        """
        file_name = self.parameter["run_info"][0:25] + "_" + str(epoch_index) + ".p"
        file_name2 = 'mid_process_' + self.parameter["run_info"][0:25] +"_" + str(epoch_index) + ".p"
        file_name3 = 'each_group_' + self.parameter["run_info"][0:25] +"_" + str(epoch_index) + ".p"
        performance_save_path = self.parameter["performance_save_path"]
        if os.path.isdir(performance_save_path) is False:
            os.mkdir(performance_save_path)
        dirs = os.listdir(performance_save_path)
        for dir in dirs:
            if self.parameter["run_info"][0:25] in dir:
                os.remove(os.path.join(performance_save_path, dir))
        
        pickle.dump(file=open(os.path.join(performance_save_path,file_name), "wb"), obj=self.learning_curve)
        if self.parameter.get("train_mode")==False:
            pickle.dump(file=open(os.path.join(performance_save_path,file_name2), "wb"), obj=self.mid_process)
            pickle.dump(file=open(os.path.join(performance_save_path,file_name3), "wb"), obj=self.each_group)
        #self.dialogue_manager.state_tracker.agent.save_visitation(epoch_index)
