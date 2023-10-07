# -*- coding: utf-8 -*-
"""
Rule-based agent.
"""

import copy
import random
import sys, os
sys.path.append(os.getcwd().replace("src/dialogue_system/agent",""))

from src.dialogue_system.agent import Agent
from src.dialogue_system import dialogue_configuration


class AgentRule(Agent):
    """
    Rule-based agent.
    """
    def __init__(self,action_set, slot_set, disease_symptom, parameter, disease_as_action=True):
        super(AgentRule,self).__init__(action_set=action_set,slot_set=slot_set,disease_symptom=disease_symptom,
                                        parameter=parameter,disease_as_action=disease_as_action)
        self.action_space = self._build_action_space(disease_symptom, disease_as_action=False)
        self.id2symptom = {}
        for iter in range(len(self.action_space)):
            temp_action = copy.deepcopy(self.action_space[iter])
            #print('temp_action',temp_action)
            self.id2symptom[iter] = list(temp_action["request_slots"].keys())[0]

    def next(self, state, turn, greedy_strategy, **kwargs):
        #self.agent_action["turn"] = turn
        disease_tag = kwargs.get("disease_tag")
        tag = disease_tag['disease_tag']
        impl_set = disease_tag['goal']['implicit_inform_slots']
        
        
        candidate_symptoms = self._get_candidate_symptoms(state=state,disease_tag=tag,impl_set = impl_set)
        #print(candidate_symptoms)
        # ban_human = False
        if len(candidate_symptoms) == 0:
            return -1, -1, True
        self.agent_action["request_slots"].clear()
        self.agent_action["explicit_inform_slots"].clear()
        self.agent_action["implicit_inform_slots"].clear()
        self.agent_action["inform_slots"].clear()
        self.agent_action["turn"] = turn
        current_slots = copy.deepcopy(state["current_slots"]["inform_slots"])
        greedy = random.random()
        
        if greedy > self.parameter.get("doctor_acc"):
            for iter in range(len(self.action_space)):
                    symptom_index = random.randint(0, len(self.action_space) - 1)
                    symptom = self.id2symptom[symptom_index]
                    # print(iter)
                    # print(action_index)
                    # print(action)
                    if symptom not in current_slots.keys():
                        break
            self.agent_action["action"] = "request"
            self.agent_action["request_slots"].clear()
            self.agent_action["request_slots"][symptom] = dialogue_configuration.VALUE_UNKNOWN
        else:
            symptom = random.choice(candidate_symptoms)
            self.agent_action["action"] = "request"
            self.agent_action["request_slots"].clear()
            self.agent_action["request_slots"][symptom] = dialogue_configuration.VALUE_UNKNOWN

        agent_action = copy.deepcopy(self.agent_action)
        agent_action.pop("turn")
        agent_action.pop("speaker")
        agent_index = self.action_space.index(agent_action)
        
        return self.agent_action, agent_index, False

    def _get_candidate_symptoms(self, state,disease_tag,impl_set):
        """
        Comparing state["current_slots"] with disease_symptom to identify which disease the user may have.
        :param state: a dict, the current dialogue state gotten from dialogue state tracker..
        :return: a list of candidate symptoms.
        """
        inform_slots = state["current_slots"]["inform_slots"]
        inform_slots.update(state["current_slots"]["explicit_inform_slots"])
        inform_slots.update(state["current_slots"]["implicit_inform_slots"])
        #wrong_diseases = state["current_slots"]["wrong_diseases"]
        candidate_symptoms = []
        # two cases
        # may activate wrong doctors(8 kind),may activate fit doctor

        # the doctor can't diagnose the disease
        # acquire remaining symptom of the disease_tag(ture disease) 
        # if disease_tag not in self.disease_symptom.keys():
        #     #print('disease_tag',disease_tag)
        #     return candidate_symptoms
        # else:
            # acquire remaining symptom of the disease_tag(ture disease) 
        for symptom in impl_set.keys():
            if symptom not in inform_slots.keys():
                candidate_symptoms.append(symptom)
                #print(symptom)
        return candidate_symptoms

    def train_mode(self):
        pass

    def eval_mode(self):
        pass