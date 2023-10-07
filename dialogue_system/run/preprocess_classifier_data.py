import numpy as np
import pickle
import json
import copy
import os
import random
import sys, os

sys.path.append(os.getcwd().replace("src/dialogue_system/run",""))
def enrich_data(train_set):
    re_train_set = copy.deepcopy(train_set)
    for item in re_train_set:
        # 以0.6概率选取隐式症状，选2次
        for i in range(1):
            imp = {}
            for key,v in item['goal']['implicit_inform_slots'].items():
                greedy = random.random()
                if greedy < 0.8:
                    # if v is True:
                    #     imp[key] = v                
                    imp[key] = v 
            item['goal']['implicit_inform_slots'] = imp
            train_set.append(item)

    return train_set
