import pickle
import json

label_all_path = ''
# set1= pickle.load(open('disease_record.p', 'rb'))
# set1= pickle.load(open('lower_reward_by_group.p', 'rb'))
# set1= pickle.load(open('master_index_by_group.p', 'rb'))
set1= pickle.load(open('symptom_by_group.p', 'rb'))
with open("symptom_by_group.json","w") as f:
    json.dump(set1,f)
    print("加载入文件完成...")