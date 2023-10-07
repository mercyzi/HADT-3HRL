import pickle
import json

label_all_path = ''
set1= pickle.load(open('disease_symptom.p', 'rb'))
with open("disease_symptom.json","w") as f:
    json.dump(set1,f)
    print("加载入文件完成...")
