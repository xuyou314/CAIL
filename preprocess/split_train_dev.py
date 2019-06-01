import json
import random
if __name__=="__main__":
    f=open("/home/xuyou/CAIL/small_train_data.json","r")
    f_train=open("/home/xuyou/CAIL/train.json","w")
    f_dev=open("/home/xuyou/CAIL/dev.json","w")
    all_data=json.load(f)
    random.shuffle(all_data['data'])
    train_dict={}
    dev_dict={}
    train_dict['data']=all_data['data'][:1900]
    dev_dict['data']=all_data['data'][1900:]
    json.dump(train_dict,f_train,ensure_ascii=False)
    json.dump(dev_dict,f_dev,ensure_ascii=False)