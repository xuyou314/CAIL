import jieba
import json
import re
import spacy
nlp = spacy.blank("zh")
import numpy as np
def word_tokenize(sent):
    doc = nlp(sent)
    return [token.text for token in doc]
if __name__=="__main__":
    f=open("small_train_data.json")
    data=json.load(f)
    sentce=data['data'][0]['paragraphs'][0]['context']
    # jieba.add_word('苏A', 1500)
    # jieba.add_word('经审理')
    # jieba.add_word('×',1)
    # jieba.add_word('××',1)
    # jieba.add_word('×××',100)
    test_str='新泾三村XXX号XXX室'
    test2_str='The dataset used for this task is'
    jieba.re_han_default = re.compile("([\u4E00-\u9FD5a-zA-Z0-9+#&\._%\xd7]+)", re.U)
    fout=open("/home/xuyou/pingan/data/wiki.zh.simple.seg.txt","a+")
    print(' '.join(jieba.cut(sentce)))
    for i in range(len(data['data'])):
        sentce = data['data'][i]['paragraphs'][0]['context']
        add_line=' '.join(jieba.cut(sentce))
        fout.write(add_line+'\n')
    fout.close()
    # arr=np.array([1])
    # arr.tostring()
#print(' '.join(jieba.cut(test_str)))
#print(' '.join(jieba.cut(test2_str)))