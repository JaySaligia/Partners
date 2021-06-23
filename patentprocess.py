import pandas as pd
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import jieba.posseg as pseg
import numpy as np
import pickle
import jieba.analyse
from tqdm import trange

stopwords_path = 'stopwords.txt'
stopwords = []
for word in open(stopwords_path, 'r', encoding='UTF-8'):
    stopwords.append(word.strip())
w2v_path = './w2vec_new.300d'
w2v = pickle.load(open(w2v_path, 'rb'))


def word_cut_trans(s):
    words = []
    for word, flag in pseg.cut(s):
        if word not in stopwords:
            words.append(word)
    return ' '.join(words)


def cal_doc2vec(model, text):
    text = word_cut_trans(text)
    vector = model.infer_vector(doc_words=text, alpha=0.025, steps=10)
    return np.array(vector)


def cal_textrank(model, text):
    keywords = jieba.analyse.textrank(text, topK=5, withWeight=True, allowPOS=('ns', 'n', 'vn', 'v'))
    a = []
    for item in keywords:
        try:
            a.append(model[item[0]])
        except:
            continue
    if len(a) == 0:
        return np.zeros(300)
    else:
        mean_val = np.mean(np.matrix(a), axis=0)
        return np.array(mean_val).flatten()


doc_model = Doc2Vec.load('doc2vec.model', mmap='r')
# test_text = '1.本外观设计产品的名称：太阳能路灯控制器。2.本外观设计产品的用途：用于城市太阳能照明系统。3.本外观设计的设计要点：主视图、后视图、左视图、右视图、俯视图、仰视图中产品的形状及图案。4.最能表明设计要点的图片或者照片：主视图。'
# print(cal_doc2vec(doc_model, test_text))
# print(cal_textrank(w2v, test_text))
patent_path = r'.\data\patents\query_result1.xlsx'
df = pd.read_excel(patent_path)
group = df.groupby('tmp_patent_id_wuhan.applicant')
store = {}
for i, g in enumerate(list(group)):
    print(i)
    corp_name = g[0]
    store[corp_name] = []
    df_pt = g[1]
    vec1 = []
    vec2 = []
    for j in trange(len(df_pt)):
        row = df_pt.iloc[j]
        abstract = str(row['corp_patent.summary'])
        vec1.append(cal_doc2vec(doc_model, abstract))
        vec2.append(cal_textrank(w2v, abstract))
    store[corp_name].append(np.matrix(vec1))
    store[corp_name].append(np.matrix(vec2))
np.save('patent_vec', store)




