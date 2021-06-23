# train patent
from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import pandas as pd
import jieba.posseg as pseg
from tqdm import trange
from tqdm import tqdm
import numpy as np

file = r'.\data\patents\query_result1.xlsx'
stopwords_path = 'stopwords.txt'
stopwords = []
for word in open(stopwords_path, 'r', encoding='UTF-8'):
    stopwords.append(word.strip())

# def word_cut(s):
#     words = []
#     for word, flag in pseg.cut(s):
#         if word not in stopwords:
#             words.append(word)
#     return ' '.join(words)
#
#
# df = pd.read_excel(file)
# train_data = list(df['corp_patent.summary'].astype(str))
# train_split = []
# for i in trange(len(train_data)):
#     train_split.append(word_cut(train_data[i]))
# np.save('patent_train', train_split)
train_split = np.load('patent_train.npy')
train_doc = []
for i, text in enumerate(tqdm(train_split)):
    words = text.split(' ')
    l = len(words)
    words[l - 1] = words[l - 1].strip()
    doc = TaggedDocument(words, tags=[i])
    train_doc.append(doc)
model = Doc2Vec(train_doc, window=5, vector_size=300, sample=1e-3, workers=5, negative=5)
model.train(train_doc, total_examples=model.corpus_count, epochs=1)
model.save('doc2vec.model')
