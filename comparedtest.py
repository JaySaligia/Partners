import re
import numpy as np
import os
import jieba.posseg as pseg
import pandas as pd
from tqdm import trange
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score


class ComaperaMethod:
    def __init__(self, ind_num):
        stopwords_path = 'stopwords.txt'
        self.stopwords = []
        for word in open(stopwords_path, 'r', encoding='UTF-8'):
            self.stopwords.append(word.strip())
        self.stop_pattern = r'[\[|<|〈|【|（|(].*?[)|）|】|〉|>|\]]'
        self.ind_raw_path = '{}_raw.txt'.format(ind_num)
        self.ind_new_path = '{}_new.txt'.format(ind_num)
        self.ind_words = []
        with open(self.ind_new_path, encoding='utf-8') as f:
            lines = f.readlines()
            words = []
            for line in lines:
                line = line.replace('\n', '').strip()
                if line.isdigit():
                    line_num = int(line)
                    if line_num > 0:
                        self.ind_words.append(words)
                        words = []
                else:
                    words.extend(line.split(' '))
            self.ind_words.append(words)
        print('init compared well')

    # 去除无效词语
    def proscope(self, s):
        pattern = r'[（|(].*?[)|）]'
        #  取出括号内容
        s = re.sub(pattern, '', s)
        s = s.replace('企业依法自主选择经营项目', '')
        s = s.replace('开展经营活动', '')
        s = s.replace('依法须经批准的项目', '')
        s = s.replace('经相关部门批准后依批准的内容开展经营活动', '')
        s = s.replace('，经相关部门批准后方可', ' ')
        s = s.replace('不得从事本市产业政策禁止和限制类项目的经营活动', '')
        s = s.replace('【', '').replace('】', '').replace('〓', '')
        s = s.replace('...详细', '')
        s = s.replace('详情', '')
        return s

    def word_cut(self, s):
        words = []
        for word, flag in pseg.cut(s):
            if word not in self.stopwords:
                words.append(word)
        return words

    # 预处理经营范围和分词
    def preprocess_scope(self, business_scope):
        business_scope = re.sub(self.stop_pattern, ' ', str(business_scope))
        business_scope = str(self.proscope(business_scope))
        return business_scope

    def single_match_scope(self, scope):
        scope = self.preprocess_scope(scope)
        corp_words = self.word_cut(scope)
        scores = []
        denominator = len(corp_words)
        for i in range(len(self.ind_words)):
            numerator = 0
            for word in corp_words:
                if word in self.ind_words[i]:
                    numerator += 1
            scores.append((numerator / denominator, i))
        scores.sort()
        return list(map(lambda x: x[1], scores))[0:3]


C = ComaperaMethod(41)
C.single_match_scope('农副产品、有色金属、矿产品、五金交电、汽车及汽车配件销售')
test = pd.read_csv('./test/test.csv')
groundTruth = []
predict = []
for i in trange(len(test)):
    y = int(test.iloc[i][0])
    groundTruth.append(y)
    y_s = C.single_match_scope(test.iloc[i][1])
    y_ = y_s[0]
    for res in y_s:
        if res == y:
            y_ = y
    predict.append(y_)
print('acc:{}\n'.format(accuracy_score(groundTruth, predict)))
print('prec:{}\n'.format(precision_score(groundTruth, predict, average='macro')))
print('rec:{}\n'.format(recall_score(groundTruth, predict, average='macro')))
print('f1:{}\n'.format(f1_score(groundTruth, predict, average='macro')))
