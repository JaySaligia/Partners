# encoding=utf-8
# 使用语义桩进行匹配
# 将企业关键词变成对应的语义桩形式
import os
import pickle
import re

import numpy as np
from collections import Counter
import jieba.posseg as pseg
from sklearn.cluster import KMeans
from sklearn import svm
from gensim import models

import mytools
from sklearn.metrics import accuracy_score, recall_score, classification_report
import pandas as pd
# n为待匹配产业词数，v为词向量维度，m为语义桩数，t为产业数
from tqdm import trange


class Cluster:
    def __init__(self, k, method=0):
        # 设置k值
        self.k = k
        # method表示分类方法,0 k-means, 1 svm
        self.method = method
        # 读取所有的储存文件
        self.store_path = 'verbs'
        self._in = []
        npy_list = os.listdir(self.store_path)
        for npy_file in npy_list:
            read_d = np.load(os.path.join(self.store_path, npy_file), allow_pickle=True).item()
            for item in read_d:
                cluster_center = read_d[item]
                if not len(cluster_center) == 0:
                    self._in.extend(read_d[item])

        print('共有{}个聚类中心'.format(len(self._in)))
        # 开始聚类
        # 使用k-means
        if method == 0:
            cluster_kmeans = KMeans(n_clusters=self.k)
            cluster_kmeans.fit(self._in)
            np.save('stakes_{}'.format(self.k), cluster_kmeans.cluster_centers_)
        print('语义桩构建完成,共有{}个桩'.format(self.k))


class Word2Stake:
    def __init__(self, stakes_num, ind_num):
        # mytools
        tools = mytools.Mytools()
        # 读取语义桩向量,以列向量的形式储存
        stake_path = 'stakes_{}.npy'.format(stakes_num)
        self.stakes = np.mat(np.array(np.load(stake_path))).T
        self.m = np.size(self.stakes, 1)
        stakes_l2 = np.array([1 / np.linalg.norm(self.stakes[:, i]) for i in range(self.m)]).reshape((1, self.m))
        self.cluster_stakes = []
        self.ind_raw_path = '{}_raw.txt'.format(ind_num)
        self.ind_new_path = '{}_new.txt'.format(ind_num)
        self.w2v_path = './w2vec_new.300d'
        self.w2v = pickle.load(open(self.w2v_path, 'rb'))
        # print(self.w2v['你好'])
        # 返回形式
        self.store_d = {}
        # 存储名称
        with open(self.ind_raw_path, encoding='utf-8') as f:
            self.store_d['name'] = f.readlines()
        self.num = len(self.store_d['name'])
        ind_vec = []
        cosine_similarity = []
        ind_stakes = []
        # 存储桩构成向量,列向量为每个桩的权重（m*t）
        with open(self.ind_new_path, encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                line = line.replace('\n', '').strip()
                if line.isdigit():
                    line_num = int(line)
                    # 处理上一个产业
                    if line_num > 0:
                        cosine_similarity = tools.cosineMats(a=np.mat(ind_vec), b=self.stakes, b_l2=stakes_l2)
                        max_index = self.find_max_index(cosine_similarity, 1)
                        ind_stakes.append(max_index)
                        ind_vec = []
                else:
                    words = line.split(' ')
                    for word in words:
                        try:
                            vec = self.w2v[word]
                            ind_vec.append(vec)
                        except:
                            continue
            # 处理最后一个类
            cosine_similarity = tools.cosineMats(a=np.mat(ind_vec), b=self.stakes, b_l2=stakes_l2)
            max_index = self.find_max_index(cosine_similarity, 1)
            ind_stakes.append(max_index)

            ind_stakes = np.mat(ind_stakes).T
            self.store_d['class'] = ind_stakes
            # 储存为矩阵形式（m*t）
            np.save('class_{}_{}'.format(stakes_num, self.num), self.store_d)
            print('分类桩构建完成:class_{}_{}.npy,大小为（{},{}）'.format(stakes_num, self.num, np.size(ind_stakes, 0),
                                                              np.size(ind_stakes, 1)))

    # 得到矩阵中每行/列最大的值对应的列号，并返回出现次数组成列向量
    def find_max_index(self, a, axis):
        maximalist = np.argmax(a, axis=axis)
        maximalist_len = len(a)
        count_list = []
        for i in range(maximalist_len):
            count_list.append(maximalist[i, 0])
        appearance = Counter(count_list)
        ret = np.array([0 if i not in appearance else appearance[i] / maximalist_len for i in range(self.m)])
        return ret


# 分类
class IndClassifyStake:
    # 指定分类的组数
    def __init__(self, stakes_num, class_num, data_dir, test=False):
        # 配置文件的路径
        self.ind_data_adr = data_dir
        # mytools
        self.tools = mytools.Mytools()
        # 读取分类信息
        store_d = np.load(os.path.join(self.ind_data_adr, 'class_{}_{}.npy'.format(stakes_num, class_num)),
                          allow_pickle=True).item()
        self.ind_names = store_d['name']
        self.ind_mat = store_d['class']
        t = np.size(self.ind_mat, 1)
        m = np.size(self.ind_mat, 0)
        print('加载语义桩分类矩阵完成，大小为({},{})'.format(m, t))
        # 读取语义桩向量,以列向量的形式储存
        stake_path = os.path.join(self.ind_data_adr, 'stakes_{}.npy'.format(stakes_num))
        self.stakes = np.mat(np.array(np.load(stake_path))).T
        v = np.size(self.stakes, 0)
        self.stakes_l2 = np.array([1 / np.linalg.norm(self.stakes[:, i]) for i in range(m)]).reshape((1, m))
        print('加载语义桩完成，大小为({},{})'.format(v, m))
        # 读取词向量
        self.w2v_path = os.path.join(self.ind_data_adr, 'w2vec_new.300d')
        self.w2v = pickle.load(open(self.w2v_path, 'rb'))
        print('加载词向量完成')
        # 停用词
        stopwords_path = os.path.join(self.ind_data_adr, 'stopwords.txt')
        self.stopwords = []
        for word in open(stopwords_path, 'r', encoding='UTF-8'):
            self.stopwords.append(word.strip())
        self.word_pattern = r'[:,.：，。、]'
        self.stop_pattern = r'[\[|<|〈|【|（|(].*?[)|）|】|〉|>|\]]'
        self.test = test
        if self.test:
            print('开启测试模式')

    # 分词
    def word_cut(self, s):
        words = []
        for word, flag in pseg.cut(s):
            if word not in self.stopwords:
                words.append(word)
        return words

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

    # 预处理经营范围和分词
    def preprocess_scope(self, business_scope):
        business_scope = re.sub(self.stop_pattern, ' ', str(business_scope))
        business_scope = str(self.proscope(business_scope))
        scopes = re.split(r'[；;]', str(business_scope))
        scopes_split = []
        for scope in scopes:
            scopes_split.append('||'.join(self.word_cut(scope)))
        return scopes_split

    # 预处理tags
    def preprocess_tags(self, tags):
        tags_split = []
        for tag in tags:
            tags_split.append('||'.join(self.word_cut(tag)))
        return tags_split

    # 直接对经营范围进行匹配
    def single_match_scope(self, scope, tags):
        scopes_split = self.preprocess_scope(scope)
        result = []
        for scope_split in scopes_split:
            result.extend(self.single_match_split(scope_split))
        if not tags == '-1000':
            tags_split = self.preprocess_tags(tags)
            for tag_split in tags_split:
                result.extend(self.single_match_split(tag_split))
        appearence = Counter(result)
        ind_top = []
        for k in sorted(appearence, key=appearence.__getitem__, reverse=True):
            ind_top.append(k)
        # 测试模式返回序号
        if self.test:
            return ind_top[0:3]
        # 非测试模式
        ret = []
        for res in range(min(3, len(ind_top))):
            ret.append(self.ind_names[ind_top[res]].replace('\n', ''))
        len_ret = len(ret)
        for i in range(3 - len_ret):
            ret.append('未知行业')
        return ret

    # 对一条经营范围进行匹配, 输入为（'||'）形式
    def single_match_split(self, scope_split):
        words = scope_split.split('||')
        corp_mat = []
        for word in words:
            try:
                corp_mat.append(self.w2v[word])
            except:
                continue
        if len(corp_mat) == 0:
            return []
        corp_mat = np.mat(corp_mat)
        # 得到余弦相似度矩阵
        cosine_mat = self.tools.cosineMats(a=corp_mat, b=self.stakes, b_l2=self.stakes_l2)
        result_mat = np.matmul(cosine_mat, self.ind_mat)
        result_list = np.argmax(result_mat, axis=1)
        count_list = []
        for res in range(len(result_list)):
            count_list.append(result_list[res, 0])
        appearence = Counter(count_list)
        ind_top = []
        for k in sorted(appearence, key=appearence.__getitem__, reverse=True):
            ind_top.append(k)
        ret = []
        for res in range(min(3, len(ind_top))):
            ret.append(ind_top[res])
        return ret


# cluster_num = 244
# C = Cluster(cluster_num)
# for i in range(7):
#     cluster_num = int((i+1) * 50)
# #     C = Cluster(cluster_num)
#     W = Word2Stake(cluster_num, 41)
# I = IndClassifyStake(cluster_num, 45, './', test=True)
# W = Word2Stake(200, 212)
# print(I.single_match_scope('汽车发动机', '汽车销售||摩托车改装||汽车配件'))
# print(I.single_match_scope('制鞋、制衣、鞋用材料生产、加工。', '-1000'))
# print(I.single_match_scope('皮革制品制造,皮箱、包（袋）制造,生产箱包皮具、鞋帽、服装。[涉及审批许可项目的，只允许在审批许可的范围和有效期限内从事经营活动]'))
# print(I.single_match_scope('生产以PVC、PU、PP、PE塑料和橡胶为原料的塑料、橡胶制品及塑料机械、雨具、模具和零配件（以上经营范围凡涉及国家专项专营规定的从其规定；涉及审批许可项目的，只允许在审批许可的范围和有效期限内从事生产经营）。'))
# print(I.single_match_scope('生产运动鞋、鞋材、运动服装（不含出口配额许可证管理品种）'))
# print(I.single_match_scope('鞋塑制品（鞋底）制造'))
# print(I.single_match_scope('运动鞋、服装、运动器材（不含弩等需经前置许可的项目）、包装袋、纸盒、纸箱、箱包、服装辅料（吊牌）、塑料制品（塑料薄膜）'))
# I.single_match_scope('日用百货、土产日杂、五金交电、针纺织品销售')
# I.single_match_scope('矿山机械、工程设备设计、制造；钢结构厂房设计、安装；矿山设备、机电配件销售')
# I.single_match_scope('食用灵芝、蘑菇培植、蔬菜种植及相关技术推广、农副产品购销')
# I.single_match_scope('小麦、玉米、红薯及烟草种植、销售')
# I.single_match_scope('摩托车配件、冲压件、机械模具、机动车零部件')
# I.single_match_scope('电线电缆、接插件销售，安防监控器材、防盗报警设备、楼宇对讲设备销售及安装')
# I.single_match_scope('发放小额贷款')
# I.single_match_scope('农业机械销售及售后服务，配件销售')
# I.single_match_scope('信息科技领域内的技术开发、技术转让、技术咨询')
# I.single_match_scope('农副产品、有色金属、矿产品、五金交电、汽车及汽车配件销售')
# I.single_match_scope('母婴用品、服饰、鞋包及玩具产品批发、零售')
# I.single_match_scope('网络动漫制作；广告设计，企业形象设计，会展策划；网站建设')
# I.single_match_scope('各种装潢材料、监控器材销售；室内外装饰设计；水电安装；园林绿化；汽车装潢')
