# -*- coding:utf-8 -*-

import os
import re
import pandas as pd
import numpy as np

from tqdm import tqdm
from tqdm import trange
from joblib import Parallel, delayed

from tag_process.tagprocess import Classify
from pat_process.patprocess import Similar
from coo_process.tagcooccur import Cooccur

import json


# import matplotlib.pyplot as plt
# plt.rcParams['font.sans-serif'] = ['SimHei']


class CorpProcess():
    def __init__(self, data_dir, corp_cuts, patent_cuts, adds_cuts):
        # 企业文件目录
        self.corp_data_adr = data_dir
        # 企业分隔路径
        self.corp_cuts_adr = os.path.join(self.corp_data_adr, corp_cuts)
        # 专利分隔路径
        self.patent_cuts_adr = os.path.join(self.corp_data_adr, patent_cuts)
        # 存储部分目录
        self.part_adds_adr = os.path.join(self.corp_data_adr, adds_cuts)
        # 相关需要文件目录
        self.corp_infor_adr = os.path.join(self.corp_data_adr, 'corp_inforn.csv')
        self.corp_patent_adr = os.path.join(self.corp_data_adr, 'corp_patent.csv')

        # 生成文件的目录
        self.corp_added_adr = os.path.join(self.corp_data_adr, 'corp_added.csv')
        # 竞争关系的目录
        self.corp_similar_adr = os.path.join(self.corp_data_adr, 'corp_similar.csv')
        # 合作关系的目录
        self.corp_partner_adr = os.path.join(self.corp_data_adr, 'corp_partner.csv')
        # 初始化分类功能
        # 企业标签分类，返回K个行业标签
        self.T = Classify('./tag_process/')

    def proscope(self, s):
        #  整理经营范围
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

    def judge(self, name):

        if '武汉东湖' in name or '武汉市东湖' in name or '东湖开发区' in name:
            new = '武汉东湖新技术开发区'
        elif '武汉临空港' in name or '武汉市临空港' in name or '临空港开发区' in name:
            new = '武汉临空港经济技术开发区'
        elif '武汉经济' in name or '武汉市经济' in name:
            new = '武汉经济技术开发区'
        else:
            new = '武汉非经济开发区'

        return new

    def zoom(self, corp_address, corp_authority):
        # 定位武汉市
        # if corp_city =='武汉市':
        if self.judge(corp_address) != '武汉非经济开发区' or self.judge(corp_authority) != '武汉非经济开发区':
            # 根据企业地址和工商局信息
            if self.judge(corp_address) != '武汉非经济开发区':
                # 两者取其一
                new = self.judge(corp_address)
            else:
                new = self.judge(corp_authority)
        else:
            new = '武汉非经济开发区'
        # 非武汉市地区
        # else:
        #	new='-1000'

        return new

    def indus(self, corp_terms, corp_tags, top_k):
        # 计算出企业分类
        if corp_tags == '-1000':
            # 当标签不存在时
            ids = self.T.make_match(self.T.proprecess(corp_terms))
            # 判断企业分类的个数
            nids = ids[:top_k] if len(ids) > top_k else ids + ['未知行业'] * (top_k - len(ids))
            # 三种产业字符串拼接
            new_ids = '/'.join(nids)
        else:
            ids = self.T.make_match(corp_tags) if len(corp_tags.split('||')) >= 5 else self.T.make_match(
                '||'.join([self.T.proprecess(corp_terms), corp_tags]))
            # 判断返回标签的情况
            if ids[0] == '未知行业':
                ids = self.T.make_match(self.T.proprecess(corp_terms))
                # 判断企业分类的个数
                nids = ids[:top_k] if len(ids) > top_k else ids + ['未知行业'] * (top_k - len(ids))
                # 三种产业字符串拼接
                new_ids = '/'.join(nids)
            else:
                nids = ids[:top_k] if len(ids) > top_k else ids + ['未知行业'] * (top_k - len(ids))
                # 三种产业字符串拼接
                new_ids = '/'.join(nids)

        return new_ids

    def apply_parallel(self, df_grouped, func):
        """利用 Parallel 和 delayed 函数实现并行运算"""
        results = Parallel(n_jobs=5)(delayed(func)(group) for name, group in df_grouped)
        # 拼接数据
        return pd.concat(results)

    def zoom_func(self, subset):
        # 区域分类
        subset['corp_infor.zoom'] = subset.apply(
            lambda x: self.zoom(x['corp_infor.address'], x['corp_infor.authority']), axis=1)

        return subset

    def indus_func(self, subset):
        # 产业分类
        subset['corp_infor.indus'] = subset.apply(
            lambda x: self.indus(x['corp_infor.business_scope'], x['corp_infor.tags'], 3), axis=1)

        return subset

    def count_addinfro(self):

        # 全国非武汉市企业有标签且注册资金在200万以上，以及全武汉市的企业（可以没有标签）
        # 列出目录下对应excel文件
        excel_list = os.listdir(self.corp_cuts_adr)
        # # 文件按顺序输出
        name_list = sorted(excel_list, key=lambda x: int(x.replace("query_result", "").split('.')[0]))
        # # 文件路径遍历
        excel_dirs = [os.path.join(self.corp_cuts_adr, i) for i in name_list]
        # # 第1列数据为索引
        pds_list = [pd.read_excel(i, header=0, index_col=0, encoding="utf-8") for i in excel_dirs]
        # # add文件横向合并
        corp_infor = pd.concat(pds_list, axis=0)
        # # 替换\t字符，为空字符串
        corp_infor = corp_infor.replace('\t+', '', regex=True)
        # # 保存csv格式,以\t分隔
        corp_infor.to_csv(self.corp_infor_adr, encoding="utf-8", sep='\t')

        corp_infor = pd.read_csv(self.corp_infor_adr, header=0, encoding="utf-8", sep='\t')
        corp_num = len(corp_infor)
        print('--------总计%d家企业-------' % corp_num)
        # 进行分块处理
        split_corp = corp_num // 100000 + 1

        # corp_infor=corp_infor.drop(['corp_infor.zoom'], axis=1)
        # print(corp_infor.columns)
        # print(corp_infor)
        # 统计武汉的企业个数
        # whu_infro = corp_infor[corp_infor['corp_infor.city'] == '武汉市']
        # print('----------湖北省武汉市在册的企业总计为%d个--------'%len(whu_infro))

        # 将数据库中为空数据替换
        corp_infor.fillna(value='-1000', inplace=True)

        print('----------企业数据分为%d部分开始标注处理------' % split_corp)

        # 把有用的数据取出来
        corp_use = corp_infor[
            ['corp_infor.corp_id', 'corp_infor.address', 'corp_infor.authority', 'corp_infor.business_scope',
             'corp_infor.tags']]

        for i in range(split_corp):
            print('-----------第%d部分数据开始标注--------' % int(i + 1))

            corp_part = corp_use[int(len(corp_infor) * i / split_corp):int(len(corp_infor) * (i + 1) / split_corp)]

            ##################################添加企业的区位分类信息##########################################
            df_grouped = corp_part.groupby(corp_part.index)
            # 添加区位
            corp_part = self.apply_parallel(df_grouped, self.zoom_func)

            print('----区位标注完毕！！--')
            ##################################添加企业的区位分类信息##########################################

            ##################################添加企业的产业分类信息##########################################
            df_grouped = corp_part.groupby(corp_part.index)

            ## 添加产业
            corp_part = self.apply_parallel(df_grouped, self.indus_func)
            ##################################添加企业的产业分类信息##########################################

            corp_part['corp_infor.indus1'] = corp_part['corp_infor.indus'].apply(lambda x: x.split('/')[0])
            corp_part['corp_infor.indus2'] = corp_part['corp_infor.indus'].apply(lambda x: x.split('/')[1])
            corp_part['corp_infor.indus3'] = corp_part['corp_infor.indus'].apply(lambda x: x.split('/')[2])

            print('----产业标注完毕！！--')

            # 删除的原始列信息
            drop_col = ['corp_infor.address', 'corp_infor.authority', 'corp_infor.business_scope', 'corp_infor.tags']

            corp_add = corp_part.drop(drop_col, axis=1)

            corp_add.to_csv(os.path.join(self.part_adds_adr, 'query_result%d.csv' % (i + 1)), encoding='utf-8',
                            sep='\t')

            print('-----------第%d部分数据梳理完毕，总计处理了%d条数据--------' % (i + 1, int(len(corp_infor) * (i + 1) / split_corp)))

        # 读取企业存储数据
        corp_infor = pd.read_csv(self.corp_infor_adr, encoding="utf-8", sep='\t')
        # 将库中为空数据替换
        corp_infor.fillna(value='-1000', inplace=True)

        # 列出目录下对应csv文件
        csv_list = os.listdir(self.part_adds_adr)
        # 文件按顺序输出
        csv_list = sorted(csv_list, key=lambda x: int(x.replace("query_result", "").split('.')[0]))

        csv_dirs = [os.path.join(self.part_adds_adr, i) for i in csv_list]
        # 第1列数据为索引
        add_pds = [pd.read_csv(i, header=0, index_col=0, encoding='utf-8', sep='\t') for i in csv_dirs]
        # add文件横向合并
        adds_part = pd.concat(add_pds, axis=0, ignore_index=False)

        # 拼接数据
        corp_out = pd.merge(corp_infor, adds_part, how='inner', on='corp_infor.corp_id')
        corp_out = corp_out.reset_index(drop=True)

        # 整理数据
        corp_out['corp_infor.business_scope'] = corp_out['corp_infor.business_scope'].apply(lambda x: self.proscope(x))
        print(corp_out)

        # 保存数据
        corp_out.to_csv(self.corp_added_adr, encoding='utf-8', sep='\t')

        print('###########企业的经济开发区位和产业分类的属性已经分析完毕！!###########')

    def plot_wuhzoom(self):
        # 对新构建的企业信息表进行统计分析
        corp_data = pd.read_csv(self.corp_added_adr, header=0, index_col=0, encoding="utf-8", sep='\t')
        print('################当前数据总计有%d个企业###################################' % len(corp_data))
        # 对武汉开发区信息进行统计
        print('################开始对武汉市企业的开发区位进行统计分析！#################')
        wuh_data = corp_data[corp_data['corp_infor.city'] == '武汉市']['corp_infor.zoom'].tolist()
        print('################武汉市总计有%d个企业###################################' % len(wuh_data))
        donghu = [i for i in wuh_data if i == '武汉东湖新技术开发区']
        print('武汉东湖新技术开发区的企业总数:', len(donghu))
        lkgang = [i for i in wuh_data if i == '武汉临空港经济技术开发区']
        print('武汉临空港经济技术开发区的企业总数:', len(lkgang))
        jjkai = [i for i in wuh_data if i == '武汉经济技术开发区']
        print('武汉经济技术开发区的企业总数:', len(jjkai))
        qita = len(wuh_data) - len(donghu) - len(lkgang) - len(jjkai)
        print('武汉非经济开发区的企业总数：', qita)

        labels = '武汉东湖开发区', '武汉临空港开发区', '武汉经济开发区', '武汉非开发区'
        fracs = [len(donghu), len(lkgang), len(jjkai), qita]
        explode = tuple([0.1] * 4)

    # plt.axes(aspect=1)
    # plt.pie(x=fracs, labels=labels, autopct='%.2f%%', explode=explode, shadow=True)
    # plt.show()

    def filter_char(self, line):
        # 过滤字符串中的英文与符号，保留汉字
        line = re.sub('[0-9|a-z|A-Z|\-|#|/|%|_|,|\'|:|=|>|<|\"|;|\-|\\|(|)|）|（|\?|\.|\*|\+|\[|\]|\^|\{|\}|\n)]+', '',
                      line)

        return line

    def count_similar(self):
        # 专利计算相似度
        S = Similar('./pat_process/')
        # 打开添加标签的新企业数据
        corp_infor = pd.read_csv(self.corp_added_adr, header=0, index_col=0, encoding="utf-8", sep='\t')

        # # 列出目录下对应excel文件
        excel_list = os.listdir(self.patent_cuts_adr)
        # # 文件按顺序输出
        name_list = sorted(excel_list, key=lambda x: int(x.replace("query_result", "").split('.')[0]))
        # # 路径遍历
        excel_dirs = [os.path.join(self.patent_cuts_adr, i) for i in name_list]
        # # 第1列数据为索引
        pds_list = [pd.read_excel(i, header=0, index_col=0, encoding="utf-8") for i in excel_dirs]
        # # add文件横向合并
        patent_infro = pd.concat(pds_list, axis=0)
        # # 替换\t字符，为空字符串
        patent_infro = patent_infro.replace('\t+', '', regex=True)
        # # 保存csv格式,以\t分隔
        patent_infro.to_csv(self.corp_patent_adr, encoding="utf-8", sep='\t')

        # 读取企业专利数据
        patent_infro = pd.read_csv(self.corp_patent_adr, encoding="utf-8", sep='\t')
        ##################################计算企业的竞争关系#########################################################
        # 取出专利标题及摘要标签
        patent_use = patent_infro[['corp_patent.corp_id', 'corp_patent.patent_name', 'corp_patent.summary']]
        # 将空数据的替换字符（摘要会不存在）
        patent_use.fillna(value=' ', inplace=True)
        # 专利的摘要和标题合并
        patent_use['corp_patent.descrp'] = patent_use.apply(
            lambda x: self.filter_char(' '.join([x['corp_patent.patent_name'], x['corp_patent.summary']])), axis=1)

        # 长度不满足的则剔除
        patent_useful = patent_use[patent_use['corp_patent.descrp'].str.len() > 50]
        # 企业信息对应的企业id
        corp_ids = corp_infor['corp_infor.corp_id'].tolist()
        # 专利知识对应的企业id
        patent_ids = patent_useful['corp_patent.corp_id'].tolist()
        # 取出共有的企业id
        corp_ids = list(set(corp_ids).intersection(set(patent_ids)))
        print('-------------当前企业数据中总计有%d个企业有专利知识-------------' % len(corp_ids))

        # 保存专利维度
        patent_dims = [0]
        # 保存余弦向量
        patent_vecs = []
        # 保存余弦转置
        trans_vecs = []
        # 保存企业id
        corp_nids = []
        # 计数
        count = 0

        for id in corp_ids:
            count += 1
            if count % 100 == 0:
                print('--------企业数据已经计算%d个公司--------' % count)
            # 取出企业的专利信息
            patent_data = patent_use[patent_use['corp_patent.corp_id'] == id]
            # 取出专利信息的列表
            patent_list = patent_data['corp_patent.descrp'].values
            # 取出专利的向量表示
            patent_vec = S.patlis_vector(patent_list)
            # 向量存在转array矩阵
            patent_arr = np.array(patent_vec)
            # 判断向量是二维，存在一些非法字符串 id = 12774103800373211699
            if len(patent_arr) == 2:
                # 当公司专利向量存在才保存
                corp_nids.append(id)
                # 记录专利的真实维度值
                patent_dims.append(patent_arr.shape[0])
                # 构建x/|x|向量
                patent_any = np.divide(patent_arr,
                                       np.sqrt(np.multiply(patent_arr, patent_arr).sum(axis=1))[:, np.newaxis])
                # [x1,x2,x3]^T
                patent_vecs.append(patent_any)
                # [x1,x2,x3]
                trans_vecs.append(patent_any.T)

        print('-------------------其中企业存在有效专利共计%d----------------' % len(corp_nids))

        #  原始矩阵的竖向量
        original_vec = np.vstack(tuple(patent_vecs))  # 竖向拼接
        #  转置矩阵的横向量
        transpose_vec = np.hstack(tuple(trans_vecs))  # 横向拼接
        #  直接计算矩阵余弦距离
        map_mat = np.dot(original_vec, transpose_vec)

        print('-----------------所有企业的关系矩阵构建完毕！！----------------')

        corp_id1s = []
        corp_id2s = []
        score_lis = []

        # 取企业第一列id
        for corp_id1 in corp_nids:
            # id1的索引
            id1_index = corp_nids.index(corp_id1)
            # 取企业第二列id
            for corp_id2 in corp_nids[id1_index + 1:]:
                # 保存企业id1
                corp_id1s.append(corp_id1)
                # 保存企业id2
                corp_id2s.append(corp_id2)
                # id2的索引
                id2_index = corp_nids.index(corp_id2)
                # 计算的矩阵切割
                mat_split = map_mat[sum(patent_dims[:id1_index + 1]):sum(patent_dims[:id1_index + 2]),
                            sum(patent_dims[:id2_index + 1]):sum(patent_dims[:id2_index + 2])]
                # 取计算的平均值
                score = np.mean(mat_split)
                # 保存计算的值
                score_lis.append(score)

        data_dict = {'corp_relation.corp_id1': corp_id1s, 'corp_relation.corp_id2': corp_id2s,
                     'corp_relation.similar': score_lis}
        #  dataframe的列表名
        columns = ['corp_relation.corp_id1', 'corp_relation.corp_id2', 'corp_relation.similar']
        #  构建dataframe数据
        corp_similar = pd.DataFrame(data_dict, columns=columns)
        # 保存关系为csv
        corp_similar.to_csv(self.corp_similar_adr, encoding="utf-8", sep='\t')

        print('###########企业的竞争关系属性已经分析完毕！!###########')

    def count_partner(self):
        # 专利列表关系计算
        C = Cooccur('./coo_process/')
        # 打开添加标签的新企业数据
        corp_infor = pd.read_csv(self.corp_added_adr, header=0, index_col=0, encoding="utf-8", sep='\t')
        # 去除无标签的数据
        corp_infor = corp_infor[corp_infor['corp_infor.tags'] != '-1000']
        # 企业信息对应的企业id
        # corp_ids = corp_infor['corp_infor.corp_id'].tolist()
        corp_ids = corp_infor['corp_infor.corp_name'].tolist()
        # 对企业id进行映射以节约空间
        # d = {}
        # for i in range(len(corp_ids)):
        #	d[i] = corp_ids[i]
        # with open('corp_id_map.json', 'w', encoding='utf-8') as f:
        #	json.dump(d, f)

        print('-------------当前企业数据中总计有%d个企业有标注标签-------------' % len(corp_ids))
        corp_tags = [str(t).split('||') for t in tqdm(corp_infor['corp_infor.tags'])]

        corp_id1s = []
        corp_id2s = []
        score_lis = []

        for i in trange(len(corp_ids)):
            for j in range(i, len(corp_ids)):
                # corp_id1s.append(corp_ids[i])
                # corp_id2s.append(corp_ids[j])
                # corp_id1s.append(i)
                # corp_id2s.append(j)
                value = round(C.jude_partner(corp_tags[i], corp_tags[j]), 6)
                if value > 0:
                    corp_id1s.append(corp_ids[i])
                    corp_id2s.append(corp_ids[j])
                    score_lis.append(value)
            # value =C.jude_partner(corp_tags[i],corp_tags[j])

        data_dict = {'corp_relation.corp_id1': corp_id1s, 'corp_relation.corp_id2': corp_id2s,
                     'corp_relation.cooccur': score_lis}
        #  dataframe的列表名
        columns = ['corp_relation.corp_id1', 'corp_relation.corp_id2', 'corp_relation.cooccur']
        #  构建dataframe数据
        corp_partner = pd.DataFrame(data_dict, columns=columns)
        # 保存关系为csv
        corp_partner.to_csv(self.corp_partner_adr, encoding="utf-8", sep='\t')

        print('###########当前数据合作关系属性已经分析完毕！!###########')

    def merge_relation(self):
        self.count_similar()
        self.count_partner()

        # 打开竞争关系
        corp_similar = pd.read_csv(self.corp_similar_adr, header=0, index_col=0, encoding="utf-8", sep='\t')
        # 打开合作关系
        corp_partner = pd.read_csv(self.corp_partner_adr, header=0, index_col=0, encoding="utf-8", sep='\t')

    # 合并两者数据


if __name__ == '__main__':
    corp = CorpProcess('./corp_data/', 'corp_inforns/', 'corp_patents/', 'corp_addeds/')
    # 计算产业分类和武汉标注
    # corp.count_addinfro()
    # 统计武汉标注比例
    # corp.plot_wuhzoom()
    # corp.merge_relation()
    # corp.count_addinfro(50)
    # corp.count_similar()
    corp.count_partner()
