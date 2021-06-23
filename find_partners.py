#  find chain partner and technology partners
import pandas as pd
import numpy as np


class FindChainPartner:
    def __init__(self):
        chain_result = './result_88.xlsx'
        self.chain_df = pd.read_excel(chain_result)

    def find_chain_partner(self, target_ind, corp_name, top):
        selected_corp = self.chain_df[self.chain_df['name'] == corp_name]
        corp_capital = float(selected_corp.iloc[0]['capital'])
        target_df = self.chain_df[(self.chain_df['ind1'] == target_ind) | (self.chain_df['ind2'] == target_ind) | (
                self.chain_df['ind3'] == target_ind)]
        target_df['weight'] = target_df['capital'].apply(lambda x: abs(float(x) / corp_capital - 1))
        target_df.sort_values(by='weight', inplace=True)
        for i in range(top):
            row = target_df.iloc[i]
            print(row['name'])
            print(row['capital'])
            print(row['scope'])
            print('--------------------')


class FindTechPartner:
    def __init__(self):
        self.scores = np.load('tech_sim_scores.npy', allow_pickle=True).item()
        self.tech_names = self.scores['names']
        self.tech_scores = self.scores['scores']
        self.tech_scores[np.isnan(self.tech_scores)] = 0
        patent_path = r'.\data\patents\query_result1.xlsx'
        df = pd.read_excel(patent_path)
        group = df.groupby('tmp_patent_id_wuhan.applicant')
        self.count = {}
        for i, g in enumerate(list(group)):
            self.count[g[0]] = len(g[1])

    def find_tech_partner(self, corp_name):
        corp_index = self.tech_names.index(corp_name)
        res = []
        for i in range(len(self.tech_names)):
            if i < corp_index:
                res.append((self.tech_scores[i][corp_index], self.tech_names[i], self.count[self.tech_names[i]]))
            elif i > corp_index:
                res.append((self.tech_scores[corp_index][i], self.tech_names[i], self.count[self.tech_names[i]]))
            else:
                continue
        res.sort()
        for item in res:
            print(item)


# F = FindChainPartner()
# F.find_chain_partner('房屋建筑业', '武汉华安水泥制品有限公司', 10)
F = FindTechPartner()
F.find_tech_partner('武汉昊诚能源科技有限公司')
