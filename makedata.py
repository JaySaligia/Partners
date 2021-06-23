# 随机找出100,000条武汉企业数据作为样本
import pandas as pd
from tqdm import trange
from random import randint

file_1 = pd.read_excel('./data/corps/武汉/wuhan_1.xlsx')
count = 0
d = {
    'name': [],
    'capital': [],
    'scope': []
}
for i in trange(len(file_1)):
    row = file_1.iloc[i]
    name = row['tmp_register_tag_wuhan_all.corp_name']
    capital = row['tmp_register_tag_wuhan_all.capital']
    scope = str(row['corp_infor.business_scope'])
    if len(scope) > 10 and randint(0, 9) < 7:
        # 加入序列
        d['name'].append(name)
        d['capital'].append(capital)
        d['scope'].append(scope)
        count += 1
        if count > 99999:
            break

df = pd.DataFrame(d)
df.to_excel('sample.xlsx')
