import pandas as pd
from tagprocessstakes import IndClassifyStake
from tqdm import trange

class_num = 212
I = IndClassifyStake(200, class_num, './', test=False)
file = pd.read_excel('sample.xlsx')
d = {
    'name': [],
    'capital': [],
    'scope': [],
    'ind1': [],
    'ind2': [],
    'ind3': []
}
for i in trange(len(file)):
    row = file.iloc[i]
    scope = row['scope']
    name = row['name']
    capital = row['capital']
    res = I.single_match_scope(scope=scope, tags='-1000')
    d['name'].append(name)
    d['capital'].append(capital)
    d['scope'].append(scope)
    d['ind1'].append(res[0])
    d['ind2'].append(res[1])
    d['ind3'].append(res[2])

df = pd.DataFrame(d)
df.to_excel('result_{}.xlsx'.format(class_num))
