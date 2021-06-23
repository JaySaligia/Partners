# 测试语义桩的结果
from tagprocessstakes import IndClassifyStake
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from tqdm import trange

# 制作41分类
# 读取分类信息
# with open('45_raw.txt', encoding='utf-8') as f:
#     names_45 = f.readlines()
#     for i in range(len(names_45)):
#         names_45[i] = names_45[i].replace('\n', '')
#
# with open('41_raw.txt', encoding='utf-8') as f:
#     names_41 = f.readlines()
#     for i in range(len(names_41)):
#         names_41[i] = names_41[i].replace('\n', '')
#
# d = {}
# with open('45_new.txt', encoding='utf-8') as f:
#     lines = f.readlines()
#     tmp = -1
#     for line in lines:
#         line = line.replace('\n', '')
#         if line.isdigit():
#             d[int(line)] = []
#             tmp = int(line)
#         else:
#             d[tmp].append(line+'\n')
#
# with open('41_new.txt', 'w', encoding='utf-8') as f:
#     for i in range(len(names_41)):
#         index_45 = names_45.index(names_41[i])
#         f.write('{}\n'.format(i))
#         f.writelines(d[index_45])
results = []
for iter in range(6):
    cluster_num = int((iter+1) * 50)
    I = IndClassifyStake(cluster_num, 41, './', test=True)
    # print(I.single_match_scope('汽车发动机', '汽车销售||摩托车改装||汽车配件'))
    test = pd.read_csv('./test/test.csv')
    groundTruth = []
    predict = []
    for i in trange(len(test)):
        y = int(test.iloc[i][0])
        groundTruth.append(y)
        y_s = I.single_match_scope(test.iloc[i][1], '-1000')
        y_ = y_s[0]
        for res in y_s:
            if res == y:
                y_ = y
        predict.append(y_)
    result = 'cluster:{}\n'.format(cluster_num)
    result += 'acc:{}\n'.format(accuracy_score(groundTruth, predict))
    result += 'prec:{}\n'.format(precision_score(groundTruth, predict, average='macro'))
    result += 'rec:{}\n'.format(recall_score(groundTruth, predict, average='macro'))
    result += 'f1:{}\n'.format(f1_score(groundTruth, predict, average='macro'))
    result += '\n'
    results.append(result)
with open('result_top3.txt', 'w', encoding='utf-8') as f:
    f.writelines(results)

    # print('acc:{}'.format(accuracy_score(groundTruth, predict)))
    # print('prec:{}'.format(precision_score(groundTruth, predict, average='macro')))
    # print('rec:{}'.format(recall_score(groundTruth, predict, average='macro')))
    # print('f1:{}'.format(f1_score(groundTruth, predict, average='macro')))

