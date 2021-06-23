import numpy as np


# 对numpy一些运算进行封装
class Mytools:
    # 对矩阵a的行向量与b的列向量分别做cosine相似度,a=[n*v],b=[v,m],a_l2=[n,1],b_l2=[1,m]
    def cosineMats(self, a, b, a_l2=[], b_l2=[]):
        n = np.size(a, 0)
        m = np.size(b, 1)
        if len(a_l2) == 0:
            a_l2 = np.array([1 / np.linalg.norm(a[i, :]) for i in range(n)]).reshape((n, 1))
        if len(b_l2) == 0:
            b_l2 = np.array([1 / np.linalg.norm(b[:, i]) for i in range(m)]).reshape((1, m))
        denominators = np.matmul(a_l2, b_l2)
        numerators = np.matmul(a, b)
        return np.multiply(denominators, numerators)

    # 拼接列表中的所有矩阵
    def connactMatFromList(self, a):
        k = len(a)
        if k == 1:
            return a[0]
        else:
            ret = np.append(a[0], a[1], axis=0)
            for i in range(2, k):
                ret = np.append(ret, a[i], axis=0)
        return ret
