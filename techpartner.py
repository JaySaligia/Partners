# calculate technology partner
from mytools import Mytools
import numpy as np
from tqdm import trange


class TechSimScore:
    def __init__(self):
        self.tools = Mytools()
        self.alpha = 0.3
        self.beta = 0.7

    def cal_socre(self, mat_a, mat_b, mat_c, mat_d):
        global_score = np.mean(self.tools.cosineMats(a=mat_a, b=mat_c.T))
        local_score = np.mean(self.tools.cosineMats(a=mat_b, b=mat_d.T))
        return self.alpha * global_score + self.beta * local_score


t = TechSimScore()
d = np.load('patent_vec.npy', allow_pickle=True).item()
corps = list(d.keys())
length = len(corps)
scores = np.zeros((length, length))
for i in trange(length):
    for j in range(i + 1, length):
        corp1 = corps[i]
        corp2 = corps[j]
        a_ = d[corp1][0]
        b_ = d[corp1][1]
        c_ = d[corp2][0]
        d_ = d[corp2][1]
        score = t.cal_socre(a_, b_, c_, d_)
        scores[i][j] = score

store = {
    'names': corps,
    'scores': scores
}
np.save('tech_sim_scores', store)
