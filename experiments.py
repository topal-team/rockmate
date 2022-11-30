from mem_tools import *
from rotor import timing
import pickle
res = []
for m in np.arange(10):
    mem = 4.6e8 + m*4e7
    for _ in range(10):
        res.append(experiment(mem,print_res=True))

with open("results2.pkl",'wb') as f:pickle.dump(res, f)
