from model import *
from srm_alg import *
from mab import *
import pdb
import random
from multiprocessing import Pool
import numpy as np
import pickle

random.seed(7)
epsilon = 0.3
m = 10
G = None
T = 1000
n_G = 50

def get_graph_list(m):
    G_list = []
    for _ in range(n_G):
        best_arm = random.randint(42, 49)
        G = GeneralGraph(50, 2, m, epsilon, best_arm)
        G_list.append((G, best_arm))
    return G_list

def get_best_arm(args):
    G, seed = args
    np.random.seed(seed)
    srm_algorithm = SRM_ALG(G[0], T)
    srm_arm, info = srm_algorithm.run()
    mab_algorithm = MAB_ALG(G[0], T)
    mab_arm = mab_algorithm.run()
    best_arm = (G[1], 1)
    return srm_arm, mab_arm, best_arm

if __name__ == '__main__':
    m_arr = list(range(10, 51, 2))

    n_sim = 100
    seeds = list(range(n_sim))
    
    simulations = []
    for m in m_arr:
        
        random.seed(42)
        G_list = get_graph_list(m)
        args = [(G, seed) for G in G_list for seed in seeds]

        best_arms = None
        with Pool(100) as p:
            best_arms = p.map(get_best_arm, args)
            # simulations.append(best_arms)
        
        correct_srm = 0
        correct_mab = 0
        for a in best_arms:
            if a[0] == a[2]:
                correct_srm += 1
            if a[1] == a[2]:
                correct_mab += 1

        simulations.append((correct_srm/len(args), correct_mab/len(args)))
        print(correct_srm/len(args), correct_mab/len(args))
    
    with open('regret_m_many_1000.pkl', 'wb') as f:
        pickle.dump(np.array(simulations), f)
