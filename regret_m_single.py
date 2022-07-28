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
T = 1500

def get_best_arm(seed):
    np.random.seed(seed)
    srm_algorithm = SRM_ALG(G, T)
    srm_arm, info = srm_algorithm.run()
    mab_algorithm = MAB_ALG(G, T)
    mab_arm = mab_algorithm.run()
    return srm_arm, mab_arm

if __name__ == '__main__':
    m_arr = list(range(10, 51, 2))

    n_sim = 100
    seeds = list(range(n_sim))
    
    simulations = []
    for m in m_arr:
        
        random.seed(42)
        G = GeneralGraph(50, 2, m, epsilon, 45)

        best_arms = None
        with Pool(100) as p:
            best_arms = p.map(get_best_arm, seeds)
            # simulations.append(best_arms)
        
        correct_srm = 0
        correct_mab = 0
        for a in best_arms:
            if a[0] == (45, 1):
                correct_srm += 1
            if a[1] == (45, 1):
                correct_mab += 1

        simulations.append((correct_srm/n_sim, correct_mab/n_sim))
        print(correct_srm/n_sim, correct_mab/n_sim)
    
    with open('regret_m_single.pkl', 'wb') as f:
        pickle.dump(np.array(simulations), f)
