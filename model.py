import networkx as nx
import pandas as pd
import random
from matplotlib.pyplot import draw
from collections import namedtuple
import numpy as np
GeneralBandit = namedtuple('GeneralBandit', 'DAG num_skew_vars m epsilon best_arm_idx')

class GeneralGraph(object):
    def __init__(self, num_nodes, indegree, num_skew_vars, epsilon, best_arm_idx):
        self.num_nodes = num_nodes
        self.indegree = indegree
        self.num_skew_vars = num_skew_vars
        self.epsilon = epsilon
        self.best_arm_idx = best_arm_idx
        self.X = list(range(self.num_nodes))
        self.Y = self.num_nodes
        self.bandit = self.generate_bandit_instance(
                                num_nodes, indegree, 
                                num_skew_vars, 
                                2*num_skew_vars, 
                                epsilon, 
                                best_arm_idx
                            )
        self.unobserved_nodes = []
        self.observed_nodes = list(range(self.num_nodes))
        self.edges = list(self.bandit.DAG.edges)
    
    def set_seed(self, seed):
        np.random.seed(seed)
    
    def get_random_DAG(self, n, k):
        # n is num of X variables and k is max indegree
        edges = []
        DG = nx.DiGraph()
        DG.add_nodes_from(range(n+1))
        for node in range(1, n):
            indegree = random.randint(0, min(node, k))
            edges += [(parent, node)  for parent in random.sample(range(0, node), k=indegree)]
        edges += [(parent, n) for parent in range(n)]
        DG.add_edges_from(edges)
        return DG
    
    def generate_bandit_instance(self, num_nodes, indegree, num_skew_vars, m, epsilon, best_arm_idx):
        DAG = self.get_random_DAG(num_nodes, indegree)
        return GeneralBandit(DAG, num_skew_vars, m, epsilon, best_arm_idx)
    
    def get_sample(self, bandit, interv_idx=None, interv_val=None):
        sample = []
        for node in range(len(bandit.DAG)-1):
            if interv_idx is not None and interv_val is not None and interv_idx==node:
                sample += [interv_val]
            elif node < len(bandit.DAG)-1-bandit.num_skew_vars:
                sample += [np.random.binomial(size=1, n=1, p=0.5)[0]]
            else:
                sample += [np.random.binomial(size=1, n=1, p=1.0/bandit.m)[0]]

        if sample[bandit.best_arm_idx]==1:
            sample += [np.random.binomial(size=1, n=1, p=0.5+bandit.epsilon)[0]]
        else:
            best_arm_prob = 0.5 if bandit.best_arm_idx < len(bandit.DAG)-1-bandit.num_skew_vars else 1.0/bandit.m
            epsilon_prime = best_arm_prob*bandit.epsilon/(1-bandit.epsilon)
            sample += [np.random.binomial(size=1, n=1, p=0.5-epsilon_prime)[0]]
        return sample

    def intervene(self, interv_idx, interv_val, T=1):
        if T==1:
            return np.array(self.get_sample(self.bandit, interv_idx=interv_idx, interv_val=interv_val))
        else:
            s = []
            for _ in range(int(T)):
                s.append(self.get_sample(self.bandit, interv_idx=interv_idx, interv_val=interv_val))
            
            s = pd.DataFrame(np.array(s), columns=list(range(0, self.num_nodes+1)))
            return s

    def sample(self, T=1):
        if T == 1:
            return np.array(self.get_sample(self.bandit))
        else:
            s = []
            for _ in range(int(T)):
                s.append(self.get_sample(self.bandit))
            s = pd.DataFrame(np.array(s), columns=list(range(0, self.num_nodes+1)))
            return s