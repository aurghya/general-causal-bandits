import pdb
import random
from cbn import CBN
import numpy as np
import networkx as nx
import pandas as pd

class SRM_ALG(object):
    label = "Algorithm 1"

    def __init__(self, model, T):
        self.model = model
        self.T = T
        self.X = model.X
        self.Y = model.Y
    
    def run(self):
        df = self.model.sample(self.T//2)
        edges = self.model.edges
        unobserved_nodes = self.model.unobserved_nodes
        observed_nodes = self.model.observed_nodes
        cbn = CBN(observed_nodes, unobserved_nodes, edges)

        m, q = cbn.identify_infrequent(df, self.X)
        A = [(X, x) for _, X in q[m:] for x in [0, 1]]
        mu_ix = cbn.estimate_mu(df, self.T, A, self.Y, 0)

        for _, X in q[:m]:
            for x in [0, 1]:
                s = self.model.intervene(X, x, self.T//(4*m))
                mu_ix[(X, x)] = np.sum(s[self.Y])/len(s[self.Y])
        
        mu_ix[0] = np.sum(df[self.Y])/len(df)

        best_action = 0
        for a in mu_ix.keys():
            if mu_ix[a] > mu_ix[best_action]:
                best_action = a
        
        info = {}        
        return best_action, info
