import pdb
from networkx.linalg.algebraicconnectivity import _rcm_estimate
from cbn import CBN
import numpy as np
import networkx as nx
import pandas as pd

class MAB_ALG(object):
    label = "Algorithm MAB"

    def __init__(self, model, T):
        self.model = model
        self.T = T
        self.X = model.X
        self.Y = model.Y
    
    def run(self):
        mu_ix = {}

        for X in self.X:
            for x in [0, 1]:
                s = self.model.intervene(X, x, self.T//(2*len(self.X)-1))
                mu_ix[(X, x)] = np.sum(s[self.Y])/len(s[self.Y])
        
        df = self.model.sample(self.T//(2*len(self.X)-1))
        mu_ix[0] = np.sum(df[self.Y])/len(df)

        best_action = 0
        for a in mu_ix.keys():
            if mu_ix[a] > mu_ix[best_action]:
                best_action = a
        
        return best_action
