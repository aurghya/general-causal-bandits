import numpy as np
import networkx as nx
import pandas as pd
import itertools
import pdb

class DisjointSet(object):
    def __init__(self, nodes):
        self.n = len(nodes)
        self.nodes = nodes
        self.parent = {}
        self.rank = {}
        for x in nodes:
            self.parent[x] = x
            self.rank[x] = 1
        self.DisjointSets = None
    
    def find(self, x):
        if self.parent[x]!=x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x, y):
        xset = self.find(x)
        yset = self.find(y)

        if xset == yset:
            return
        if self.rank[xset] < self.rank[yset]:
            self.parent[xset] = yset
        elif self.rank[xset] > self.rank[yset]:
            self.parent[yset] = xset
        else:
            self.parent[yset] = xset
            self.rank[xset] = self.rank[xset] + 1
    
    def get_sets(self, unobserved_nodes):
        partitions = {self.parent[x]:[] for x in self.nodes}
        for n in self.nodes:
            partitions[self.parent[n]].append(n)
        for x in partitions.keys():
            partitions[x] = list(set(partitions[x]) - set(unobserved_nodes))
        return partitions
    

class CBN(object):
    def __init__(self, observed_vars, unobserved_vars, edges):

        self.observed_nodes = observed_vars
        self.unobserved_nodes = unobserved_vars
        self.nodes = set(observed_vars).union(set(unobserved_vars))
        
        # create graph
        self.G = nx.DiGraph()
        self.G.add_nodes_from(self.nodes)
        self.G.add_edges_from(edges)

        # Essential components
        self.c_components = self.get_c_component(self.G, self.unobserved_nodes)
        self.node_to_c_component = {}
        for s in self.c_components.keys():
            for v in self.c_components[s]:
                self.node_to_c_component[v] = s

        # samples and parameters
        self.df = None
        self.N = 10000
        self.t = 10
    
    # Given a graph and unobserved variables find c components
    def get_c_component(self, G, unobserved_nodes):
        nodes = set(G.nodes)
        disjoint_sets = DisjointSet(nodes)
        for v in self.unobserved_nodes:
            for x in list(G.succesors(v)):
                disjoint_sets.union(v, x)
        return disjoint_sets.get_sets(unobserved_nodes)
    
    def reduce_graph(self, G, X, Y):
        # Get nodes for H_i, W
        S = self.c_components[self.node_to_c_component[X]]
        W = set(S)
        for v in S:
            W = W.union(set(G.predecessors(v)))
        W = W.union(set([Y]))
        W = W - set(self.unobserved_nodes)

        # U_h are uobserved in H_i
        U_h = set(self.nodes) - W
        H = nx.DiGraph()
        H.add_nodes_from(W)
        
        # H_i complement for computing edges in H_i
        Hc = nx.DiGraph(self.G)
        for e in list(Hc.edges):
            if e[0] in W and e[1] in W:
                Hc.remove_edge(*e)
        
        # add edges in H_i
        U_hix = []
        k = 0
        for v1 in W:
            for v2 in W:
                if v1==v2:
                    continue
                if v1 in list(self.G.successors(v2)) or v1 in list(nx.descendants(Hc, v2)):
                    H.add_edge(v2, v1)
                elif v2 in list(self.G.successors(v1)) or v2 in list(nx.descendants(Hc, v1)):
                    H.add_edge(v1, v2)
                for u in U_h:
                    d = list(nx.descendants(Hc, u))
                    if v1 in d and v2 in d:
                        u_new = 'Uix' + str(k)
                        H.add_edge(u_new, v1)
                        H.add_edge(u_new, v2)
                        U_hix.append(u_new)
                        k = k+1
                        break
        return H, U_hix

    def learn_dix(self, H_i, U_ix, X, x, Y):
        D_ix = {}
        parents = {}
        nodes = list(nx.topological_sort(H_i))
        S = self.get_c_component(H_i, U_ix)
        Sx = None
        for s in S.keys():
            c_comp = set(S[s])
            for v in S[s]:
                c_comp = c_comp.union(H_i.predecessors(v))
            c_comp = c_comp - set(U_ix)
            for v in S[s]:
                parents[v] = c_comp.intersection(set(nodes[:nodes.index(v)]))
            
            if X in S[s]:
                Sx = s

        observed_nodes = set(nodes) - set(U_ix)
        for v in observed_nodes:
            D_ix[v] = {}
            pa = sorted(list(parents[v]))

            if v in S[Sx]:
                df_var = self.df[pa + [v]]
                D_ix[v]['parents'] = pa
                if len(pa) == 0:
                    D_ix[v]['distribution'] = (np.sum(df_var[v])+1)/(len(df_var[v])+2)
                else:
                    pa_val = list(itertools.product([0, 1], repeat=len(pa)))
                    D_ix[v]['distribution'] = {tuple(z):0.5 for z in pa_val}
                    cpd_var_df = (df_var.groupby(pa)[v].sum()/df_var.groupby(pa)[v].count()).reset_index(name = 'probability')
                    for row in cpd_var_df.values:
                        D_ix[v]['distribution'][tuple(row[:-1])] = row[-1]
            else:
                if X in pa:
                    pa = pa[:pa.index(X)] + pa[pa.index(X)+1:]
                    df_var = self.df[pa + [v]].loc[self.df[X]==x]
                    D_ix[v]['parents'] = pa
                    if len(pa) == 0:
                        D_ix[v]['distribution'] = (np.sum(df_var[v])+1)/(len(df_var[v])+2)
                    else:
                        pa_val = list(itertools.product([0, 1], repeat=len(pa)))
                        D_ix[v]['distribution'] = {tuple(z):0.5 for z in pa_val}
                        cpd_var_df = (df_var.groupby(pa)[v].sum()/df_var.groupby(pa)[v].count()).reset_index(name = 'probability')
                        for row in cpd_var_df.values:
                            D_ix[v]['distribution'][tuple(row[:-1])] = row[-1]
                else:
                    df_var = self.df[pa + [v]]
                    D_ix[v]['parents'] = pa
                    if len(pa) == 0:
                        D_ix[v]['distribution'] = (np.sum(df_var[v])+1)/(len(df_var[v])+2)
                    else:
                        pa_val = list(itertools.product([0, 1], repeat=len(pa)))
                        D_ix[v]['distribution'] = {tuple(z):0.5 for z in pa_val}
                        cpd_var_df = (df_var.groupby(pa)[v].sum()/df_var.groupby(pa)[v].count()).reset_index(name = 'probability')
                        for row in cpd_var_df.values:
                            D_ix[v]['distribution'][tuple(row[:-1])] = row[-1]
        return D_ix
    
    def generate_samples(self, D_ix, nodes, N):
        df = {}
        for v in nodes:
            df[v] = []
        
        # pdb.set_trace()
        
        for _ in range(N):
            for v in nodes:
                p = 0
                q = D_ix[v]
                if 'parents' in q:

                    z = tuple([df[pa][-1] for pa in q['parents']])
                    p = q['distribution'][z]
                else:
                    p = q['distribution']
                df[v].append(np.random.binomial(1, p))

        return pd.DataFrame(df)

    def estimate_mu_ix(self, X, x, Y):
        H_i, U_hix = self.reduce_graph(self.G, X, Y)
        D_ix = self.learn_dix(H_i, U_hix, X, x, Y)
        top_nodes = list(nx.topological_sort(H_i))
        nodes = []
        for v in top_nodes:
            if v not in U_hix:
                nodes.append(v)
        M = self.generate_samples(D_ix, nodes, self.N)
        return np.sum(M[Y])/len(M[Y])
    
    def estimate_mu(self, df, N, X, Y, t):
        self.df = df
        self.N = N
        self.t = t
        mu = {a:self.estimate_mu_ix(a[0], a[1], Y) for a in X}
        return mu
    
    def identify_infrequent(self, df, X):
        N = len(df)
        parent = {x:set() for x in X}

        for x in X:
            parent[x] = parent[x].union(set(self.G.predecessors(x)))
        
        q = []
        vals = [0, 1]
        for v in X:
            q_i = []
            for x in vals:
                pa = sorted(list(parent[v]))
                pa_val = list(itertools.product([0, 1], repeat=len(pa)))
                dist = {tuple(z):0 for z in pa_val}

                df_var = df[pa + [v]].loc[df[v]==x]
                cpd_var_df = (df_var.groupby(pa + [v]).size()/N).reset_index(name='probability')
                for row in cpd_var_df.values:
                    dist[tuple(row[:-2])] = row[-1]
                
                q_ix = 1
                for z in dist.keys():
                    if dist[z] < q_ix:
                        q_ix = dist[z]
                
                q_i.append(q_ix)
            
            q.append((min(q_i), v))
        
        q.sort(key=lambda x:x[0])

        m = 1
        for i, q_ix in enumerate(q):
            if q_ix[0] > 1.0/(i+1):
                return i, q
        
        return len(q), q