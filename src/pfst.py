import fst
import itertools as it
import numpy as np
from numpy import zeros, ones, exp, log
from collections import defaultdict as dd
from scipy.optimize import fmin_l_bfgs_b as lbfgs
from arsenal.alphabet import Alphabet
from fst_ryan import FST
import cProfile


class PFST(FST):
    """
    Stochastic Edit Distance
    """
    def __init__(self, alphabet):
        super(PFST, self).__init__(alphabet)
        self.alphabet = alphabet

        # features
        self.atoms = -1
        self.features = []
        self.feature2origin = []
        self.data_features = []


    def extract_features(self, data):
        " Extract the features on the arcs "
        
        self.data_features = []
        self.feature_on_arcs()
        for x, y in data:
            x_prime = self.copy(x)
            y_prime = self.copy(y)
            self.to_zeros(x_prime)
            self.to_zeros(y_prime)
            result = x_prime >> self.machine >> y_prime

            features = []
            for i, state in enumerate(result):
                lst = []
                if state.final != fst.LogWeight.ZERO:
                    lst.append(int(state.final))
                else:
                    lst.append(-1)
                    
                for arc in state:
                    lst.append(int(arc.weight))

                features.append((i, lst))

            self.data_features.append(features)


    def lll(self, x, y):
        " examplar log-likelihood "

        result = x >> self.machine >> y
        return -float(result.shortest_distance(True)[0])

        
    def ll(self, data):
        " log-likelihood for locally normalized models "

        self.local_renormalize()        
        ll = 0.0
        for x, y in data:
            # TODO: fix this!
            ll -= self.lll(x, y)

        return ll


    def grad_fd(self, data, EPS=0.1):
        " gradient for the locally normalized models with a finite-difference "
        
        g = zeros((self.atoms))
        for x, y in data:
            for i in xrange(self.atoms):
                self.theta[i] += EPS
                self.local_renormalize()

                ll1 = self.lll(x, y)
                self.theta[i] -= 2 * EPS
                self.local_renormalize()
                ll2 = self.lll(x, y)
                self.theta[i] += EPS
                self.local_renormalize()
            
                val = (ll2 - ll1) / (2.0 * EPS)
                g[i] += val

        return g


    def grad(self, data):
        " gradient for locally normalized models "
        self.local_renormalize()
        g = zeros((self.atoms))
        # TODO : to fix
        counts = dd(float)
        for i, (x, y) in enumerate(data):
            result = x >> self.machine >> y
            alphas = result.shortest_distance()
            betas = result.shortest_distance(True)
            Z = betas[0]

            features = self.data_features[i]
            for state_id, lst in features:
                if lst[0] == -1:
                    pass
                else:
                    feat = lst[0]
                    score = exp(-float(alphas[state_id] * betas[state_id] / Z))
                    g[feat] -= score
                    counts[self.feature2origin[feat]] += score

                state = result[state_id]
                for j, arc in enumerate(state):
                    feat = lst[j+1]
                    if feat != 0.0:
                        score = exp(-float(alphas[state_id] * betas[arc.nextstate] * arc.weight / Z))
                        g[feat] -= score
                        counts[self.feature2origin[feat]] += score
   
        # TODO:  can be made more efficient?
        for i, lst in self.features:
            if i in counts:
                v = counts[i]

                state = self.machine[i]
                if lst[0] != -1:
                    p = exp(-float(state.final))
                    g[lst[0]] += p * v

                for j, arc in enumerate(state):
                    p = exp(-float(arc.weight))
                    g[lst[j+1]] += p * v

        return g

    def train(self, data):
        " trains the machine using L-BFGS "
        def f(theta):
            self.theta = theta
            return self.ll(data)
    
        def g(theta):
            self.theta = theta
            return self.grad(data)
    
        lbfgs(f, self.theta, fprime=g, disp=2)


    def decode(self, data, n=1):
        " decode the data "

        strings = []
        for x in data:
            result = fst.StdVectorFst(x >> self.machine)
            result.project_output()
            best = result.shortest_path(n=n)
            for path in best.paths():
                string = ""
                for arc in path:
                    if arc.olabel != 0:
                        string += best.osyms.find(arc.olabel)
                strings.append(string)
        return strings
            
                        
