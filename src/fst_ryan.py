import fst
import itertools as it
import numpy as np
from numpy import zeros, ones, exp, log
from collections import defaultdict as dd
from scipy.optimize import fmin_l_bfgs_b as lbfgs
from arsenal.alphabet import Alphabet
import cProfile

class FST(object):
    """

    """
    def __init__(self, alphabet):
        self.alphabet = alphabet

        # features
        self.atoms = -1
        self.features = []
        self.feature2origin = []
        self.data_features = []

    def create_features(self):
        " Extract the atomic features "

        self.features = []
        self.feature2origin = [-1]
        counter = 1
        for state, arcs in self.ids.items():
            lst = []
            for i, (arc, tup) in enumerate(arcs.items()):
                if i == 0 and arc != 0:
                    lst.append(-1)
                action, value, ngram = tup
                lst.append(counter)
                self.feature2origin.append(state)
                counter += 1
            self.features.append((state, lst))
        self.atoms = counter


    def feature_on_arcs(self):
        " Put the feature integers on the arcs "
        
        for state_id, lst in self.features:
            state = self.machine[state_id]

            if lst[0] != -1:
                state.final = fst.LogWeight(lst[0])
            i = 1
            for arc in state:
                arc.weight = fst.LogWeight(lst[i])
                i += 1

        """
        for state in self.machine:
            print "STATE", state, float(state.final)
            for arc in state:
                print "ARC", arc, float(arc.weight)
        print; print; print
        import sys; sys.exit(0)
        """
                
    def copy(self, f):
        " Copy a transducer "

        if not isinstance(f, fst.LogVectorFst):
            raise("Requires Log-Transducer")
        
        g = fst.LogVectorFst()
        g.isyms = self.sigma
        g.osyms = self.sigma
        for _ in xrange(len(f)):
            g.add_state()
        g.start = 0

        for i, state in enumerate(f):
            g[i].final = state.final
            for j, arc in enumerate(state):
                g.add_arc(i, arc.nextstate, arc.ilabel, arc.olabel, arc.weight)
            
        return g
        
    def to_zeros(self, f):
        " Zero out the arcs "

        for state in f:
            if state.final != fst.LogWeight.ZERO:
                state.final = fst.LogWeight.ONE
            for arc in state:
                arc.weight = fst.LogWeight.ONE


    def local_renormalize(self):
        """
        Locally Renormalize
        """
        for state_id, lst in self.features:
            state = self.machine[state_id]
            Z = 0.0
            for atom in lst:
                if atom != -1:
                    Z += exp(self.theta[atom])
            Z = log(Z)

            if lst[0] != -1:
                state.final = fst.LogWeight(-self.theta[lst[0]]+Z)
            i = 1
            for arc in state:
                arc.weight = fst.LogWeight(-self.theta[lst[i]]+Z)
                i += 1

    def train(self, data):
        " trains the machine using L-BFGS "
        def f(theta):
            self.theta = theta
            return self.ll(data)
    
        def g(theta):
            self.theta = theta
            return self.grad(data)
    
        lbfgs(f, self.theta, fprime=g, disp=2)


