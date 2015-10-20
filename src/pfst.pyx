#!/usr/bin/env python
# -*- coding: utf-8 -*-
#cython: boundscheck=False
#cython: wraparound=False
#cython: nonecheck=False
#cython: cdivision=True
#cython: infertypes=True
#cython: c_string_type=unicode, c_string_encoding=ascii
#cython: profile=False
#distutils: language = c++
#distutils: libraries = ['stdc++']
#distutils: extra_compile_args = ["-std=c++11"]

import fst
import itertools as it
import numpy as np
from numpy import zeros, ones, exp, log
from collections import defaultdict as dd
from scipy.optimize import fmin_l_bfgs_b as lbfgs
from arsenal.alphabet import Alphabet
from fst_ryan import FST
import cProfile

cimport fst._fst as fst
cimport fst.libfst as libfst

from libc.math cimport log, exp
from libcpp.vector cimport vector
from libcpp.map cimport map
from libcpp.pair cimport pair
from cython.operator cimport dereference as deref, preincrement as inc #derefere

cdef class PFST(object):
    """
    Stochastic Edit Distance
    """
    cdef int atoms
    cdef vector[pair[int, vector[int]]] features
    cdef map[int, vector[int]] features_map

    cdef vector[vector[pair[int, vector[int]]]] data_features
    cdef double[:] theta
    cdef fst.LogVectorFst machine
    cdef libfst.LogVectorFst *x
    cdef libfst.LogVectorFst *y
    
    def __init__(self, alphabet):
        super(PFST, self).__init__(alphabet)
        self.alphabet = alphabet

        # features
        self.atoms = -1
        self.features = vector[pair[int, vector[int]]]()
        self.features_map = map[int, vector[int]]()

        self.feature2origin = []
        self.data_features = vector[vector[pair[int, vector[int]]]]()

        self.theta = np.zeros((1))

    def create_features(self):
        " Extract the atomic features "

        self.features = vector[pair[int, vector[int]]]()
        self.feature2origin = [-1]
        counter = 1
        for state, arcs in self.ids.items():
            lst = []
            for i, (arc, tup) in enumerate(arcs.items()):
                action, value, ngram = tup
                lst.append(counter)
                self.feature2origin.append(state)
                counter += 1
            self.features.push_back((state, lst))
            
        # also put in map
        for i, lst in self.features:
            self.features_map[i] = lst

        self.atoms = counter


    def feature_on_arcs(self):
        " Put the feature integers on the arcs "
        
        for state_id, lst in self.features:
            state = self.machine[state_id]
            i = 0
            for arc in state:
                arc.weight = fst.LogWeight(lst[i])
                i += 1


    def extract_features(self, data):
        " Extract the features on the arcs "
        
        self.data_features = vector[vector[pair[int, vector[int]]]]()

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
                for arc in state:
                    lst.append(int(arc.weight))

                features.append((i, lst))

            self.data_features.push_back(features)


    def lll(self, x, y):
        " examplar log-likelihood "
        cdef fst.LogVectorFst _x = x
        cdef fst.LogVectorFst _y = y
        return self._lll(_x.fst, _y.fst)
        
    cdef double _lll(self, libfst.LogVectorFst *x, libfst.LogVectorFst *y):
        " examplar log-likelihood "

        cdef libfst.LogVectorFst machine = self.machine.fst[0]
        cdef libfst.LogVectorFst *middle = new libfst.LogVectorFst()
        cdef libfst.LogVectorFst *composed = new libfst.LogVectorFst()

        middle.SetInputSymbols(machine.MutableInputSymbols())
        middle.SetOutputSymbols(machine.MutableOutputSymbols())
        composed.SetInputSymbols(machine.MutableInputSymbols())
        composed.SetOutputSymbols(machine.MutableOutputSymbols())

        #cdef libfst.ILabelCompare[libfst.LogArc] icomp
        #libfst.ArcSort(x, icomp)
        libfst.Compose(x[0], machine, middle)
        #libfst.ArcSort(middle, icomp)
        libfst.Compose(middle[0], y[0], composed)

        cdef vector[libfst.LogWeight] betas
        libfst.ShortestDistance(composed[0], &betas, True)

        del middle
        del composed
        return betas[0].Value()


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
            print alphas
            print betas
            for state_id, lst in features:
                state = result[state_id]
                for j, arc in enumerate(state):
                    feat = lst[j]
                    if feat != 0.0:
                        score = exp(-float(alphas[state_id] * betas[arc.nextstate] * arc.weight / Z))
                        g[feat] -= score
                        counts[self.feature2origin[feat]] += score
   
        # TODO:  can be made more efficient?
        for i, lst in self.features:
            if i in counts:
                v = counts[i]

                state = self.machine[i]
                #if lst[0] != -1:
                #    p = exp(-float(state.final))
                #    g[lst[0]] += p * v

                for j, arc in enumerate(state):
                    p = exp(-float(arc.weight))
                    g[lst[j]] += p * v

        return g

    cdef void _observed(self, int i, libfst.LogVectorFst *x, libfst.LogVectorFst *y, double[:] g, map[int, double] counts):
        " computes the observed counts for a given x, y pair "

        cdef libfst.LogVectorFst machine = self.machine.fst[0]
        cdef libfst.LogVectorFst *middle = new libfst.LogVectorFst()
        cdef libfst.LogVectorFst *composed = new libfst.LogVectorFst()

        middle.SetInputSymbols(machine.MutableInputSymbols())
        middle.SetOutputSymbols(machine.MutableOutputSymbols())
        composed.SetInputSymbols(machine.MutableInputSymbols())
        composed.SetOutputSymbols(machine.MutableOutputSymbols())

        #cdef libfst.ILabelCompare[libfst.LogArc] icomp
        #libfst.ArcSort(x, icomp)
        libfst.Compose(x[0], machine, middle)
        #libfst.ArcSort(middle, icomp)
        libfst.Compose(middle[0], y[0], composed)

        cdef vector[libfst.LogWeight] alphas
        cdef vector[libfst.LogWeight] betas

        libfst.ShortestDistance(composed[0], &betas, False)
        libfst.ShortestDistance(composed[0], &betas, True)

        cdef double logZ = betas[0].Value()
        cdef double score
        cdef int j, state_id, feat

        """
        features = self.data_features[i]
        for state_id, lst in features:
                state = result[state_id]
                for j, arc in enumerate(state):
                    feat = lst[j]
                    if feat != 0.0:
                        score = exp(-float(alphas[state_id] * betas[arc.nextstate] * arc.weight / Z))
                        g[feat] -= score
                        counts[self.feature2origin[feat]] += score

        """

    cdef void _grad(self, data, double[:] g):
        " gradient for locally normalized models "
        self._local_renormalize(self.theta)
        """
        # TODO : to fix
        #counts = dd(float)
        cdef map[int, double] counts = map[int, double]()
        cdef int i
        for i, (x, y) in enumerate(data):
            result = x >> self.machine >> y
            alphas = result.shortest_distance()
            betas = result.shortest_distance(True)
            Z = betas[0]
            features = self.data_features[i]
            print alphas
            print betas
            for state_id, lst in features:
                state = result[state_id]
                for j, arc in enumerate(state):
                    feat = lst[j]
                    if feat != 0.0:
                        score = exp(-float(alphas[state_id] * betas[arc.nextstate] * arc.weight / Z))
                        g[feat] -= score
                        counts[self.feature2origin[feat]] += score
   
        # TODO:  can be made more efficient?
        for i, lst in self.features:
            if i in counts:
                v = counts[i]

                state = self.machine[i]
                #if lst[0] != -1:
                #    p = exp(-float(state.final))
                #    g[lst[0]] += p * v

                for j, arc in enumerate(state):
                    p = exp(-float(arc.weight))
                    g[lst[j]] += p * v

        return g
        """

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
            
                        
    def copy(self, f):
        " Copy a transducer "

        if not isinstance(f, fst.LogVectorFst):
            raise("Requires Log-Transducer")
        
        g = fst.LogVectorFst()
        g.isyms = f.isyms
        g.osyms = f.osyms
        for _ in xrange(len(f)):
            g.add_state()
        g.start = f.start

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
        self._local_renormalize(self.theta)
        return
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

    cdef void _local_renormalize(self, double[:] theta):
        """
        Locally renormalize the PFST. 

        Cython Note: This function should be completely white.
        """
        cdef int state_id, i, atom, index
        cdef vector[int] lst
        cdef double Z, logZ, weight

        cdef libfst.LogVectorFst machine = self.machine.fst[0]
        cdef libfst.LogArc *arc
        cdef libfst.ArcIterator[libfst.LogVectorFst]* it
        cdef vector[pair[int, vector[int]]].iterator vit
        cdef pair[int, vector[int]] p
        vit = self.features.begin()
        
        while vit != self.features.end():
            p = deref(vit)
            state_id = p.first
            lst = p.second
        
            Z = 0.0
            for atom in lst:
                if atom != -1:
                    Z += exp(theta[atom])
            logZ = log(Z)

            i = 0
            it = new libfst.ArcIterator[libfst.LogVectorFst](machine, state_id)
            while not it.Done():
                index = lst[i]
                weight = -theta[index]+logZ
                arc = <libfst.LogArc*> &it.Value()
                arc.weight.SetValue(weight)
                i += 1
                it.Next()
            del it
            inc(vit)


    def train(self, data):
        " trains the machine using L-BFGS "
        def f(theta):
            self.theta = theta
            return self.ll(data)
    
        def g(theta):
            self.theta = theta
            return self.grad(data)
    
        lbfgs(f, self.theta, fprime=g, disp=2)


    property atoms:
        def __get__(self):
            return self.atoms
        
    property theta:
        def __get__(self):
            return self.theta
        def __set__(self, theta):
            self.theta = theta

    property machine:
        def __get__(self):
            return self.machine
        def __set__(self, machine):
            self.machine = machine
