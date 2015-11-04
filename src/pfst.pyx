#!/usr/bin/env python
# -*- coding: utf-8 -*-
#cython: boundscheck=False
#cython: wraparound=False
#cython: nonecheck=False
#cython: cdivision=True
#cython: infertypes=True
#cython: c_string_type=unicode, c_string_encoding=ascii
#cython: profile=True
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
    cdef int atoms, num_features
    cdef vector[pair[int, vector[int]]] features
    cdef map[int, vector[int]] features_map
    cdef map[int, double] counts

    cdef vector[int] feature2origin
    cdef vector[vector[pair[int, vector[int]]]] data_features

    # NOTE: having two different weight vectors
    # allows one to switch between optimizing in ``feature space''
    # and ``arc space'' fluidly 

    # one ``theta'' per arc
    cdef double[:] theta
    # one weight per feature
    cdef double[:] weights 

    cdef fst.LogVectorFst machine
    cdef libfst.LogVectorFst *x
    cdef libfst.LogVectorFst *y

    cdef vector[vector[int]] attributes
    
    def __init__(self, alphabet):
        super(PFST, self).__init__(alphabet)
        self.alphabet = alphabet

        # features
        self.atoms = -1
        self.features = vector[pair[int, vector[int]]]()
        self.features_map = map[int, vector[int]]()

        self.feature2origin = vector[int]()
        self.data_features = vector[vector[pair[int, vector[int]]]]()
        self.int2feat = {}

        self.theta = np.zeros((1))
        self.weights = np.zeros((1))

        
    def create_attributes(self, attributes):
        " Extract the phonological attribute and associate them with the arcs "
        
        unique = set([])
        self.attributes = vector[vector[int]](len(self.int2feat)+1)
        cdef int i
        for i in xrange(len(self.int2feat)+1):
            self.attributes[i] = vector[int]()

        for k, lst in attributes:
            for att in lst:
                unique.add(att)
                self.attributes[k].push_back(att)

        self.num_features = len(unique)
        self.weights = zeros((self.num_features))

    cpdef _get_attributes(self, int i):
        " gets the attributes for a specific arc "
        return self.attributes[i]


    def create_features(self):
        " Extract the atomic features "

        self.features = vector[pair[int, vector[int]]]()
        self.feature2origin = vector[int]()
        self.feature2origin.push_back(-1)

        counter = 1
        for state, arcs in self.ids.items():
            lst = []
            for i, (arc, tup) in enumerate(arcs.items()):
                action, value, ngram = tup
                lst.append(counter)
                self.feature2origin.push_back(state)
                self.int2feat[counter] = tup
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
            ll += self.lll(x, y)

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
            
                val = (ll1 - ll2) / (2.0 * EPS)
                g[i] += val

        return g


    def feature_grad_fd(self, data, EPS=0.1):
        " gradient for the locally normalized models with a finite-difference "
    
        feat_g = zeros((self.num_features))
        for x, y in data:
            for i in xrange(self.num_features):
                self.weights[i] += EPS
                self._feature_local_renormalize(self.weights)
                ll1 = self.lll(x, y)
                self.weights[i] -= 2 * EPS
                self._feature_local_renormalize(self.weights)
                ll2 = self.lll(x, y)
                self.weights[i] += EPS
                self._feature_local_renormalize(self.weights)
                val = (ll1 - ll2) / (2.0 * EPS)
                feat_g[i] += val

        return feat_g


    def grad(self, data):
        " gradient for locally normalized models "
        self._local_renormalize(self.theta)
        cdef double[:] g = zeros((self.atoms))
        
        cdef fst.LogVectorFst _x
        cdef fst.LogVectorFst _y
        self.counts = map[int, double]()
        for i, (x, y) in enumerate(data):
            _x, _y = x, y
            self._observed(i, _x.fst, _y.fst, g)

        cdef libfst.LogVectorFst machine = self.machine.fst[0]
        cdef map[int, double].iterator mit
        cdef libfst.ArcIterator[libfst.LogVectorFst] *it

        cdef pair[int, double] p
        cdef int state_id, j, feat
        cdef double value, prob
        cdef vector[int] lst

        mit = self.counts.begin()
        while mit != self.counts.end():
            p = deref(mit)
            state_id = p.first
            value = p.second
            
            lst = self.features_map[state_id]
            it = new libfst.ArcIterator[libfst.LogVectorFst](machine, state_id)
            j = 0

            while not it.Done():
                arc = <libfst.LogArc*> &it.Value()
                feat = lst[j]
                if feat != 0.0:
                    prob = exp(-arc.weight.Value())
                    g[feat] += prob * value
                j += 1
                it.Next()

            del it
            inc(mit)

        return np.asarray(g)


    def feature_grad(self, data):
        " Computes the gradient in feature space "
        
        self._feature_local_renormalize(self.weights)

        cdef double[:] arc_grad = self.grad(data)
        cdef double[:] feat_grad = np.zeros((self.num_features))

        cdef int i, feat
        cdef vector[int] features
        cdef vector[int].iterator it
        
        # TODO: remove attribute / feature distinction .. it's a nightmare
        for i in xrange(self.attributes.size()):
            features = self.attributes[i]
            it = features.begin()
            while it != features.end():
                feat = deref(it)
                feat_grad[feat] += arc_grad[i]
                inc(it)

        return np.asarray(feat_grad)
        #return np.asarray(arc_grad)

        
    cdef void _observed(self, int i, libfst.LogVectorFst *x, libfst.LogVectorFst *y, double[:] g):
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

        libfst.ShortestDistance(composed[0], &alphas, False)
        libfst.ShortestDistance(composed[0], &betas, True)

        cdef double logZ = betas[0].Value()
        cdef double score, weight
        cdef int j, state_id, feat, origin

        cdef vector[int] lst
        cdef vector[pair[int, vector[int]]] features = self.data_features[i]
        cdef vector[pair[int, vector[int]]].iterator vit
        cdef libfst.ArcIterator[libfst.LogVectorFst] *it

        cdef pair[int, vector[int]] p
        vit = features.begin()
        
        while vit != features.end():
            p = deref(vit)
            state_id = p.first
            lst = p.second
            j = 0
            
            it = new libfst.ArcIterator[libfst.LogVectorFst](composed[0], state_id)
            while not it.Done():
                arc = <libfst.LogArc*> &it.Value()
                feat = lst[j]
                if feat != 0.0:
                    score = exp(-alphas[state_id].Value() - betas[arc.nextstate].Value() - arc.weight.Value() + logZ)
                    g[feat] -= score
                    origin = self.feature2origin[feat]
                    if self.counts.count(origin) == 0:
                        self.counts[origin] = score
                    else:
                        self.counts[origin] += score

                j += 1
                it.Next()
            del it
            inc(vit)

        del middle
        del composed


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


    cdef void _feature_local_renormalize(self, double[:] weights):
        """
        Locally renormalize with the feaure weights
        """

        cdef double[:] theta = np.zeros((self.atoms))
        cdef int i, feat
        cdef vector[int] features
        cdef vector[int].iterator it

        for i in xrange(self.atoms):
            features = self.attributes[i]
            it = features.begin()
            while it != features.end():
                feat = deref(it)
                theta[i] += weights[feat]
                inc(it)

        self.theta = theta
        self._local_renormalize(self.theta)


    cdef void _local_renormalize(self, double[:] theta):
        """
        Locally renormalize the PFST. 

        Cython Note: This function should be completely white.
        """
        cdef int state_id, i, atom, index
        cdef list lst
        #cdef vector[int] lst

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

    property weights:
        def __get__(self):
            return self.weights
        def __set__(self, weights):
            self.weights = weights

    property machine:
        def __get__(self):
            return self.machine
        def __set__(self, machine):
            self.machine = machine
