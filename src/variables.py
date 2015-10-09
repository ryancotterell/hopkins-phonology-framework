import variational_approximation as va
import fst
import numpy as np
import collections
from utils import peek

class Variable(object):
    """
    Variable in the factor graph
    """
    def __init__(self, _id, sigma, num_edges):
        self._id = _id
        self.sigma = sigma
        self.edges = [None] * num_edges

    def pass_message(self):
        """
        Passes the message along
        """
        self.edges[1].m_f = self.edges[0].m_v

    def __str__(self):
        return "VAR: " + self._id
        
    def __repr__(self):
        return str(self)


class Variable_Observed(Variable):
    """
    Observed variable in the graph
    """
    def __init__(self, _id, sigma, sr):
        self._id = _id
        self.sigma = sigma
        self.sr = sr
        self.edges = [None] 

    def __str__(self):
        return "VAR: " + self._id
        
    def __repr__(self):
        return str(self)


class Variable_EP(Variable):
    """
    Variable with EP approximation
    during message computation
    
    """
    def __init__(self, _id, sigma, num_edges):
        self._id = _id
        self.sigma = sigma
        self.edges = [None] + [None] * num_edges

    def add_edge(self,edge):
        """ 
        Adds edges to the variable
        in the factor graph
        """
        self.edges.append(edge)

    def compute_messages(self, marginal=False, verbose=False, prune_n=100, tolerance=5.0):
        """
        Compute messages
        """
        pass

    def pass_message(self):
        """
        """
        # initialize the message to exponential unigram
        # this ensures normalizability
        belief = fst.LogVectorFst(self.sigma, self.sigma)
        belief.isyms = self.sigma
        belief.osyms = self.sigma
        belief.start = belief.add_state()
        belief[0].final = fst.LogWeight.ONE
        for k, v in self.sigma.items():
            belief.add_arc(0, 0, v, v, 2.8)        
            
        # extract all 4-grams from top 5 strings

        contexts = set([])

        for edge in self.edges:
            for path in fst.StdVectorFst(edge.m_v).shortest_path(n=1).paths():
                string = ""
                for arc in path:
                    if arc.ilabel > 0:
                        string += self.sigma.find(arc.ilabel)
                # unigrams
                for c in list(string):
                    contexts.add(c)
                # bigrams
                for c1, c2 in zip(list("^" + string), list(string)):
                    contexts.add(c1+c2)
                if len(self.edges) < 5:
                    # trigrams
                    for (c1, c2), c3 in zip(zip(list("^" + string), list(string)), list(string)):
                        contexts.add(c1+c2+c3)

        contexts = list(contexts)

        print "START EP", len(self.edges)
        # update to hold one out like EP should
        
        for i in xrange(1):
            for edge in self.edges[:]:

                #if len(self.edges) == 3:
                    
                    #print contexts
                    #print "EDGE M_V", edge.f
                    #peek(edge.m_v, 10)

                tmp = belief >> edge.m_v #fst.LogVectorFst(fst.StdVectorFst(edge.m_v).shortest_path(n=10))
                approx = None
                #if len(self.edges) > 3:
                #    approx = va.bigram(tmp)
                #else:
                approx = va.var(tmp)
                approx.create_machine(contexts)
                
                approx.q.isyms = self.sigma
                approx.q.osyms = self.sigma
                
                approx.estimate()
                belief = approx.q

        for edge in self.edges[:]:
            edge.m_f = belief
            
        print "BELIEF"
        peek(belief, 10)

    def __str__(self):
        return "VAR: " + self._id
        
    def __repr__(self):
        return str(self)

        
