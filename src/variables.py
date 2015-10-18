import variational_approximation as va
import fst
import numpy as np
from collections import defaultdict as dd
from utils import peek

class Variable(object):
    """
    Variable in the factor graph
    """
    def __init__(self, _id, sigma, num_edges):
        self._id = _id
        self.sigma = sigma
        self.edges = [None] * num_edges
        self.belief = None
        
    def pass_message(self):
        """
        Passes the message along
        """
        self.edges[1].m_f = self.edges[0].m_v

    def compute_belief(self):
        " Computes the belief "
        
        self.belief = self.edges[0].m_v
        for edge in self.edges[1:]:
            self.belief = self.belief >> edge.m_v


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
        self.value = fst.linear_chain(sr, semiring="log", syms=self.sigma)
        self.edges = [] 
        
    def pass_message(self, i):
        """ 
        Pass the message
        """
        for edge in self.edges:
            edge.m_f = self.value

    def __str__(self):
        return "VAR: " + self._id
        
    def __repr__(self):
        return str(self)


class Variable_Pruned(Variable):
    """
    Simple K-best pruning for a variable's belief
    computation. Useful for machines
    that we know in advance will be small, e.g., 
    the alignment variables. This is faster 
    when applicable. 
    """
    def __init__(self, _id, sigma, num_edges, k=100):
        self._id = _id
        self.sigma = sigma
        self.edges = [None] * num_edges
        self.k = k

    def pass_message(self):
        """
        Passes the message along
        """
        print "DONE"
        incoming = []
        for edge in self.edges:
            if edge == None:
                continue
            incoming.append(edge.m_v)
            
        belief = incoming[0]
        for m in incoming[1:]:
            belief = fst.LogVectorFst(fst.StdVectorFst(belief).shortest_path(n=self.k))
            belief = belief >> m
            #belief = belief.determinize()
            #belief.minimize()

        belief = fst.LogVectorFst(fst.StdVectorFst(belief).shortest_path(n=self.k))
        belief = belief.determinize()
        belief.minimize()
        print "BELIEF"
        peek(belief, 10)
    
        # faster to just pass the belief
        for edge in self.edges:
            if edge == None:
                continue
            edge.m_f = belief

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
        self.edges = [None] + [None]*num_edges

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
            belief.add_arc(0, 0, v, v, 3.0)        
            
        # extract all 4-grams from top 5 strings

        contexts = set([])
        for edge in self.edges:
            if edge == None:
                continue
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
                # if len(self.edges) < 5:
                #     # trigrams
                #     for (c1, c2), c3 in zip(zip(list("^^" + string), list("^" + string)), list(string)):
                #         contexts.add(c1+c2+c3)
                        
        contexts = list(contexts)

        print "START EP", len(self.edges)
        print "CONTEXTS", len(contexts)
        # update to hold one out like EP should
        
        for i in xrange(2):
            for edge in self.edges[:]:
                if edge == None:
                    continue
               
                #print "EDGE", edge, edge.m_v
                #if len(self.edges) == 3:
                    
                    #print contexts
                # print str(i) + "\tEDGE M_V", edge.v
                # peek(edge.m_v, 10)
                tmp = edge.m_v >> belief  #fst.LogVectorFst(fst.StdVectorFst(edge.m_v).shortest_path(n=10))
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

                """
                print "APPROX"
                try:
                    peek(approx.q, 10)
                except:
                    for state in belief:
                        print state, state.final
                        for arc in state:
                            print arc
                    import sys; sys.exit(0)
                """
                #print "B"
                #peek(belief, 10)
                #edge.m_f = belief

        #belief = belief >> self.edges[0].m_v

        print belief
        print "BELIEF 1"
        peek(belief, 10)

        for edge in self.edges[:]:
            if edge == None:
                continue
            edge.m_f = belief


    def __str__(self):
        return "VAR: " + self._id
        
    def __repr__(self):
        return str(self)

        

class Variable_EP2(Variable):
    """
    Variable with EP approximation
    during message computation
    
    """
    def __init__(self, _id, sigma, num_edges):
        self._id = _id
        self.sigma = sigma
        self.edges = []

    def compute_messages(self, marginal=False, verbose=False, prune_n=100, tolerance=5.0):
        """
        Compute messages
        """
        pass

    def compute_belief(self):
        """
        Computes the belief
        """
        belief = None
        for edge in self.edges:
            if belief is None:
                belief = edge.m_v
            else:
                belief.arc_sort_output()
                belief = belief >> edge.m_v

        return belief

    def pass_message(self, i):
        """
        """

        contexts = set([])
        for edge in self.edges:
            if edge == None:
                continue
            for path in fst.StdVectorFst(edge.m_v).shortest_path(n=2).paths():
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
                # trigrams
                for (c1, c2), c3 in zip(zip(list("^^" + string), list("^" + string)), list(string)):
                    trigram = c1+c2+c3
                    if trigram[0:2] == "^^":
                        trigram = trigram[1:]
                    contexts.add(c1+c2+c3)
        contexts = list(contexts)
        #print contexts

        message = None
        for j, edge in enumerate(self.edges):
            if i == j:
                continue
       
            
            #edge.m_v.write("message.fst", True, True)
            edge.m_v.remove_epsilon()
            #for state in edge.m_v:
            #    for arc in state:
            #        assert arc.ilabel != 0
            
            #edge.m_v.minimize()
            
            print "START ESTIMATING", len(edge.m_v), len(contexts)
            approx = va.var(edge.m_v)
            approx.create_machine(contexts)
            approx.q.isyms = self.sigma
            approx.q.osyms = self.sigma
            approx.estimate()
            print "DONE ESTIMATING"
            #raw_input()
            #print "INCOMING", edge
            #peek(edge.m_v, 10)
            
            if message is None:
                message = approx.q
            else:
                message.arc_sort_output()
                edge.m_v.arc_sort_input()
                message = message >> approx.q
            #print "HERE", self
            #peek(edge.m_v, 10)

 

        approx = va.var(message)
        approx.create_machine(contexts)
        approx.q.isyms = self.sigma
        approx.q.osyms = self.sigma
        approx.estimate()
        #print "HERE2"
        #print "MESSAGE"

        #peek(message, 10)
        self.edges[i].m_f = approx.q


    def __str__(self):
        return "VAR: " + self._id
        
    def __repr__(self):
        return str(self)

        
