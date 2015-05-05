#cython: wraparound=False
#cython: boundscheck=False
#cython: nonecheck=False
#cython: profile=False

import sys


cdef inline double logaddexp(double x, double y):
    """
    Needs to be rewritten
    """
    cdef double tmp = x - y
    if tmp > 0:
        return x + log1p(exp(-tmp))
    elif tmp <= 0:
        return y + log1p(exp(tmp))
    else:
        return x + y
  




cdef class approximation_machine:
    def __init__(self,p,sigma):
        """
        Assumes nested
        """
        self.p = p
        self.sigma = sigma


    def __dealloc__(self):
        pass

    def create_machine(self):
        pass

    def gradient(self):
        pass

    def xent(self):
        pass

    cdef double _xent(self):
        pass
  
    def extract_backpointers(self):
        self._extract_backpointers()

    cdef void _extract_backpointers(self):
        cdef int state_id = 0
        cdef int backpointer = 1
        cdef libfst.ArcIterator[libfst.LogVectorFst]* it
        cdef libfst.LogArc *arc
        cdef libfst.LogWeight *weight

        for state_id in range(self.q.fst.NumStates()):
            it = new libfst.ArcIterator[libfst.LogVectorFst](self.q.fst[0], state_id)

            while not it.Done():
                arc = <libfst.LogArc*> &it.Value()
                weight = new libfst.LogWeight(backpointer)
                arc[0].weight = weight[0]
                backpointer += 1
                it.Next()
                del weight
            del it

        cdef libfst.LogVectorFst *result = new libfst.LogVectorFst()
       
        
        result.SetInputSymbols(self.sigma.table)
        result.SetOutputSymbols(self.sigma.table)
        libfst.ArcMap(self.p.fst[0], result, libfst.RmLogWeightMapper())
      
        for state_id in range(self.p.fst.NumStates()):

            if self.p.fst.Final(state_id).Value() == libfst.LogWeightZero().Value():
                result.SetFinal(state_id,libfst.LogWeightZero())

        
        cdef libfst.LogVectorFst *composed = new libfst.LogVectorFst()
        composed.SetInputSymbols(self.sigma.table)
        composed.SetOutputSymbols(self.sigma.table)
        
        cdef libfst.ILabelCompare[libfst.LogArc] icomp
        libfst.ArcSort(self.q.fst, icomp)

        libfst.Compose(result[0], self.q.fst[0], composed)
        
      
        # back pointers
        cdef int arc_counter = 0
        del self.arc2backpointer
        self.arc2backpointer = new map[int,int]()

        state_id = 0
        for state_id in range(composed.NumStates()):
            it = new libfst.ArcIterator[libfst.LogVectorFst](composed[0], state_id)

            while not it.Done():
                arc = <libfst.LogArc*> &it.Value()
                backpointer = <int> floor(arc[0].weight.Value() + 0.5)
                self.arc2backpointer[0][arc_counter] = backpointer
                arc_counter += 1
                it.Next()
                
            del it

        del composed
        del result

    cdef void _to_ones(self):
        cdef int state_id = 0
        cdef libfst.ArcIterator[libfst.LogVectorFst]* it
        cdef libfst.LogArc *arc
        cdef libfst.LogWeight *weight

        for state_id in range(self.q.fst.NumStates()):
            it = new libfst.ArcIterator[libfst.LogVectorFst](self.q.fst[0], state_id)

            while not it.Done():
                arc = <libfst.LogArc*> &it.Value()
                weight = new libfst.LogWeight(libfst.LogWeightOne())
                arc[0].weight = weight[0]
                it.Next()
                del weight
            del it

    cdef void _to_zeros(self):
        cdef int state_id = 0
        cdef libfst.ArcIterator[libfst.LogVectorFst]* it
        cdef libfst.LogArc *arc
        cdef libfst.LogWeight *weight

        for state_id in range(self.q.fst.NumStates()):
            it = new libfst.ArcIterator[libfst.LogVectorFst](self.q.fst[0], state_id)

            while not it.Done():
                arc = <libfst.LogArc*> &it.Value()
                weight = new libfst.LogWeight(libfst.LogWeightZero())
                arc[0].weight = weight[0]
                it.Next()
                
            del it

    
    def estimate(self,smoothing=float("-inf"),backpointers=True):
        if backpointers:
            self._estimate_backpointers(smoothing)

        self.q.isyms = self.p.isyms
        self.q.osyms = self.p.osyms

    cdef void _estimate_backpointers(self,double smoothing):
        self._extract_backpointers()
        self._extract_backpointers()
        self._to_ones()
        cdef libfst.LogVectorFst *composed = new libfst.LogVectorFst()
        composed.SetInputSymbols(self.sigma.table)
        composed.SetOutputSymbols(self.sigma.table)
        
        cdef libfst.ILabelCompare[libfst.LogArc] icomp
        libfst.ArcSort(self.q.fst, icomp)

        libfst.Compose(self.p.fst[0], self.q.fst[0], composed)
        
        self._to_zeros()

        cdef int state_id,state_id1,state_id2
        cdef int arc_counter = 0
        cdef int backpointer
        cdef double new_value = 0.0

        cdef libfst.ArcIterator[libfst.LogVectorFst]* it
        cdef libfst.LogWeight *weight

        cdef vector[libfst.LogWeight] alphas
        cdef vector[libfst.LogWeight] betas
        libfst.ShortestDistance(composed[0], &alphas, False)
        libfst.ShortestDistance(composed[0], &betas, True)

        cdef map[int,double] backpointer2value = map[int,double]()
        for state_id1 in range(composed.NumStates()):
            it = new libfst.ArcIterator[libfst.LogVectorFst](composed[0], state_id1)

            while not it.Done():
                arc = <libfst.LogArc*> &it.Value()
                state_id2 = arc.nextstate
                backpointer = self.arc2backpointer[0][arc_counter]
            
                if backpointer2value.count(backpointer) == 0:
                    backpointer2value[backpointer] = -(alphas[state_id1].Value() + betas[state_id2].Value() + arc[0].weight.Value()) 
                else:
                    backpointer2value[backpointer] = logaddexp(backpointer2value[backpointer],-(alphas[state_id1].Value() + betas[state_id2].Value() + arc[0].weight.Value()))
                arc_counter += 1
                it.Next()
            
            del it
    
        arc_counter = 0
        cdef int tmp_arc_counter
        cdef double normalizer

        for state_id in range(self.q.fst[0].NumStates()):
            normalizer = -libfst.LogWeightZero().Value()

            it = new libfst.ArcIterator[libfst.LogVectorFst](self.q.fst[0], state_id)
            tmp_arc_counter = arc_counter
            while not it.Done():
                arc = <libfst.LogArc*> &it.Value()

                if backpointer2value.count(arc_counter+1) > 0:
                    normalizer = logaddexp(normalizer,logaddexp(backpointer2value[arc_counter+1],smoothing))
                arc_counter += 1
                it.Next()
            del it

            it = new libfst.ArcIterator[libfst.LogVectorFst](self.q.fst[0], state_id)
            while not it.Done():
                arc = <libfst.LogArc*> &it.Value()

                arc = <libfst.LogArc*> &it.Value()
                if backpointer2value.count(tmp_arc_counter+1) > 0:
                    weight = new libfst.LogWeight(-(logaddexp(backpointer2value[tmp_arc_counter+1],smoothing) - normalizer))
                    arc[0].weight = weight[0]
                else:
                    weight = new libfst.LogWeight(libfst.LogWeightZero())
                    arc[0].weight = weight[0]
                
                tmp_arc_counter += 1
                it.Next()
                del weight
            del it


        del self.arc2backpointer

    property p:
        def __get__(self):
            return self.p
    property q:
        def __get__(self):
            return self.q

cdef class unigram(approximation_machine):
    def __init__(self,p):
        super(unigram,self).__init__(p,p.isyms)
        self._create_machine()

    #def create_machine(self):
    #    self._create_machine()

    cdef void _create_machine(self):
        self.q = fst.LogVectorFst()

        del self.state2prefix
        del self.prefix2state

        self.state2prefix = new map[int,string]()
        self.prefix2state = new map[string,int]()
        
        self.q.fst.AddState()
        self.state2prefix[0][0] = b''
        self.prefix2state[0][b''] = 0
        self.q.start = 0

        self.q.fst.AddState()
        self.state2prefix[0][1] = b'<EOS>'
        self.prefix2state[0][b'<EOS>'] = 1
        
        cdef long key
        cdef int state_counter = 2
        cdef string prefix
        cdef libfst.LogArc *arc
        cdef libfst.LogWeight *weight

        weight = new libfst.LogWeight(libfst.LogWeightOne())
        self.q.fst.SetFinal(1,weight[0])

        # from start
        for key in range(1,self.sigma.table.NumSymbols()):
            weight = new libfst.LogWeight(libfst.LogWeightOne())
            arc = new libfst.LogArc(key, key, weight[0], 0)
            self.q.fst.AddArc(0, arc[0])
            del weight
            del arc
        # add final
        weight = new libfst.LogWeight(libfst.LogWeightOne())
        arc = new libfst.LogArc(0, 0, weight[0], 1)
        self.q.fst.AddArc(0, arc[0])

        del weight
        del arc
        del self.state2prefix
        del self.prefix2state



cdef class bigram(approximation_machine):    
    def __init__(self,p):
        super(bigram,self).__init__(p,p.isyms)
        self._create_machine()

    cdef void _create_machine(self):
        self.q = fst.LogVectorFst()

        del self.state2prefix
        del self.prefix2state

        self.state2prefix = new map[int,string]()
        self.prefix2state = new map[string,int]()
        
        self.q.fst.AddState()
        self.state2prefix[0][0] = b'<BOS>'
        self.prefix2state[0][b'<BOS>'] = 0
        self.q.start = 0

        self.q.fst.AddState()
        self.state2prefix[0][1] = b'<EOS>'
        self.prefix2state[0][b'<EOS>'] = 1
        
        cdef long key,key1,key2
        cdef int state_counter = 2
        cdef string prefix
        cdef libfst.LogArc *arc
        cdef libfst.LogWeight *weight

        weight = new libfst.LogWeight(libfst.LogWeightOne())
        self.q.fst.SetFinal(1,weight[0])

        # add states
        for key in range(1,self.sigma.table.NumSymbols()):
            prefix = self.sigma.table.Find(key)
            self.q.fst.AddState()
            self.state2prefix[0][state_counter] = prefix
            self.prefix2state[0][prefix] = state_counter
            state_counter += 1

        # add arcs
        for key1 in range(1,self.sigma.table.NumSymbols()):
            for key2 in range(1,self.sigma.table.NumSymbols()):
                weight = new libfst.LogWeight(libfst.LogWeightOne())
                arc = new libfst.LogArc(key2, key2, weight[0], key2 + 1)
                self.q.fst.AddArc(key1 + 1, arc[0])
            
                del weight
                del arc
            # add final
            weight = new libfst.LogWeight(libfst.LogWeightOne())
            arc = new libfst.LogArc(0, 0, weight[0], 1)
            self.q.fst.AddArc(key1 + 1, arc[0])

            del weight
            del arc

        # from start
        for key2 in range(1,self.sigma.table.NumSymbols()):
            weight = new libfst.LogWeight(libfst.LogWeightOne())
            arc = new libfst.LogArc(key2, key2, weight[0], key2 + 1)
            self.q.fst.AddArc(0, arc[0])
            
            del weight
            del arc
        # add final
        weight = new libfst.LogWeight(libfst.LogWeightOne())
        arc = new libfst.LogArc(0, 0, weight[0], 1)
        self.q.fst.AddArc(0, arc[0])


        del weight
        del arc
        del self.state2prefix
        del self.prefix2state


cdef class trigram(approximation_machine):    
    def __init__(self,p):
        super(trigram,self).__init__(p,p.isyms)
        self._create_machine()

    cdef void _create_machine(self):
        self.q = fst.LogVectorFst()

        del self.state2prefix
        del self.prefix2state

        self.state2prefix = new map[int,string]()
        self.prefix2state = new map[string,int]()
        
        self.q.fst.AddState()
        self.state2prefix[0][0] = b'<BOS>'
        self.prefix2state[0][b'<BOS>'] = 0
        self.q.start = 0

        self.q.fst.AddState()
        self.state2prefix[0][1] = b'<EOS>'
        self.prefix2state[0][b'<EOS>'] = 1
        
        cdef long key,key1,key2,key3
        cdef int state_counter = 2
        cdef string prefix, prefix1, prefix2, prefix3, prefix_dest
        
        cdef libfst.LogArc *arc
        cdef libfst.LogWeight *weight

        weight = new libfst.LogWeight(libfst.LogWeightOne())
        self.q.fst.SetFinal(1,weight[0])
        del weight

        # add states
        for key1 in range(1,self.sigma.table.NumSymbols()):
            prefix1 = self.sigma.table.Find(key1)
            self.q.fst.AddState()
            self.state2prefix[0][state_counter] = prefix1
            self.prefix2state[0][prefix1] = state_counter
            state_counter += 1

            for key2 in range(1,self.sigma.table.NumSymbols()): 
                prefix2 = self.sigma.table.Find(key2)
                prefix = prefix1 + prefix2
                self.q.fst.AddState()
                self.state2prefix[0][state_counter] = prefix
                self.prefix2state[0][prefix] = state_counter
                state_counter += 1

        # add arcs
        for key1 in range(1,self.sigma.table.NumSymbols()):

            # add final
            prefix1 = self.sigma.table.Find(key1)
            weight = new libfst.LogWeight(libfst.LogWeightOne())
            arc = new libfst.LogArc(0, 0, weight[0], 1)
            self.q.fst.AddArc(self.prefix2state[0][prefix1], arc[0])
            
            del weight
            del arc

            for key2 in range(1,self.sigma.table.NumSymbols()):
                prefix2 = self.sigma.table.Find(key2)
                prefix = prefix1 + prefix2

                # add final
                weight = new libfst.LogWeight(libfst.LogWeightOne())
                arc = new libfst.LogArc(0, 0, weight[0], 1)
                self.q.fst.AddArc(self.prefix2state[0][prefix], arc[0])
                
                del weight
                del arc
                
                # add start
                weight = new libfst.LogWeight(libfst.LogWeightOne())
                arc = new libfst.LogArc(key2, key2, weight[0], self.prefix2state[0][prefix])
                self.q.fst.AddArc(self.prefix2state[0][prefix1], arc[0])
                
                del weight
                del arc

                for key3 in range(1,self.sigma.table.NumSymbols()):
                    prefix3 = self.sigma.table.Find(key3)
                    prefix_dest = prefix2 + prefix3
                    
                    weight = new libfst.LogWeight(libfst.LogWeightOne())
                    arc = new libfst.LogArc(key3, key3, weight[0], self.prefix2state[0][prefix_dest])
                    self.q.fst.AddArc(self.prefix2state[0][prefix], arc[0])
                   
                    del weight
                    del arc

        # from start
        for key2 in range(1,self.sigma.table.NumSymbols()):
            prefix_dest = self.sigma.table.Find(key2)
            weight = new libfst.LogWeight(libfst.LogWeightOne())
            arc = new libfst.LogArc(key2, key2, weight[0], self.prefix2state[0][prefix_dest])
            self.q.fst.AddArc(0, arc[0])
            
            del weight
            del arc
        # add start to final for epsilon
        weight = new libfst.LogWeight(libfst.LogWeightOne())
        arc = new libfst.LogArc(0, 0, weight[0], 1)
        self.q.fst.AddArc(0, arc[0])

        del weight
        del arc

        del self.state2prefix
        del self.prefix2state


cdef class vo(approximation_machine):
    def __init___(self,p,ngrams):
        print p
        print ngrams
        #super(vo,self).__init__(p,p.isyms)
        #self._create_machine(ngrams)
        
    def create_machine(self,ngrams):
        self._create_machine(ngrams)

    cdef void _create_machine(self,vector[string] ngrams):
        self.q = fst.LogVectorFst()

        del self.state2prefix
        del self.prefix2state

        self.state2prefix = new map[int,string]()
        self.prefix2state = new map[string,int]()
        
        self.q.fst.AddState()
        self.state2prefix[0][0] = b'<BOS>'
        self.prefix2state[0][b'<BOS>'] = 0
        self.q.start = 0

        self.q.fst.AddState()
        self.state2prefix[0][1] = b'<EOS>'
        self.prefix2state[0][b'<EOS>'] = 1
        
        cdef long key,key1,key2
        cdef int state_counter = 2
        cdef string prefix
        cdef libfst.LogArc *arc
        cdef libfst.LogWeight *weight
        
        weight = new libfst.LogWeight(libfst.LogWeightOne())
        self.q.fst.SetFinal(1,weight[0])
        del weight
        # add states
        cdef int index = 0
        for index in range(ngrams.size()):
            prefix = ngrams[index].substr(0,ngrams[index].size()-1)
            if self.prefix2state.count(prefix) == 0:
                self.q.fst.AddState()
                self.prefix2state[0][prefix] = state_counter
                self.state2prefifx[0][state_counter] = prefix
                state_counter += 1
                


            
        # add arcs
        cdef int state_id = 0
        for state_id in range(self.q.fst.NumStates()):
            if state_id == 1:
                continue

            prefix = string(self.state2prefix[0][state_id])
            
            print(state_id)
            print(prefix)
            print("")
            




