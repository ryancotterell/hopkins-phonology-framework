import fst
import itertools as it
import numpy as np
from numpy import zeros, ones, exp, log
from collections import defaultdict as dd
from scipy.optimize import fmin_l_bfgs_b as lbfgs
from arsenal.alphabet import Alphabet
from pfst import PFST
from last_vowel import LastVowel
from utils import peek
import cProfile


class SED(PFST):
    """
    Stochastic Edit Distance
    """
    def __init__(self, alphabet, llc, urc, ulc, sigma=None, EOS="#"):
        super(SED, self).__init__(alphabet)

        self.llc = llc
        self.urc = urc
        self.ulc = ulc

        self.machine = None
        self.EOS = EOS

        if sigma is not None:
            self.sigma = sigma
        else:
            self.sigma = fst.SymbolTable()
            for symbol in self.alphabet:
                if symbol != self.EOS:
                    self.sigma[symbol] = len(self.sigma)
        
        # machines
        self.create_machine()
        self.create_features()
        self.feature_on_arcs()

        # weights
        self.theta = zeros(self.atoms)
        
    def _valid_rc(self, context):
        """
        Checks whether the passed in
        context is a valid right context
        """
        for c1, c2 in zip(list("^"+context), list(context)):
            if c1 == self.EOS and c2 != self.EOS:
                return False
        return True

    def _valid_lc(self, context):
        """
        Checks whether the passed in
        context is a valid left contetx
        """
        for c1, c2 in zip(list(self.EOS+context), list(context)):
            if c1 != self.EOS and c2 == self.EOS:
                return False
        return True


    def create_machine(self):
        """
        Creates the stochastic edit distance
        machine using the construction
        give in Cotterell et al. (2014)
        """
        self.machine = fst.LogVectorFst()
        self.machine.isyms = self.sigma
        self.machine.osyms = self.sigma
        self.ngram2state = Alphabet()

        END = "END"
        self.machine.add_state()
        self.machine[0].final = True
        self.ngram2state.add(END)
        # build-up states
        for i in xrange(self.urc):
            for urc in it.product(self.alphabet, repeat=i):
                urc = "".join(urc)
                if self._valid_rc(urc):
                    self.ngram2state.add((self.EOS*self.llc, self.EOS*self.ulc, "".join(urc)))
                    self.machine.add_state()
        self.machine.start = 1
        

        # full context
        for llc in it.product(self.alphabet, repeat=self.llc):
            llc = "".join(llc)
            if not self._valid_lc(llc):
                continue
            for ulc in it.product(self.alphabet, repeat=self.ulc):
                ulc = "".join(ulc)
                if not self._valid_lc(ulc):
                    continue
                for urc in it.product(self.alphabet, repeat=max(0, self.urc-1)):
                    urc = "".join(urc)
                    if not self._valid_rc(urc):
                        continue

                    if (llc, ulc, urc) not in self.ngram2state:
                        self.ngram2state.add((llc, ulc, urc))
                        self.machine.add_state()

                for urc in it.product(self.alphabet, repeat=self.urc):
                    urc = "".join(urc)
                    if not self._valid_rc(urc):
                        continue
                    
                    self.ngram2state.add((llc, ulc, urc))
                    self.machine.add_state()
                    if urc[0] == self.EOS:
                        self.machine.add_arc(self.ngram2state[(llc, ulc, urc)], 0, 0, 0, 0.0)
                        self.machine[self.ngram2state[(llc, ulc, urc)]].final = True

        # effectively an assertion statement
        self.ngram2state.freeze()
        self.ids = dd(dict)

        # create the machine
        
        for ngram in self.ngram2state:
            if ngram == END:
                continue
            llc, ulc, urc = ngram
            state1 = self.ngram2state[ngram]
            counter = 1
            if len(urc) < self.urc:
                for symbol in self.alphabet:
                    if symbol != self.EOS:
                        urc2 = urc+symbol
                        if not self._valid_rc(urc2):
                            continue

                        state2 = self.ngram2state[(llc, ulc, urc2)]
                        self.machine.add_arc(state1, state2, self.sigma.find(symbol), 0, 0.0)

                    else:
                        urc2 = urc+self.EOS
                        if not self._valid_rc(urc2):
                            continue

                        state2 = self.ngram2state[(llc, ulc, urc2)]
                        self.machine.add_arc(state1, state2, 0, 0, 0.0)

            else:
                for symbol in self.alphabet:
                    if symbol == self.EOS:
                        continue
                    
                    # terminate
                    if self.machine[state1].final != fst.LogWeight.ZERO:
                        self.ids[state1][0] = ('end', '', (llc, ulc, urc))
                        self.machine[state1].final = fst.LogWeight.ZERO

                    # insertion
                    llc2 = llc[1:]+symbol
                    # special case
                    if self.llc == 0:
                        llc2 = ''

                    state2 = self.ngram2state[(llc2, ulc, urc)]
                    self.machine.add_arc(state1, state2, 0, self.sigma.find(symbol), counter)
                    self.ids[state1][counter] = ('ins', symbol, (llc, ulc, urc))
                    counter += 1

                    if len(urc) > 0 and urc[0] != self.EOS:
                        # substitution
                        llc2 = llc[1:]+symbol
                        if self.llc == 0:
                            llc2 = ''

                        ulc2 = ulc[1:]+urc[0]
                        if self.ulc == 0:
                            ulc2 = ''
                        urc2 = urc[1:]
                        ngram2 = llc2, ulc2, urc2

                        state2 = self.ngram2state[(llc2, ulc2, urc2)]
                        self.machine.add_arc(state1, state2, 0, self.sigma.find(symbol), counter)
                        self.ids[state1][counter] = ('sub', symbol, (llc, ulc, urc))
                        counter +=1 

                # deletion
                if urc[0] != self.EOS:
                    ulc2 = ulc[1:]+urc[0]
                    if self.ulc == 0:
                        ulc2 = ''
                    urc2 = urc[1:]
                    state2 = self.ngram2state[(llc, ulc2, urc2)]
                    self.machine.add_arc(state1, state2, 0, 0, counter)
                    self.ids[state1][counter] = ('del', '', (llc, ulc, urc))
                    counter += 1





def main():
    letters = "a"#bcdefghijklmnopqrstuvwxyzAE"
    #letters = "a"
    sed = SED(["#"]+list(letters), 0, 1, 1)

    #lv = LastVowel(list(letters), list("aeAEiouy"))

    
    #x = fst.linear_chain("bAbabAbap", syms=lv.sigma, semiring="log")
    print np.asarray(sed.theta)

    sed.theta = np.zeros(sed.atoms) 
    sed.theta = np.random.rand(sed.atoms) 
    sed.local_renormalize()
    #print "ONE"
    sed.local_renormalize()
    x = fst.linear_chain("a", syms=sed.sigma, semiring="log")
    y = fst.linear_chain("a", syms=sed.sigma, semiring="log")
    data = [(x, y)]

    data = [ (x, y) for x, y in data ]
    sed.extract_features(data)
    
    #lv.extract_features([ y for x, y in data ])
    sed.local_renormalize()
    #lv.local_renormalize()

    # g_fd = sed.grad_fd(data, EPS=0.0001)
    # g = sed.grad(data)
    # print g
    # print g_fd
    # print np.allclose(g, g_fd, atol=0.01)
    # import sys; sys.exit(0)

    print "FOUR"
    import time
    start = time.time()
    sed.grad(data)
    end = time.time()
    print end - start
    print sed.decode([ x for x, y in data ], n=5)
    cProfile.runctx("sed.grad(data)", {'sed' : sed, 'data' : data }, {})

    #sed.train(data)
    #lv.train([ y for x, y in data ])

    #print sed.decode([ x for x, y in data ], n=5)
    
    
    """
    x = data[0][0]
    tmp = x >> sed.machine
    tmp.project_output()
    tmp.arc_sort_output()
    result = tmp >> lv.machine
    print len(result)
    peek(result, 10)
    """

    #cProfile.run('letters = "abcdefghijklmnopqrstuvwxyz"; sed = SED(["#"] + list(letters), 2, 2, 2); sed.create_machine()')

if __name__ == "__main__":
    main()
