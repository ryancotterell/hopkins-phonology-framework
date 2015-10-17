import fst
import itertools as it
import numpy as np
from numpy import zeros, ones, exp, log
from collections import defaultdict as dd
from scipy.optimize import fmin_l_bfgs_b as lbfgs
from arsenal.alphabet import Alphabet
from pfst import PFST
import cProfile

class Skip(PFST):
    """
    Skip-Class Model 
    """
    def __init__(self, alphabet, remember, llc, urc, ulc, EOS="#"):
        super(Skip, self).__init__(alphabet)

        self.remember = remember
        self.remember_set = set(remember)

        self.llc = llc
        self.urc = urc
        self.ulc = ulc

        self.machine = None

        self.EOS = EOS
        self.sigma = fst.SymbolTable()
        for symbol in self.alphabet:
            if symbol != self.EOS:
                self.sigma[symbol] = len(self.sigma)

        self.create_machine()


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
        Creates a machine that remembers a certain class of
        letters. For instance, this will construct a machine that
        remembers the last sibilant or the last vowel
        """
        self.machine = fst.LogVectorFst()
        self.machine.isyms = self.sigma
        self.machine.osyms = self.sigma
        self.ngram2state = Alphabet()

        # build-up states
        for i in xrange(self.urc):
            for urc in it.product(self.remember, repeat=i):
                urc = "".join(urc)
                if self._valid_rc(urc):
                    self.ngram2state.add((self.EOS*self.llc, self.EOS*self.ulc, "".join(urc)))
                    self.machine.add_state()
        self.machine.start = 0

        # full context
        for llc in it.product(self.remember, repeat=self.llc):
            llc = "".join(llc)
            if not self._valid_lc(llc):
                continue
            for ulc in it.product(self.remember, repeat=self.ulc):
                ulc = "".join(ulc)
                if not self._valid_lc(ulc):
                    continue
                for urc in it.product(self.remember, repeat=max(0, self.urc-1)):
                    urc = "".join(urc)
                    if not self._valid_rc(urc):
                        continue

                    if (llc, ulc, urc) not in self.ngram2state:
                        self.ngram2state.add((llc, ulc, urc))
                        self.machine.add_state()

                for urc in it.product(self.remember, repeat=self.urc):
                    urc = "".join(urc)
                    if not self._valid_rc(urc):
                        continue
                    
                    self.ngram2state.add((llc, ulc, urc))
                    self.machine.add_state()
                    if urc[0] == self.EOS:
                        self.machine[self.ngram2state[(llc, ulc, urc)]].final = True

        # effectively an assertion statement
        self.ngram2state.freeze()
        self.ids = dd(dict)

        # create the machine
        for ngram in self.ngram2state:
            llc, ulc, urc = ngram
            state1 = self.ngram2state[ngram]
            counter = 1
            if len(urc) < self.urc:
                for symbol in self.alphabet:

                    if symbol != self.EOS:
                        urc2 = urc
                        if symbol in self.remember_set:
                            urc2 = urc[1:]+symbol
                            if not self._valid_rc(urc2):
                                continue
            
                        state2 = self.ngram2state[(llc, ulc, urc2)]
                        self.machine.add_arc(state1, state2, self.sigma.find(symbol), 0, 0.0)

                    """
                    else:
                        urc2 = urc+self.EOS
                        if not self._valid_rc(urc2):
                            continue

                        state2 = self.ngram2state[(llc, ulc, urc2)]
                        print "STATE", state1, state2, 0, 0, 0.0
                        self.machine.add_arc(state1, state2, 0, 0, 0.0)
                    """

            """
            else:
                for symbol in self.alphabet:
                    if symbol == self.EOS:
                        continue
                    
                    # terminate
                    if self.machine[state1].final != fst.LogWeight.ZERO:
                        self.ids[state1][0] = ('end', '', (llc, ulc, urc))

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

            """

def main():
    letters = "abcdefi"
    remember = "aei"
    skip = Skip(letters, remember, 2, 2, 2)


if __name__ == "__main__":
    main()


