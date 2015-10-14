import fst
import itertools as it
from arsenal.alphabet import Alphabet
import cProfile


class SED(object):
    """
    Stochastic Edit Distance
    """
    def __init__(self, alphabet, llc, urc, ulc, EOS="#"):
        self.alphabet = alphabet

        self.llc = llc
        self.urc = urc
        self.ulc = ulc

        self.machine = None
        self.EOS = EOS

        self.sigma = fst.SymbolTable()
        for symbol in self.alphabet:
            if symbol != self.EOS:
                self.sigma[symbol] = len(self.sigma)
        
        
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
        
        self.ngram2state = Alphabet()

        # build-up states
        for i in xrange(self.urc):
            for urc in it.product(self.alphabet, repeat=i):
                urc = "".join(urc)
                if self._valid_rc(urc):
                    self.ngram2state.add((self.EOS*self.ulc, self.EOS*self.llc, "".join(urc)))
                    self.machine.add_state()
        self.machine.start = 0

        # full context
        for llc in it.product(self.alphabet, repeat=self.llc):
            llc = "".join(llc)
            if not self._valid_lc(llc):
                continue
            for ulc in it.product(self.alphabet, repeat=self.ulc):
                ulc = "".join(ulc)
                if not self._valid_lc(ulc):
                    continue
                for urc in it.product(self.alphabet, repeat=self.urc-1):
                    urc = "".join(urc)
                    if not self._valid_rc(urc):
                        continue
                    self.ngram2state.add((llc, ulc, urc))
                    self.machine.add_state()

                for urc in it.product(self.alphabet, repeat=self.urc):
                    urc = "".join(urc)
                    if not self._valid_rc(urc):
                        continue
                    self.ngram2state.add((llc, ulc, urc))
                    self.machine.add_state()


        # great the machine
        for ngram in self.ngram2state:
            llc, ulc, urc = ngram
            state1 = self.ngram2state[ngram]
            if len(urc) < self.urc:
                for symbol in self.alphabet:
                    if symbol == self.EOS:
                        continue
                    urc2 = urc+symbol
                    if not self._valid_rc(urc2):
                        continue
                    try:
                        #print llc, ulc, urc2
                        state2 = self.ngram2state[(llc, ulc, urc2)]
                        self.machine.add_arc(state1, state2, self.sigma.find(symbol), 0, 0.0)
                    except KeyError:
                        print "ERROR"

            else:
                for symbol in self.alphabet:
                    if symbol == self.EOS:
                        continue
                    
                    # substitution
                    llc2 = llc[1:]+symbol
                    ulc2 = ulc[:1]+urc[0]
                    urc2 = urc[1:]
                    state2 = self.ngram2state[(llc2, ulc2, urc2)]
                    self.machine.add_arc(state1, state2, 0, self.sigma.find(symbol), 0.0)

                    # insertion
                    llc2 = llc[1:]+symbol
                    state2 = self.ngram2state[(llc2, ulc, urc)]
                    self.machine.add_arc(state1, state2, 0, self.sigma.find(symbol), 0.0)

                # deletion
                ulc = ulc[1:]+urc[0]
                urc2 = urc[1:]
                state2 = self.ngram2state[(llc, ulc, urc2)]
                self.machine.add_arc(state1, state2, 0, 0, 0.0)
                
        print len(self.machine)

def main():
    letters = "a"#bcdefghijklmnopqrstuvwxyz"
    sed = SED(["#"] + list(letters), 1, 1, 1)
    sed.create_machine()
    #cProfile.run('letters = "abcdefghijklmnopqrstuvwxyz"; sed = SED(["#"] + list(letters), 0, 2, 2); sed.create_machine()')

if __name__ == "__main__":
    main()
