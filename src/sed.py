import fst
import itertools as it
import numpy as np
from numpy import zeros, ones, exp, log
from collections import defaultdict as dd
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
        
        # features
        self.atoms = -1
        self.features = []

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

        # build-up states
        for i in xrange(self.urc):
            for urc in it.product(self.alphabet, repeat=i):
                urc = "".join(urc)
                if self._valid_rc(urc):
                    self.ngram2state.add((self.EOS*self.llc, self.EOS*self.ulc, "".join(urc)))
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

        # for k, v in self.ngram2state.items():
        #     print k, v

        # for state_i, state in enumerate(self.machine):
        #     for arc in state:
        #         print state_i, arc.weight

        # for state, arcs in self.ids.items():
        #     for arc, tup in arcs.items():
        #         action, value, ngram = tup
        #         print state, arc, action, value, ngram

                
    def create_features(self):
        " Extract the atomic features "
        self.features = []
        counter = 1
        for state, arcs in self.ids.items():
            lst = []
            for i, (arc, tup) in enumerate(arcs.items()):
                if i == 0 and arc != 0:
                    lst.append(-1)
                action, value, ngram = tup
                lst.append(counter)
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

    def copy(self, f):
        " Copy a transducer "

        if not isinstance(f, fst.LogVectorFst):
            raise("Requires Log-Transducer")
        
        g = fst.LogVectorFst()
        for _ in xrange(len(f)):
            g.add_state()

        for i, state in enumerate(f):
            g[i].final = state.final
            for j, arc in enumerate(state):
                g.add_arc(i, arc.nextstate, arc.ilabel, arc.olabel, arc.weight)
            
        return g
        
    def to_zeros(self, f):
        " Zero out the arcs "

        for state in f:
            if state.final != fst.LogWeight.ZERO:
                state.final = fst.LogWeight.ZERO
            for arc in state:
                arc.weight = fst.LogWeight.ZERO

    def extract_features(self, data):
        " Extract the features on the arcs "
        
        self.feature_on_arcs()
        for x, y in data:
            x_prime = self.copy(x)
            y_prime = self.copy(y)
            self.to_zeros(x_prime)

            for state in x:
                for arc in state:
                    print arc
            

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

    def lll(self, x, y):
        " examplar log-likelihood "
        result = x >> self.machine >> y
        return -float(result.shortest_distance(True)[0])
        
    def ll(self, data):
        " log-likelihood for locally normalized models "
    
        ll = 0.0
        for x, y in data:
            # TODO: fix this!
            ll = log(exp(ll)+exp(self.lll(x, y)))
        return ll

    def grad_fd(self, data, EPS=0.001):
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

    def grad(self, data):
        " gradient for locally normalized models "
        pass
        """
        for x, y in data:
            result = x >> self.machine >> y
            alphas = result.shortest_distance()
            betas = result.shortest_distance(True)
            print [ exp(-float(x)) for x in alphas ]
            print [ exp(-float(x)) for x in betas ]
            print exp(-float(self.machine[1].final))
            print betas

            for state in self.machine:
                print "STATE", state
                for arc in state:
                    print "ARC", arc


            one = exp(-float(alphas[2] * betas[2])) * 1.0 / 3
            two = exp(-float(alphas[2] * betas[2])) * 1.0 / 3
            print "ONE", one
            print "TWO", two
            print two - one
        """
        """
            for state_id, lst in self.features:
                state = self.machine[state_id]

                if lst[0] != -1:
                    # FINAL WEIGHT FIRST
                    pass
                    #state.final = fst.LogWeight(-self.theta[lst[0]]+Z)

        """

def main():
    letters = "a"#bcdefg"# defghijklmnopqrstuvwxyz"
    sed = SED(["#"]+list(letters), 0, 1, 0)
    #sed.theta = np.random.rand(sed.atoms)
    sed.local_renormalize()

    x = fst.linear_chain("a", syms=sed.sigma, semiring="log")
    y = fst.linear_chain("a", syms=sed.sigma, semiring="log")
    data = [(x, y)]

    sed.extract_features(data)

    #print sed.features
    #print sed.ll(data)
    #print sed.grad_fd(data)
    #sed.grad(data)
    """
    string = fst.linear_chain("aaaaa", syms=sed.sigma, semiring="log")
    result = string >> sed.machine
    result.project_output()
    for state in result:
        Z = 0.0
        Z += exp(-float(state.final))
        for arc in state:
            Z += exp(-float(arc.weight))
        print "Z", Z
    print exp(-float(result.shortest_distance(True)[0]))
    """

    """
    for state in sed.machine:
        print state, state.final
        for arc in state:
            print arc.weight
    """
    #cProfile.run('letters = "abcdefghijklmnopqrstuvwxyz"; sed = SED(["#"] + list(letters), 0, 2, 2); sed.create_machine()')

if __name__ == "__main__":
    main()
