"""
Evaluates the model based on three evaluation metrics:
1) Cross-Entropy
2) Accuracy
3) Expected Edit Distance
"""
import fst
import numpy as np
import numpy.random as npr
from numpy import exp
from Levenshtein import distance

class Evaluator(object):
    """ STUB """
    pass

class TemplaticEvaluator(Evaluator):
    """
    Implementation of the evaluator
    """
    def __init__(self, model, dev, test):
        self.model = model
        self.dev = dev
        self.test = test


    def evaluate(self, dev=True):
        " Run the evaluation metrics "
        if dev:
            return self.aggregate(self.dev)
        else:
            return self.aggregate(self.test)


    def aggregate(self, words):
        " Aggregate over data "
        N = 0
        xent, acc, exp_edit = 0.0, 0.0, 0.0
        for word in words:
            pattern = word.pattern_id
            root = word.root_id
            suffixes = word.suffix_ids
            prefixes = word.prefix_ids
            # TODO: doesn't support multiple prefixes and suffixes
            suffix = suffixes[0] if len(suffixes) == 1 else None
            prefix = prefixes[0] if len(prefixes) == 1 else None

            # predict the word form
            prediction = self.model.predict(root, pattern, prefix, suffix)

            if prediction is not None:
                # metrics
                xent += self.cross_entropy(word.sr, prediction)
                acc += self.accuracy(word.sr, prediction)
                exp_edit += self.expected_edit_distance(word.sr, prediction)
                N += 1

        return xent / N, acc / N, exp_edit / N


    def cross_entropy(self, truth, prediction):
        " Computes cross entropy "
        
        machine = fst.linear_chain(truth, semiring="log", syms=prediction.isyms)
        result = machine >> prediction
        return float(result.shortest_distance(True)[result.start])


    def accuracy(self, truth, prediction):
        " Computes accuracy "
        
        prediction = fst.StdVectorFst(prediction)
        machine = prediction.shortest_path(n=1)
        machine.project_output()

        string = ""
        for path in machine.paths():
            for arc in path:
                if arc.ilabel > 0:
                    string += machine.isyms.find(arc.ilabel)

        print string, truth
        return 0.0 if string == truth else 1.0
        

    def expected_edit_distance(self, truth, prediction, n=50):
        " Computes the expected edit distance "
        prediction.project_output()
        prediction = prediction.push_weights()
        
        # sample paths
        def sample():        
            string = ""
            cur = prediction[prediction.start]
            while True:
                p = []
                p.append(exp(-float(cur.final)))
                for arc in cur:
                    p.append(exp(-float(arc.weight)))
                p = np.array(p)
                p /= p.sum()

                sampled = npr.choice(range(len(p)), p=p)
                if sampled == 0:
                    break
                else:
                    for counter, arc in enumerate(cur):
                        if counter+1 == sampled:
                            if arc.ilabel > 0:
                                string += prediction.isyms.find(arc.ilabel)
                            cur = prediction[arc.nextstate]

                # HACK: ensure finite termination
                if len(string) > 20:
                    return string

            return string

        edit_distance = 0.0
        for i in xrange(n):
            sampled = unicode(sample())
            edit_distance += distance(unicode(truth), sampled)
        return edit_distance / n
