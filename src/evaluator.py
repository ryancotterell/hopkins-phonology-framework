"""
Evaluates the model based on three evaluation metrics:
1) Cross-Entropy
2) Accuracy
3) Expected Edit Distance
"""

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
            
                N += 1

    def cross_entropy(self, truth, prediction):
        " Computes cross entropy "
        pass
        

    def accuracy(self, truth, prediction):
        " Computes accuracy "
        pass


    def expected_edit_distance(def, truth, prediction):
        " Computes the expected edit distance "
        pass
