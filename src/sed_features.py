"""
Extract the features for a stochastic
edit distance model

"""
class SED_Features(object):
    """
    Feature extraction for stochastic
    contextual edit distance
    """
    def __init__(self, symbols, attrs_in=None):
        self.num_features = 0
        self.attributes = {}
        for symbol in symbols:
            self.attributes[symbol] = [symbol]

        if attrs_in is not None:
            # READ in the FILE "
            pass
            
    def _copy(self):
        " is a copy action ? "
        pass
    
    def _bigram(self, lower, llc):
        " well-formedness features "
        pass

    def _edit(self, upper, lower):
        " faithfulness features "
        pass

    def _attrs(self, symbol):
        " extract the attributes for a given symbol "
        pass

    def extract(self, action, lower, context):
        """
        Extract the features
        """
        llc, urc, ulc = context
        pass

    
