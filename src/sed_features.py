from arsenal.alphabet import Alphabet

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
        self.features = Alphabet()
        self.features.add("COPY")

        self.attributes = {}
        for symbol in symbols:
            self.attributes[symbol] = [symbol]

        if attrs_in is not None:
            # READ in the FILE "
            pass

            
    def _copy(self, action, char, llc, ulc, urc):
        " is a copy action ? "

        if action == 'sub' and len(urc) > 0 and char == urc[-1]:
            return [ 0 ]
        return [ ]

    
    def _bigram(self, action, char, llc, ulc, urc):
        " well-formedness features "

        features = []
        for i in xrange(len(llc)):
            prefix = llc[:i+1]
            string = "BIGRAM(" + prefix + char + ")"
            features.append(self.features[string])

        return features


    def _edit(self, action, char, llc, ulc, urc):
        " faithfulness features "

        features = []
        if len(urc) > 0:
            if action in ['sub', 'del']:
                string = "EDIT(" + urc[-1] + "," + char + ")"
                self.features.add(string)
                features.append(self.features[string])
            else:
                string = "EDIT(," + char + ")"
                self.features.add(string)
                features.append(self.features[string])

        return features

        
    def _attrs(self, symbol):
        " extract the attributes for a given symbol "
        pass

    def extract(self, state):
        """
        Extract the features
        """
        action, char, (llc, ulc, urc) = state
        extractors = [self._copy, self._edit, self._bigram]

        features = []
        for extractor in extractors:
            features.extend(extractor(action, char, llc, ulc, urc))

        return features
