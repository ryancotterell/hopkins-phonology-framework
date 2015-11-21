
class TemplaticWord(object):
    def __init__(self, sr, root_id, pattern_id, prefix_ids, suffix_ids):
        self.sr = sr
        self.root_id = root_id
        self.pattern_id = pattern_id
        self.main_id = root_id + " + " + pattern_id
        self.ur_id = " + ".join(prefix_ids) + self.main_id + " + " + " + ".join(suffix_ids)

        self.suffix_ids, self.prefix_ids = [], []
        for prefix_id in prefix_ids:
            self.prefix_ids.append(prefix_id)
        for suffix_id in suffix_ids:
            self.suffix_ids.append(suffix_id)


class TwoMorphemeWord(object):
    """
    Immutable
    """ 
    def __init__(self,sr,morpheme1,morpheme2):
        self.sr = sr
        self.morpheme1 = morpheme1
        self.morpheme2 = morpheme2

        
