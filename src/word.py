class TemplaticWord(object):
    """
    Templatic Word
    """

    def __init__(self, sr, root_id, pattern_id, suffix_id):
        self.sr = sr
        self.root_id = root_id
        self.pattern_id = pattern_id
        self.suffix_id = suffix_id
        self.main_id = root_id + " + " + pattern_id
        self.ur_id = self.main_id + " + " + self.suffix_id


class RedupWord(object):
    """
    Reduplication Wrd
    """
    def __init__(self, sr, main_id, prefix_id, suffix_id, prefix_redup=False, suffix_redup=False):
        self.sr = sr
        self.main_id = main_id
        self.prefix_id = prefix_id
        self.suffix_id = suffix_id
        self.prefix_redup = prefix_redup
        self.suffix_redup = suffix_redup
