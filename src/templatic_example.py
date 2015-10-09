import codecs
import sys
import fst
import numpy as np
import utils

from words import TwoMorphemeWord
from words import TemplaticWord
from models import ConcatPhonologyModel
from models import TemplaticPhonologyModel

def read(file_in):
    data = []
    with codecs.open(file_in, 'rb', encoding="utf8") as f:
        for line in f:
            line = line.strip()
            sr, morphemes = line.split("\t")
            av = morphemes.split(",")

            root, pattern = None, None
            prefixes, suffixes = [], []
            for item in av:
                a, v = item.split("=")

                if a == "ROOT":
                    root = v
                elif a == "PATTERN":
                    pattern = v
                elif a == "PREFIX":
                    prefixes.append(v)
                elif a == "SUFFIX":
                    suffixes.append(v)
            
            if "past 1 singular" in suffixes or \
               "past 2 masculine singular" in suffixes or \
               "past 3 masculine singular" in suffixes or \
               "past 3 feminine singular" in suffixes or \
               "past 2 masculine plural" in suffixes or \
               "past 2 feminine plural" in suffixes or \
               "past 1 plural" in suffixes or \
               "past 3 plural" in suffixes:
                tw = TemplaticWord(sr, root, pattern, prefixes, suffixes)
                data.append(tw)
    return data

words = read(sys.argv[1])[:100]
print words
model = TemplaticPhonologyModel(words)
model.inference(10)

