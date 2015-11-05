import codecs
import sys
import fst
import random
import numpy as np
import utils
from utils import peek

from words import TwoMorphemeWord
from words import TemplaticWord
from models import ConcatPhonologyModel
from models import TemplaticPhonologyModel
from sed import SED

import pstats, cProfile


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
               "past 2 masculine singular" in suffixes:# or \
               #"past 3 masculine singular" in suffixes:# or \
               #"past 3 feminine singular" in suffixes:
          
                tw = TemplaticWord(sr, root, pattern, prefixes, suffixes)
                data.append(tw)
                
            if pattern in ["PST1", "PST1"] and root in ["sm'", "fhm", "ktb", "lms", "ftH"]:
                tw = TemplaticWord(sr, root, pattern, prefixes, suffixes)
                data.append(tw)
                  
                

            # for suffix in suffixes:
            #     if "past" in suffix:
            #         tw = TemplaticWord(sr, root, pattern, prefixes, suffixes)
            #         data.append(tw)
            #         break


            #if len(suffixes) >= 0 and len(prefixes) == 0:
            #    tw = TemplaticWord(sr, root, pattern, prefixes, suffixes)
            #    data.append(tw)

    return data



words = read(sys.argv[1])[:]
for word in words:
    print word.sr

random.shuffle(words)
train = words[:20]
test = words[20:]
words = train

sigma = fst.SymbolTable()
sigma["#"] = 1
letters = set([])
for word in words:
    for c in list(word.sr):
        if c not in sigma:
            sigma[c] = len(sigma)
            letters.add(c)

print "...creating stochastic edit machine"
sed = SED(["#"]+list(letters), 2, 1, 0, sigma=sigma)
print "...done"

sed.weights[0] = 5
sed.feature_local_renormalize()

#phonology = utils.phonology_edit(sigma, .99)

# make the model

# train the model
# iterations of EM
for iteration in xrange(3):
    # E-step
    # TODO:
    # shoudln't have to rebuild the factor graph... 
    # some sort of state bug
    
    phonology = sed.machine
    model = TemplaticPhonologyModel(words, phonology, sigma)
    model.inference(5)

    # M-step
    data = model.training_data(n=5)
    for ur, sr in data:
        print "UR"
        peek(ur, 10)
        print "SR"
        peek(sr, 10)

    sed.extract_features(data)
    sed.local_renormalize()
    sed.feature_train(data, maxiter=100)
    for ur, sr in data:
        print "UR"
        peek(ur, 10)
        print "SR"
        peek(sr, 10)
    
    print sed.decode([x for x, y in data ])

#cProfile.runctx("model.inference(2)", globals(), locals(), "Profile.prof")
#s = pstats.Stats("Profile.prof")
#s.strip_dirs().sort_stats("time").print_stats()
