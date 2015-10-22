# sandbox for reduplication

from random import shuffle
import numpy as np
import fst
from factors import ExponentialUnaryFactor
from factors import PhonologyFactor
from factors import TwoWayConcat
from factors import ThreeWayConcat
from factors import TwoWayTemplatic
from variables import Variable
from variables import Variable_EP
from variables import Variable_Observed
from variables import Variable_Pruned
from edges import Edge

class NWayConcat(object):
    """
    Model for n-way concatenative phonology
    """

    def __init__(self, words, phonology, sigma):
        self.phonology = phonology
        self.sigma = sigma

        # get number of unique morphemes
        # cache their integer ids
        self.morphemes_to_id = {}
        self.sr_to_id = {}
        self.id_to_morphemes = {}
        self.id_to_sr = {}

        self.id2main = {}
        self.main2id = {}

        self.morphemeid2mainid = {}
        self.surface_forms = {}

        morphemes_count = {}
        self.num_morphemes = {}
       
        self.morpheme_id_to_type = {}
        self.has_suffix = {}
        self.has_prefix = {}

        for word in words:
            # assumes surface alphabet is the
            # same as the underlying alphabet
            for c in list(word.sr):
                assert c in self.sigma

            # intern sr
            cur_len =  len(self.sr_to_id)
            self.sr_to_id[word.ur_id] = cur_len
            self.id_to_sr[cur_len] = word.ur_id
            self.surface_forms[cur_len] = word.sr
            
            self.num_morphemes[cur_len] = 1
            
            # intern the variables
            if word.main_id not in self.morphemes_to_id:
                self.morphemes_to_id[word.main_id] = len(self.morphemes_to_id)
                self.id_to_morphemes[len(self.morphemes_to_id)-1] = word.main_id
                morphemes_count[self.morphemes_to_id[word.main_id]] = 1
                self.morpheme_id_to_type[len(self.morphemes_to_id)-1] = 1
            else:
                morphemes_count[self.morphemes_to_id[word.main_id]] += 1

            for prefix_id in word.prefix_ids:
                if prefix_id not in self.morphemes_to_id:
                    self.morphemes_to_id[prefix_id] = len(self.morphemes_to_id)
                    self.id_to_morphemes[len(self.morphemes_to_id)-1] = prefix_id
                    morphemes_count[self.morphemes_to_id[prefix_id]] = 1
                    self.morpheme_id_to_type[len(self.morphemes_to_id)-1] = 0
                else:
                    morphemes_count[self.morphemes_to_id[prefix_id]] += 1
                self.num_morphemes[cur_len] += 1

            if len(word.prefix_ids) > 0:
                self.has_prefix[cur_len] = 1
            else:
                self.has_prefix[cur_len] = 0

            for suffix_id in word.suffix_ids:
                if suffix_id not in self.morphemes_to_id:
                    self.morphemes_to_id[suffix_id] = len(self.morphemes_to_id)
                    self.id_to_morphemes[len(self.morphemes_to_id)-1] = suffix_id
                    morphemes_count[self.morphemes_to_id[suffix_id]] = 1
                    self.morpheme_id_to_type[len(self.morphemes_to_id)-1] = 2
                else:
                    morphemes_count[self.morphemes_to_id[suffix_id]] += 1

                self.num_morphemes[cur_len] += 1

            if len(word.suffix_ids) > 0:
                self.has_suffix[cur_len] = 1
            else:
                self.has_suffix[cur_len] = 0

            if word.main_id not in self.main2id:
                self.main2id[word.main_id] = len(self.main2id)
                self.id2main[len(self.main2id)-1] = word.main_id
                self.morphemeid2mainid[self.morphemes_to_id[word.main_id]] = len(self.main2id) - 1

        # data structures
        self.level1_variables = np.ndarray((len(self.sr_to_id)), dtype='object')
        self.level2_variables = np.ndarray((len(self.sr_to_id)), dtype='object')
        self.phono_factors = np.ndarray((len(self.sr_to_id)), dtype='object')
        self.concat_factors = np.ndarray((len(self.sr_to_id)), dtype='object')
        
        self.level3_variables = np.ndarray((len(self.morphemes_to_id)), dtype='object')
        self.unary_factors = np.ndarray((len(self.morphemes_to_id)), dtype='object')
       
        # initialize layers 1 and 2
        for sr, _id in self.sr_to_id.items():
            # add the level 2 and level 1 variables

            self.level1_variables[_id] = Variable_Observed(sr, self.sigma, self.surface_forms[_id])
        
            self.phono_factors[_id] = PhonologyFactor(self.sigma, self.phonology)
            # add edge between observed node and phonology
            
            self.phono_factors[_id].edges[0] = Edge(self.level1_variables[_id], self.phono_factors[_id], self.sigma)
    
            # add edge between phonology and level 1
            #self.level2_variables[_id] = Variable_EP_Greedy(self.sigma,2)
            self.level2_variables[_id] = Variable("UR-" + sr, self.sigma, 2)

            tmp_edge = Edge(self.level2_variables[_id], self.phono_factors[_id], self.sigma)
            self.level2_variables[_id].edges[0] = tmp_edge
            self.phono_factors[_id].edges[1] = tmp_edge

        # initialize layer 3
        for morpheme, _id in self.morphemes_to_id.items():
            self.level3_variables[_id] = Variable_EP(morpheme, self.sigma, 0)

            # TOOD move to another place
            """
            self.unary_factors[_id] = ExponentialUnaryFactor(self.sigma)
            tmp_edge = Edge(self.level3_variables[_id], self.unary_factors[_id], self.sigma)
            self.unary_factors[_id].edges[0] = tmp_edge
            self.level3_variables[_id].edges[0] = tmp_edge
            """

        # initialize 
        old = morphemes_count
        morphemes_count = {}

        # concatenative factors
        for word in words:
            _id = self.sr_to_id[word.ur_id]

            # get the left-to-right order of the morphemes
            ordered_morpheme_ids = []
            for prefix_id in word.prefix_ids:
                ordered_morpheme_ids.append(self.morphemes_to_id[prefix_id])
            ordered_morpheme_ids.append(self.morphemes_to_id[word.main_id])
            for suffix_id in word.suffix_ids:
                ordered_morpheme_ids.append(self.morphemes_to_id[suffix_id])

            if len(ordered_morpheme_ids) == 2:
                self.concat_factors[_id] = TwoWayConcat(self.sigma)
            elif len(ordered_morpheme_ids) == 3:                
                self.concat_factors[_id] = ThreeWayConcat(self.sigma)
            else:
                raise Exception("Illegal Morpheme Number")
            
            tmp_edge = Edge(self.level2_variables[_id], self.concat_factors[_id], self.sigma)
            self.concat_factors[_id].edges[0] = tmp_edge
            self.level2_variables[_id].edges[1] = tmp_edge
                
            for morpheme_id_i, morpheme_id in enumerate(ordered_morpheme_ids):
                if morpheme_id not in morphemes_count:
                    morphemes_count[morpheme_id] = 0
                morphemes_count[morpheme_id] += 1

                index = len(ordered_morpheme_ids)-morpheme_id_i
                
                tmp_edge = Edge(self.level3_variables[morpheme_id], self.concat_factors[_id], self.sigma)
                self.level3_variables[morpheme_id].edges.append(tmp_edge)
                self.concat_factors[_id].edges[morpheme_id_i+1] = tmp_edge


    def inference(self, iterations=2):
        " Perform Inference by EP "
        # only done once
        for f in self.phono_factors:
            f.pass_up()

        for f in self.unary_factors:
            f.pass_down()
	    
        for v in self.level2_variables:
            v.pass_message()

        for f in self.concat_factors:
            f.pass_message()

        stuff = []
        for f in self.concat_factors:
            stuff.append(f)

        for v in self.level3_variables:
            stuff.append(v)
            
        for iteration in xrange(iterations):

            for f in self.concat_factors:
                f.pass_to_right()

            for v in self.level3_variables:
                if u"+" not in str(v):
                    v.pass_message()

            for f in self.concat_factors:
                f.pass_message()
            
            for v in self.level3_variables:
                if u"+" in str(v):
                    v.pass_message()


class ReduplicationModel(NWayConcat):
    """
    Reduplication Model
    """

    def __init__(self, words, phonology, sigma):
        super(ReduplicationModel, self).__init__(words, phonology, sigma)
        
        # add reduplication on top it 




afternoon = "Furab"
afternoon2 = "FuFurab"



        
