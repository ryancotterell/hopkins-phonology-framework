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

#from exceptions import IllegalMorphemeNumber
import utils
from utils import peek

# TODO:
# seed random number generator

class ConcatPhonologyModel(object):
    """
    Concatenative phonology model.
    Currently supports words with 2 and 3 
    morphemes.
    """

    def __init__(self, words):
        # get number of unique morphemes
        # cache their integer ids
        self.morpheme1_to_id = {}
        self.morpheme2_to_id = {}
        self.sr_to_id = {}
        self.id_to_morpheme1 = {}
        self.id_to_morpheme2 = {}
        self.id_to_sr = {}

        morpheme1_count = {}
        morpheme2_count = {}
        
        self.sigma = fst.SymbolTable()
        # designated split symbol
        self.sigma["#"] = 1
        for word in words:
            # intern the variables
            if word.morpheme1 not in self.morpheme1_to_id:
                self.morpheme1_to_id[word.morpheme1] = len(self.morpheme1_to_id)
                self.id_to_morpheme1[len(self.morpheme1_to_id)-1] = word.morpheme1
                morpheme1_count[word.morpheme1] = 1
            else:
                morpheme1_count[word.morpheme1] += 1

            if word.morpheme2 not in self.morpheme2_to_id:
                self.morpheme2_to_id[word.morpheme2] = len(self.morpheme2_to_id)
                self.id_to_morpheme2[len(self.morpheme2_to_id)-1] = word.morpheme2
                morpheme2_count[word.morpheme2] = 1
            else:
                morpheme2_count[word.morpheme2] += 1

            if word.sr not in self.sr_to_id:
                self.sr_to_id[word.sr] = len(self.sr_to_id)
                self.id_to_sr[len(self.sr_to_id)-1] = word.sr

            # assumes surface alphabet is the
            # same as the underlying alphabet
            for c in list(word.sr):
                if c not in self.sigma:
                    self.sigma[c] = len(self.sigma)
                    
        # instantiates the variables
        self.unary_factors1 = np.ndarray((len(self.morpheme1_to_id)), dtype='object')
        self.unary_factors2 = np.ndarray((len(self.morpheme2_to_id)), dtype='object')
        self.level3_variables1 = np.ndarray((len(self.morpheme1_to_id)), dtype='object')
        self.level3_variables2 = np.ndarray((len(self.morpheme2_to_id)), dtype='object')

        # initialize layer 3
        for morpheme1, _id in self.morpheme1_to_id.items():
            # add unary factor
            self.unary_factors1[_id] = ExponentialUnaryFactor(self.sigma)
            self.level3_variables1[_id] = Variable_EP(morpheme1, self.sigma, morpheme1_count[morpheme1])
            
            tmp_edge = Edge(self.level3_variables1[_id], self.unary_factors1[_id], self.sigma)
            self.unary_factors1[_id].edges[0] = tmp_edge
            self.level3_variables1[_id].edges[0] = tmp_edge

        for morpheme2, _id in self.morpheme2_to_id.items():
            # add level three variables
            self.unary_factors2[_id] = ExponentialUnaryFactor(self.sigma)
            self.level3_variables2[_id] = Variable_EP(morpheme2, self.sigma, morpheme2_count[morpheme2])

            tmp_edge = Edge(self.level3_variables2[_id], self.unary_factors2[_id], self.sigma)
            self.unary_factors2[_id].edges[0] = tmp_edge
            self.level3_variables2[_id].edges[0] = tmp_edge

        self.level1_variables = np.ndarray((len(self.sr_to_id)), dtype='object')
        self.level2_variables = np.ndarray((len(self.sr_to_id)), dtype='object')
        self.phono_factors = np.ndarray((len(self.sr_to_id)), dtype='object')
        self.concat_factors = np.ndarray((len(self.sr_to_id)), dtype='object')

        # initialize layers 1 and 2
        for sr, _id in self.sr_to_id.items():
            # add the level 2 and level 1 variables
            self.level1_variables[_id] = Variable_Observed("SR-" + sr, self.sigma, sr)
            # create the phonology factor
            self.phono_factors[_id] = PhonologyFactor(self.sigma, utils.phonology_edit(self.sigma, .99))
            # add edge between observed node and phonology
            tmp_edge = Edge(self.level1_variables[_id], self.phono_factors[_id], self.sigma)
            self.phono_factors[_id].edges[0] = tmp_edge

            # add edge between phonology and level 1
            self.level2_variables[_id] = Variable("UR-" + sr, self.sigma, 2)

            tmp_edge = Edge(self.level2_variables[_id], self.phono_factors[_id], self.sigma)
            self.level2_variables[_id].edges[0] = tmp_edge
            self.phono_factors[_id].edges[1] = tmp_edge
           
            # concatenation factors
            self.concat_factors[_id] = TwoWayConcat(self.sigma)
            tmp_edge =  Edge(self.level2_variables[_id], self.concat_factors[_id], self.sigma)
            self.concat_factors[_id].edges[0] = tmp_edge
            self.level2_variables[_id].edges[1] = tmp_edge
        
        # initialize the edges between layer 2 and layer 3
        # reuse
        morpheme1_count = {}
        morpheme2_count = {}

        for word in words:
            sr_id = self.sr_to_id[word.sr]
            morpheme1_id = self.morpheme1_to_id[word.morpheme1]
            morpheme2_id = self.morpheme2_to_id[word.morpheme2]

            if word.morpheme1 not in morpheme1_count:
                morpheme1_count[word.morpheme1] = 1
            else:
                morpheme1_count[word.morpheme1] += 1

            if word.morpheme2 not in morpheme2_count:
                morpheme2_count[word.morpheme2] = 1
            else:
                morpheme2_count[word.morpheme2] += 1

            tmp_edge = Edge(self.level3_variables1[morpheme1_id], morpheme1_count[word.morpheme1], self.sigma)
            self.concat_factors[sr_id].edges[1] = tmp_edge
            self.level3_variables1[morpheme1_id].edges[morpheme1_count[word.morpheme1]] = tmp_edge

            tmp_edge = Edge(self.level3_variables2[morpheme2_id], morpheme2_count[word.morpheme2], self.sigma)
            self.concat_factors[sr_id].edges[2] = tmp_edge
            self.level3_variables2[morpheme2_id].edges[morpheme2_count[word.morpheme2]] = tmp_edge
        

    def inference(self, iterations=2):
        # only done once
        for f in self.phono_factors:
            f.pass_up()

        for f in self.unary_factors1:
            f.pass_down()
        
        for f in self.unary_factors2:
            f.pass_down()
	    
        for f in self.level2_variables:
            f.pass_message()
            
        for iteration in xrange(iterations):
            #print "ITERATION", iteration
            #print "PASSING TO LEFT"
            for f in self.concat_factors:
                f.pass_to_left()

            #print "PASSING TO RIGHT"
            for f in self.concat_factors:
                f.pass_to_right()
                
            #print "DONE PASSING LEFT"
            for f in self.level3_variables1:
                f.pass_message()

            #print "DONE PASSING RIGHT"
            for f in self.level3_variables2:
                f.pass_message()


class TemplaticPhonologyModel(object):
    """
    Model for templatic morphology
    """
    def __init__(self, words, phonology, sigma, phono_approx=False):

        # CLASS
        vowels = set(["A", "a", "I", "i", "O", "o", "E", "e", "U", "u"])
        self.class1 = []
        self.class2 = []

        self.phonology = phonology
        self.sigma = sigma

        self.phono_approx = phono_approx

        # get number of unique morphemes
        # cache their integer ids
        self.morphemes_to_id = {}
        self.sr_to_id = {}
        self.id_to_morphemes = {}
        self.id_to_sr = {}

        self.id2main = {}
        self.main2id = {}
        self.root2id = {}
        self.id2root = {}
        self.pattern2id = {}
        self.id2pattern = {}

        self.morphemeid2mainid = {}
        self.mainid2patternid = {}
        self.mainid2rootid = {}

        self.surface_forms = {}

        morphemes_count = {}
        root_count = {}
        pattern_count = {}
  
        self.num_morphemes = {}
       
        self.morpheme_id_to_type = {}
        self.has_suffix = {}
        self.has_prefix = {}

        # self.sigma = fst.SymbolTable()
        # self.sigma["#"] = 1

        self.delta = fst.SymbolTable()
        self.delta["C"] = 1
        self.delta["V"] = 2

        for word in words:
            # assumes surface alphabet is the
            # same as the underlying alphabet
            for c in list(word.sr):
                assert c in self.sigma
                #self.sigma[c] = len(self.sigma)
                if c in vowels and c not in self.class1:
                    self.class1.append(c)
                if c not in vowels and c not in self.class2:
                    self.class2.append(c)

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

            # add root and pattern
            if word.pattern_id not in pattern_count:
                pattern_count[word.pattern_id] = 0
            pattern_count[word.pattern_id] += 1

            if word.root_id not in root_count:
                root_count[word.root_id] = 0
            root_count[word.root_id] += 1

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

            if word.root_id not in self.root2id:
                self.root2id[word.root_id] = len(self.root2id)
                self.id2root[len(self.root2id)-1] = word.root_id

            self.mainid2rootid[self.main2id[word.main_id]] = self.root2id[word.root_id]

            if word.pattern_id not in self.pattern2id:
                self.pattern2id[word.pattern_id] = len(self.pattern2id)
                self.id2pattern[len(self.pattern2id)-1] = word.pattern_id
                #self.mainid2patternid[self.morphemes_to_id[word.main_id]] = len(self.pattern2id) - 1
                
            self.mainid2patternid[self.main2id[word.main_id]] = self.pattern2id[word.pattern_id]

        self.level1_variables = np.ndarray((len(self.sr_to_id)), dtype='object')
        self.level2_variables = np.ndarray((len(self.sr_to_id)), dtype='object')
        self.phono_factors = np.ndarray((len(self.sr_to_id)), dtype='object')
        self.concat_factors = np.ndarray((len(self.sr_to_id)), dtype='object')
        
        self.level3_variables = np.ndarray((len(self.morphemes_to_id)), dtype='object')
        self.unary_factors = np.ndarray((len(self.morphemes_to_id)), dtype='object')
        self.binyan_factors = np.ndarray((len(self.main2id)), dtype='object')
        
        self.pattern_variables = np.ndarray((len(self.pattern2id)), dtype='object')
        self.alignment_variables = np.ndarray((len(self.pattern2id)), dtype='object')
        self.root_variables = np.ndarray((len(self.root2id)), dtype='object')
       
        # initialize the root and pattern variables
        for k, v in root_count.items():
            self.root_variables[self.root2id[k]] = Variable_EP("Root " + k, self.sigma, v)
        for k, v in pattern_count.items():
            self.pattern_variables[self.pattern2id[k]] = Variable_EP("Pattern " + k, self.sigma, v)
            self.alignment_variables[self.pattern2id[k]] = Variable_Pruned("Alignment " + k, self.delta, v)
            
        # binyan factors
        for main, _id in self.main2id.items():
            self.binyan_factors[_id] = TwoWayTemplatic(self.class2, self.class1, self.sigma, self.delta)

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

        root_count = {}
        pattern_count = {}
        # initialize layer 3
        for morpheme, _id in self.morphemes_to_id.items():
            self.level3_variables[_id] = Variable_EP(morpheme, self.sigma, 0)
            self.unary_factors[_id] = ExponentialUnaryFactor(self.sigma)

            if self.morpheme_id_to_type[_id] == 1: # is a stem?
                # binyan
                tmp_edge = Edge(self.level3_variables[_id], self.binyan_factors[self.morphemeid2mainid[_id]], self.sigma)
                self.binyan_factors[self.morphemeid2mainid[_id]].edges[0] = tmp_edge
                self.level3_variables[_id].edges.append(tmp_edge)
                        
                # pattern
                if self.mainid2patternid[self.morphemeid2mainid[_id]] not in pattern_count:
                    pattern_count[self.mainid2patternid[self.morphemeid2mainid[_id]]] = 0
                                    
                tmp_edge = Edge(self.pattern_variables[self.mainid2patternid[self.morphemeid2mainid[_id]]], self.binyan_factors[self.morphemeid2mainid[_id]], self.sigma)
                self.pattern_variables[self.mainid2patternid[self.morphemeid2mainid[_id]]].edges[pattern_count[self.mainid2patternid[self.morphemeid2mainid[_id]]]] = tmp_edge
                self.binyan_factors[self.morphemeid2mainid[_id]].edges[3] = tmp_edge

                # alignment
                tmp_edge = Edge(self.alignment_variables[self.mainid2patternid[self.morphemeid2mainid[_id]]], self.binyan_factors[self.morphemeid2mainid[_id]], self.delta)
                self.alignment_variables[self.mainid2patternid[self.morphemeid2mainid[_id]]].edges[pattern_count[self.mainid2patternid[self.morphemeid2mainid[_id]]]] = tmp_edge
                self.binyan_factors[self.morphemeid2mainid[_id]].edges[2] = tmp_edge
                
                # root
                if self.mainid2rootid[self.morphemeid2mainid[_id]] not in root_count:
                    root_count[self.mainid2rootid[self.morphemeid2mainid[_id]]] = 0

                tmp_edge = Edge(self.root_variables[self.mainid2rootid[self.morphemeid2mainid[_id]]], self.binyan_factors[self.morphemeid2mainid[_id]].edges[1], self.sigma)
                self.root_variables[self.mainid2rootid[self.morphemeid2mainid[_id]]].edges[root_count[self.mainid2rootid[self.morphemeid2mainid[_id]]]] = tmp_edge
                self.binyan_factors[self.morphemeid2mainid[_id]].edges[1] = tmp_edge
                
                pattern_count[self.mainid2patternid[self.morphemeid2mainid[_id]]] += 1
                root_count[self.mainid2rootid[self.morphemeid2mainid[_id]]] += 1

            else:
                tmp_edge = Edge(self.level3_variables[_id], self.unary_factors[_id], self.sigma)
                self.unary_factors[_id].edges[0] = tmp_edge
                self.level3_variables[_id].edges[0] = tmp_edge

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
                #raise IllegalMorphemeNumber(len(ordered_morpheme_ids))
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


    def training_data(self, n=5):
        " Gets the training data "

        for factor in self.concat_factors:
            factor.pass_down()
            print factor
            peek(factor.variables[0], 10)

        data = []
        for var1, var2 in zip(self.level1_variables, self.level2_variables):
            var2.compute_belief()
            ur = fst.LogVectorFst(fst.StdVectorFst(var2.belief).shortest_path(n=n))
            sr = var1.value
            data.append((ur, sr))

        return data


    def predict(self, root, pattern, prefix, suffix):
        " Captures it in a try block "
        try:
            return self._predict(root, pattern, prefix, suffix)
        except KeyError:
            return None


    def _predict(self, root, pattern, prefix, suffix):
        """ 
        Return the prediction based on a combination
        of a root, pattern, prefix and suffix and
        then run it through the phonology.
        """
        pattern_id = self.pattern2id[pattern]
        root_id = self.root2id[root]

        # HACK: to get out interdigitator
        interdigitator1 = self.binyan_factors[0].interdigitator1
        interdigitator2 = self.binyan_factors[0].interdigitator2
        replacer_class1 = self.binyan_factors[0].replacer_class1
        replacer_class2 = self.binyan_factors[0].replacer_class2
        
        pattern_var = self.pattern_variables[pattern_id].belief
        root_var = self.root_variables[root_id].belief
        alignment_var = self.alignment_variables[pattern_id].belief

        prediction = root_var >> replacer_class1 \
              >> interdigitator1 >> alignment_var \
              >> interdigitator2 >> replacer_class2 >> pattern_var

        # switch the labels
        for state in prediction:
            for arc in state:
                if arc.ilabel > 0 and arc.olabel == 0:
                    arc.olabel = arc.ilabel
                elif arc.olabel > 0 and arc.ilabel == 0:
                    arc.ilabel = arc.olabel

        prediction.project_output()

        if prefix is not None:
            prefix_var = self.level3_variables[self.morphemes_to_id[prefix]]
            prediction = prefix_var.belief + prediction
        if suffix is not None:
            suffix_var = self.level3_variables[self.morphemes_to_id[suffix]]
            prediction = prediction + suffix_var.belief

        prediction = prediction >> self.phonology
        prediction.project_output()
        return prediction


    def inference(self, iterations=2):
        " Perform Inference by EP "
        # only done once
        for f in self.phono_factors:
            f.pass_up(self.phono_approx)

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
                
            if iteration > 3:
                for f in self.binyan_factors:
                    f.pass_up_through()
                for v in self.pattern_variables:
                    v.pass_message()
                for v in self.root_variables:
                    v.pass_message()
                for v in self.alignment_variables:
                    v.pass_message()
                for f in self.binyan_factors:
                    f.pass_down_through()

