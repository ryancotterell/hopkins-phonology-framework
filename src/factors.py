import numpy as np
import fst
from utils import peek

class Factor(object):
    pass

class ExponentialUnaryFactor(Factor):
    """
    Exponential Unary Factor
    """
    def __init__(self, sigma):
        self.edges = [None]

        # underlying FSA
        self.sigma = sigma
        self.unary = fst.LogVectorFst(self.sigma, self.sigma)
        self.unary.isyms = self.sigma
        self.unary.osyms = self.sigma
        self.unary.start = self.unary.add_state()
        self.unary[0].final = fst.LogWeight.ONE
        for k, v in self.sigma.items():
            self.unary.add_arc(0, 0, v, v, 3.0)        


    def pass_down(self):
        """
        Pass the message down
        """
        self.edges[0].m_v = self.unary
    

class PhonologyFactor(Factor):
    """
    Phonology Factor
    """

    def __init__(self, sigma, phonology):
        self.edges = [None, None]
        self.phonology = phonology
        # Underlying FST
        self.sigma = sigma
        # sets the surface form and underyling form
        self.sr, self.ur = None, None


    def pass_up(self):
        self.sr = fst.linear_chain(self.edges[0].v.sr, syms=self.sigma, semiring="log")
        self.ur = self.sr >> self.phonology.inverse()
        self.ur.project_output()
        self.edges[1].m_v = self.ur


class TwoWayConcat(Factor):
    """
    Two Way Concatenative Factor
    """
    
    def __init__(self, sigma):
        self.sigma = sigma
        self.variables = np.ndarray((3), dtype="object")

        # init messages
        for i in range(3):
            self.variables[i] = fst.LogVectorFst()
            self.variables[i].isyms = self.sigma
            self.variables[i].osyms = self.sigma
            self.variables[i].start = self.variables[i].add_state()
            self.variables[i][0].final = True
            for k,v in self.sigma.items():
                if v > 1:
                    self.variables[i].add_arc(0, 0, v, v, 0.0)

        self.edges = np.ndarray((3),dtype="object")

        self.deleter_right = fst.LogVectorFst(self.sigma, self.sigma)
        self.deleter_right.isyms = self.sigma
        self.deleter_right.osyms = self.sigma
        self.deleter_right.start = self.deleter_right.add_state()
        self.deleter_right.add_state()
        self.deleter_right[1].final = True
        for k,v in self.sigma.items():
            if v > 1:
                self.deleter_right.add_arc(0, 0, v, 0, 0.0)
                self.deleter_right.add_arc(1, 1, v, v, 0.0)
            self.deleter_right.add_arc(0, 1, 1, 0, 0.0)

        self.deleter_left = fst.LogVectorFst(self.sigma, self.sigma)
        self.deleter_left.isyms = self.sigma
        self.deleter_left.osyms = self.sigma
        self.deleter_left.start = self.deleter_left.add_state()
        self.deleter_left.add_state()
        self.deleter_left[1].final = True

        for k,v in self.sigma.items():
            if v > 1:
                self.deleter_left.add_arc(0, 0, v, v, 0.0)
                self.deleter_left.add_arc(1, 1, v, 0, 0.0)
            self.deleter_left.add_arc(0, 1, 1, 0, 0.0)

        self.right_separator = fst.LogVectorFst(self.sigma, self.sigma)
        self.right_separator.isyms = self.sigma
        self.right_separator.osyms = self.sigma

        self.right_separator.start = self.right_separator.add_state()
        self.right_separator.add_state()
        self.right_separator.add_state()
        self.right_separator[2].final = True
        self.right_separator.add_arc(0, 1, 1, 1, 0.0)

        self.separator = fst.LogVectorFst(self.sigma, self.sigma)
        self.separator.start = self.separator.add_state()
        self.separator.add_state()
        self.separator[1].final = True
        self.separator.add_arc(0, 1, 1, 1, 0.0)

        for k,v in self.sigma.items():
            if v > 1:
                self.right_separator.add_arc(1, 1, v, v ,0.0)
        self.right_separator.add_arc(1, 2, 0 , 0, 0.0)

        self.left_separator = fst.LogVectorFst(self.sigma, self.sigma)
        self.left_separator.start = self.left_separator.add_state()

        self.left_separator.add_state()
        self.left_separator[1].final = True
        
        for k,v in self.sigma.items():
            if v > 1:
                self.left_separator.add_arc(0, 0, v, v, 0.0)
        self.left_separator.add_arc(0, 1, 1, 1, 0.0)

        self.right_extractor = fst.LogVectorFst(self.sigma, self.sigma)
        self.right_extractor.start = self.right_extractor.add_state()

        # make the splitter machine
        self.splitter = fst.LogVectorFst(self.sigma, self.sigma)
        self.splitter.start = self.splitter.add_state()
        self.splitter.add_state()
        self.splitter[1].final = True

        for k, v in self.sigma.items():
            if v > 1:
                self.splitter.add_arc(0, 0, v, v, 0.0)
        self.splitter.add_arc(0, 1, 0, 1, 0.0)

        for k, v in self.sigma.items():
            if v > 1:
                self.splitter.add_arc(1, 1, v, v, 0.0)

        self.splitter.arc_sort_input()
        self.splitter.isyms = self.sigma
        self.splitter.osyms = self.sigma

    
    def pass_to_right(self):
        """
        pass to right
        """
        self.variables[0] = self.edges[0].m_f >> self.splitter
        self.variables[0].project_output()
        
        self.variables[0].arc_sort_output()
        self.variables[2] = (self.variables[0] >> (self.edges[1].m_f + self.right_separator)) 
        self.variables[2].arc_sort_output()
        self.variables[2].project_output()
        self.variables[2] = self.variables[2] >> self.deleter_right
        self.variables[2].project_output()

        self.variables[0].isyms = self.sigma
        self.variables[0].osyms = self.sigma
        self.variables[2].isyms = self.sigma
        self.variables[2].osyms = self.sigma

        self.edges[0].m_v = self.variables[0]
        self.edges[2].m_v = self.variables[2]


    def pass_to_left(self):
        """
        pass to left
        """
        self.variables[0] = self.edges[0].m_f >> self.splitter
        self.variables[0].project_output()
        
        self.variables[0].arc_sort_output()
        self.variables[1] = self.variables[0] >> (self.left_separator + self.edges[2].m_f)

        self.variables[0].isyms = self.sigma
        self.variables[0].osyms = self.sigma
        self.variables[1].isyms = self.sigma
        self.variables[1].osyms = self.sigma
      
        self.variables[1].arc_sort_output()
        self.variables[1].project_output()
        
        self.variables[1] = self.variables[1] >> self.deleter_left
        self.variables[1].project_output()

        self.edges[0].m_v = self.variables[0]
        self.edges[1].m_v = self.variables[1]
                
    def pass_message(self):
        """
        pass all the message (left and right)
        """
        self.pass_to_right()
        self.pass_to_left()

    def pass_down(self):
        """
        pass down (only for training)
        """
        self.variables[0] = self.edges[2].message + self.separator + self.edges[1].message
        self.variables[0].project_output()
        self.variables[0].arc_sort_output()


class ThreeWayConcat(Factor):
    """
    Concatenate Three Morphemes Together
    """

    def __init__(self,sigma):
        self.sigma = sigma
        self.variables = np.ndarray((4),dtype="object")
        
        # init messages
        for i in range(4):
            self.variables[i] = fst.LogVectorFst()
            self.variables[i].isyms = self.sigma
            self.variables[i].osyms = self.sigma
            self.variables[i].start = self.variables[i].add_state()
            self.variables[i][0].final = True
			
        for k,v in self.sigma.items():
            if v > 1:
                self.variables[i].add_arc(0,0,v,v,0.0)

        self.edges = np.ndarray((4), dtype="object")
        
        self.separator = fst.LogVectorFst(self.sigma,self.sigma)
        self.separator.start = self.separator.add_state()
        self.separator.add_state()
        self.separator[1].final = True
        self.separator.add_arc(0,1,1,1,0.0)
        
        self.deleter_right = fst.LogVectorFst(self.sigma,self.sigma)
        self.deleter_right.isyms = self.sigma
        self.deleter_right.osyms = self.sigma
        self.deleter_right.start = self.deleter_right.add_state()
        self.deleter_right.add_state()
        self.deleter_right.add_state()
        self.deleter_right[2].final = True
		
        for k,v in self.sigma.items():
            if v > 1:
                self.deleter_right.add_arc(0, 0, v, 0, 0.0)
                self.deleter_right.add_arc(1, 1, v, 0, 0.0)
                self.deleter_right.add_arc(2, 2, v, v, 0.0)
                
        self.deleter_right.add_arc(0, 1, 1, 0, 0.0)
        self.deleter_right.add_arc(1, 2, 1, 0, 0.0)

        self.deleter_middle = fst.LogVectorFst(self.sigma,self.sigma)
        self.deleter_middle.isyms = self.sigma
        self.deleter_middle.osyms = self.sigma
        self.deleter_middle.start = self.deleter_middle.add_state()
        self.deleter_middle.add_state()
        self.deleter_middle.add_state()
        self.deleter_middle[2].final = True

        for k,v in self.sigma.items():
            if v > 1:
                self.deleter_middle.add_arc(0, 0, v, 0, 0.0)
                self.deleter_middle.add_arc(1, 1, v, v, 0.0)
                self.deleter_middle.add_arc(2, 2, v, 0, 0.0)

        self.deleter_middle.add_arc(0, 1, 1, 0, 0.0)
        self.deleter_middle.add_arc(1, 2, 1, 0, 0.0)

        self.deleter_left = fst.LogVectorFst(self.sigma,self.sigma)
        self.deleter_left.isyms = self.sigma
        self.deleter_left.osyms = self.sigma
        self.deleter_left.start = self.deleter_left.add_state()
        self.deleter_left.add_state()
        self.deleter_left.add_state()
        self.deleter_left[2].final = True
        
        for k,v in self.sigma.items():
            if v > 1:
                self.deleter_left.add_arc(0, 0, v, v, 0.0)
                self.deleter_left.add_arc(1, 1, v, 0, 0.0)
                self.deleter_left.add_arc(2, 2, v, 0, 0.0)

        self.deleter_left.add_arc(0, 1, 1, 0, 0.0)
        self.deleter_left.add_arc(1, 2, 1, 0, 0.0)

        self.right_separator = fst.LogVectorFst(self.sigma, self.sigma)
        self.right_separator.start = self.right_separator.add_state()
        self.right_separator.add_state()
        self.right_separator.add_state()
        self.right_separator[2].final = True
        self.right_separator.add_arc(0, 1, 1, 1, 0.0)
        for k,v in self.sigma.items():
            if v > 1:
                self.right_separator.add_arc(1, 1, v, v, 0.0)
        self.right_separator.add_arc(1, 2, 0, 0, 0.0)

        # left separator
        self.left_separator = fst.LogVectorFst(self.sigma, self.sigma)
        self.left_separator.start = self.left_separator.add_state()
        self.left_separator.add_state()
        self.left_separator[1].final = True
        
        for k,v in self.sigma.items():
            if v > 1:
                self.left_separator.add_arc(0, 0, v, v, 0.0)
        self.left_separator.add_arc(0, 1, 1, 1, 0.0)

        # make the splitter machine
        self.splitter = fst.LogVectorFst(self.sigma, self.sigma)
        self.splitter.start = self.splitter.add_state()
        self.splitter.add_state()
        self.splitter.add_state()
        self.splitter[2].final = True

        for k, v in self.sigma.items():
            if v > 1:
                self.splitter.add_arc(0, 0, v, v, 0.0)
        self.splitter.add_arc(0, 1, 0, 1, 0.0)
        self.splitter.add_arc(1, 2, 0, 1, 0.0)

        for k, v in self.sigma.items():
            if v > 1:
                self.splitter.add_arc(1, 1, v, v, 0.0)
                self.splitter.add_arc(2, 2, v, v, 0.0)

        self.splitter.arc_sort_input()
        self.splitter.isyms = self.sigma
        self.splitter.osyms = self.sigma

    
    def pass_to_right(self):
        """
        pass to right
        """

        self.variables[0] = self.edges[0].m_f >> self.splitter
        self.variables[0].project_output()
        self.variables[0].arc_sort_output()
        self.variables[3] = self.variables[0] >> (self.edges[1].m_f + self.separator + self.edges[2].m_f + self.right_separator)
        self.variables[3].arc_sort_output()
        self.variables[3].project_output()
        self.variables[3].isyms = self.sigma
        self.variables[3].osyms = self.sigma

        self.variables[3] = self.variables[3] >> self.deleter_right
        self.variables[3].project_output()

        self.edges[3].m_v = self.variables[3]


    def pass_to_left(self):
        """
        pass to left
        """

        self.variables[0] = self.edges[0].m_f >> self.splitter
        self.variables[0].project_output()
        self.variables[0].arc_sort_output()
        self.variables[1] = self.variables[0] >> (self.left_separator + self.edges[2].m_f + self.separator + self.edges[3].m_f)
        self.variables[1].arc_sort_output()
        self.variables[1].project_output()
        self.variables[1].isyms = self.sigma
        self.variables[1].osyms = self.sigma

        self.variables[1] = self.variables[1] >> self.deleter_left
        self.variables[1].project_output()

        self.edges[1].m_v = self.variables[1]


    def pass_to_middle(self):
        """
        pass to middle
        """
        self.variables[0] = self.edges[0].m_f >> self.splitter
        self.variables[0].project_output()
        self.variables[0].arc_sort_output()
        self.variables[2] = self.variables[0] >> (self.edges[1].m_f + self.right_separator + self.separator + self.edges[3].m_f)
        self.variables[2].arc_sort_output()
        self.variables[2].project_output()
        self.variables[2].isyms = self.sigma
        self.variables[2].osyms = self.sigma

        self.variables[2] = self.variables[2] >> self.deleter_middle
        self.variables[2].project_output()

        self.edges[2].m_v = self.variables[2]


    def pass_message(self):
        """
        pass all the messages (right, middle and left)
        """
        self.pass_to_right()
        self.pass_to_middle()
        self.pass_to_left()

    def pass_down(self):
        """
        pass down (only for training)
        """
        self.variables[0] = self.edges[2].message + self.separator + self.edges[1].message
        self.variables[0].project_output()
        self.variables[0].arc_sort_output()


class TwoWayTemplatic(Factor):
    """
    Two way template factor
    """
    def __init__(self, class1, class2, sigma, delta):
        """
        class1 : the first class of letters
        class2 : the second class of letters
        """

        self.sigma = sigma
        self.delta = delta
        
        self.class1 = class1
        self.class2 = class2

        self.edges = [None, None, None, None]

        # class 1 extractor
        self.extract_class1 = fst.LogVectorFst(isyms=self.sigma, osyms=self.sigma)
        self.extract_class1.start = self.extract_class1.add_state()
        self.extract_class1[0].final = True

        for v in self.class1:
            self.extract_class1.add_arc(0,0, self.sigma[v], self.sigma[v], 0.0)
        for c in self.class2:
            self.extract_class1.add_arc(0,0, self.sigma[c], 0, 0.0)

        # class 2 extractor
        self.extract_class2 = fst.LogVectorFst(isyms=self.sigma, osyms=self.sigma)
        self.extract_class2.start  = self.extract_class2.add_state()
       
        self.extract_class2[0].final = True

        for v in self.class1:
            self.extract_class2.add_arc(0, 0, self.sigma[v], 0, 0.0)
        for c in self.class2:
            self.extract_class2.add_arc(0, 0, self.sigma[c], self.sigma[c], 0.0)
            
        # alignment extractor
        self.extract_alignment = fst.LogVectorFst(isyms=self.sigma, osyms=self.delta)
        self.extract_alignment.start = self.extract_alignment.add_state()
        self.extract_alignment[0].final = True

        for v in self.class1:
            self.extract_alignment.add_arc(0,0,self.sigma[v],1,0.0)
        for c in self.class2:
            self.extract_alignment.add_arc(0,0,self.sigma[c],2,0.0)

        # interdigitator
        self.interdigitator1 = fst.LogVectorFst(isyms=self.delta, osyms=self.delta)
        self.interdigitator1.start = self.interdigitator1.add_state()

        self.interdigitator2 = fst.LogVectorFst(isyms=self.delta, osyms=self.delta)
        self.interdigitator2.start = self.interdigitator2.add_state()

        self.interdigitator1[0].final = True
        self.interdigitator1.add_arc(0, 0, 1, 1, 0.0)
        self.interdigitator1.add_arc(0, 0, 0, 2, 0.0)

        self.interdigitator2[0].final = True
        self.interdigitator2.add_arc(0, 0, 1, 0, 0.0)
        self.interdigitator2.add_arc(0, 0, 2, 2, 0.0)
        
        # replacers
        self.replacer_class1 = fst.LogVectorFst(isyms=self.sigma, osyms=self.delta)
        self.replacer_class1.start = self.replacer_class1.add_state()
        
        self.replacer_class2 = fst.LogVectorFst(isyms=self.delta,osyms=self.sigma)
        self.replacer_class2.start = self.replacer_class2.add_state()

        for c in self.class1:
            self.replacer_class1.add_arc(0, 0, self.sigma.find(c), 1)
        self.replacer_class1[0].final = True

        for c in self.class2:
            self.replacer_class2.add_arc(0, 0, 2, self.sigma.find(c))
        self.replacer_class2[0].final = True
        
        # message
        self.ur_message_down = fst.LogVectorFst(isyms=self.sigma, osyms=self.sigma)
        self.ur_message_down.start = self.ur_message_down.add_state()
        self.ur_message_down[0].final = True
        for k, v in self.sigma.items():
            self.ur_message_down.add_arc(0, 0, v, v, 0.0)


    def pass_up_through(self):
        """
        pass the message up
        """
        for edge_i, edge in enumerate(self.edges):          
            if edge_i == 0:          
                # extract class 1
                self.up_message_class1 = (edge.m_f  >> self.extract_class1)
                self.up_message_class1.project_output()
                #self.up_message_class1.remove_epsilon()
                #self.up_message_class1 = self.up_message_class1.determinize()
                #self.up_message_class1.minimize()
          
                # extract class 2
                self.up_message_class2 = (edge.m_f  >> self.extract_class2)
                self.up_message_class2.project_output()
                #self.up_message_class2.remove_epsilon()
                #self.up_message_class2 = self.up_message_class2.determinize()
                #self.up_message_class2.minimize()

                # extract alignment
                self.up_message_alignment = (edge.m_f  >> self.extract_alignment)
                self.up_message_alignment.project_output()
                #self.up_message_alignment.remove_epsilon()
                #self.up_message_alignment = self.up_message_alignment.determinize()
                #self.up_message_alignment.minimize()
            
            else:
                # extract class 1
                self.up_message_class1.arc_sort_output()
                self.up_message_class1 = self.up_message_class1 >> (edge.m_f  >> self.extract_class1)
                self.up_message_class1.project_output()
                #self.up_message_class1.remove_epsilon()
                #self.up_message_class1 = self.up_message_class1.determinize()
                #self.up_message_class1.minimize()
             
                # extract class 2
                self.up_message_class2.arc_sort_output()
                self.up_message_class2 = self.up_message_class2 >> (edge.m_f  >> self.extract_class2)
                self.up_message_class2.project_output()
                #self.up_message_class2.remove_epsilon()
                #self.up_message_class2 = self.up_message_class2.determinize()
                #self.up_message_class2.minimize()

                # extract alignment
                self.up_message_alignment.arc_sort_output()
                tmp = (edge.m_f  >> self.extract_alignment)
                tmp.project_output()
                self.up_message_alignment = self.up_message_alignment >> tmp
                self.up_message_alignment.project_output()
                #self.up_message_alignment.remove_epsilon()
                #self.up_message_alignment = self.up_message_alignment.determinize()
                #self.up_message_alignment.minimize()

        print "CLASS 1"
        peek(self.up_message_class1, 10)

        print "CLASS 2"
        peek(self.up_message_class2, 10)

    def pass_down_through(self):
        """
        Pass the message down
        """
        tmp = self.class1_edge.message >> self.replacer_class1 \
              >> self.interdigitator1 >> self.alignment_edge.message \
              >> self.interdigitator2 >> self.replacer_class2 >> self.class2_edge.message
        
        for state in tmp:
            for arc in state:
                if arc.ilabel > 0 and arc.olabel == 0:
                    arc.olabel = arc.ilabel
                elif arc.olabel > 0 and arc.ilabel == 0:
                    arc.ilabel = arc.olabel
        tmp.project_        
        self.ur_message_down = tmp



