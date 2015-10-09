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


    def pass_up(self):
        self.sr = fst.linear_chain(self.edges[0].v.sr, syms=self.sigma, semiring="log")
        self.ur = self.sr >> self.phonology.inverse() >> self.splitter
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
        self.separator.add_arc(0,1,1,1,0.0)

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

    
    def pass_to_right(self):
        """
        pass to right
        """
        self.variables[0] = self.edges[0].m_f
        
        self.variables[0].arc_sort_output()
        self.variables[2] = (self.edges[0].m_f >> (self.edges[1].m_f + self.right_separator)) 
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
        self.variables[0] = self.edges[0].m_f
        
        self.variables[0].arc_sort_output()
        self.variables[1] = self.edges[0].m_f >> (self.left_separator + self.edges[2].m_f)

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
        self.pass_to_right()
        self.pass_to_left()

    def pass_down(self):
        self.variables[0] = self.edges[2].message + self.separator + self.edges[1].message
        self.variables[0].project_output()
        self.variables[0].arc_sort_output()
