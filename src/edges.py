import fst
import numpy as np

class Edge:
    """
    Edge connecting a factor and a variable
    """
    def __init__(self, v, f, sigma):
        self.v = v
        self.f = f
        self.sigma = sigma
        
        # variable message
        self.m_v = fst.LogVectorFst(self.sigma, self.sigma)
        self.m_v.isyms = self.sigma
        self.m_v.osyms = self.sigma

        self.m_v.start = self.m_v.add_state()
        self.m_v[0].final = fst.LogWeight.ONE
        for k, v in self.sigma.items():
            self.m_v.add_arc(0, 0, v, v, 1.0)        

        # factor message
        self.m_f = fst.LogVectorFst(self.sigma, self.sigma)
        self.m_f.isyms = self.sigma
        self.m_f.osyms = self.sigma

        self.m_f.start = self.m_f.add_state()
        self.m_f[0].final = fst.LogWeight.ONE
        for k, v in self.sigma.items():
            self.m_f.add_arc(0, 0, v, v, 1.0)        

        
        
