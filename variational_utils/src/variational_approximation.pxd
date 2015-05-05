from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp.map cimport map
from libc.math cimport exp,log,log1p,floor

cimport fst._fst as fst
cimport fst.libfst as libfst

"""
ctypedef struct unigram:
    cdef char one

ctypedef struct bigram:
    cdef char one
    cdef char two

ctypedef struct trigram:
    cdef char one
    cdef char two
    cdef char three
"""




cdef class approximation_machine:
    cdef fst.LogVectorFst q
    cdef fst.LogVectorFst p
    cdef fst.SymbolTable sigma

    cdef map[string,int] *prefix2state
    cdef map[int,string] *state2prefix

    cdef map[int,int] *arc2backpointer

    cdef void _to_ones(self)
    cdef void _to_zeros(self)
    cdef void _extract_backpointers(self)
    cdef void _estimate_backpointers(self,double smoothing)

    cdef double _xent(self)

cdef class unigram(approximation_machine):
    cdef void _create_machine(self)

cdef class bigram(approximation_machine):
    cdef void _create_machine(self)

cdef class trigram(approximation_machine):
    cdef void _create_machine(self)

cdef class vo(approximation_machine):
    cdef vector[string] prefixes
    cdef void _create_machine(self,vector[string] ngrams)
