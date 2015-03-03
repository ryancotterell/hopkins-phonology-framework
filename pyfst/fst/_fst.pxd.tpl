cimport libfst
cimport _fst
cimport sym

cdef class SymbolTable:
    cdef sym.SymbolTable* table

cdef class _Fst:
    pass
{{#types}}
cdef class {{weight}}:
    cdef libfst.{{weight}}* weight

cdef class {{arc}}:
    cdef libfst.{{arc}}* arc

cdef class {{state}}:
    cdef public int stateid
    cdef libfst.{{fst}}* fst

cdef class {{fst}}(_Fst):
    cdef libfst.{{fst}}* fst
    cdef public SymbolTable isyms, osyms

{{/types}}
cdef class SignedLogWeight:
    cdef libfst.SignedLogWeight* weight

cdef class ExpectationWeight:
    cdef libfst.ExpectationWeight* weight

cdef class ExpectationArc:
    cdef libfst.ExpectationArc* arc

cdef class ExpectationState:
    cdef public int stateid
    cdef libfst.ExpectationVectorFst* fst

cdef class ExpectationVectorFst(_Fst):
    cdef libfst.ExpectationVectorFst* fst
    cdef public SymbolTable isyms, osyms
