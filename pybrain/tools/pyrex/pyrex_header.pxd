import numpy

# accessing math functions directly from C
cdef extern from "math.h":
   double sqrt(double x)
   double exp(double x)
   double tanh(double x)

# defining the types
cdef extern from "numpy/arrayobject.h":

   # integer pointer
   ctypedef int intp

   # hmmm. ?
   ctypedef extern class numpy.dtype [object PyArray_Descr]:
       cdef int type_num, elsize, alignment
       cdef char type, kind, byteorder, hasobject
       cdef object fields, typeobj

   # defining numpy arrays
   ctypedef extern class numpy.ndarray [object PyArrayObject]:
       cdef char *data
       cdef int nd
       cdef intp *dimensions
       cdef intp *strides
       cdef object base
       cdef dtype descr
       cdef int flags

   # we don't understand this, probably magic.
   void import_array()



