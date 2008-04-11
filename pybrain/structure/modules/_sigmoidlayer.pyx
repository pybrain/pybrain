from pyrex_header cimport ndarray, import_array
from pybrain.tools.pyrex.pyrex_shared import sigmoid
import_array()


def SigmoidLayer_forwardImplementation(self, ndarray inbuf, ndarray outbuf):
    cdef double * inbufp 
    inbufp = <double*> inbuf.data
    cdef double * outbufp
    outbufp = <double*> outbuf.data
    cdef int c
    cdef int buf_len 
    buf_len = inbuf.dimensions[0]
    for c from 0 <= c < buf_len:
        outbufp[0] = sigmoid(inbufp[0])
        outbufp = outbufp + 1
        inbufp = inbufp + 1
    	