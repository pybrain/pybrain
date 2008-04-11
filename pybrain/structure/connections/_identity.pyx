from pyrex_header cimport ndarray, import_array
import_array()


def IdentityConnection_forwardImplementation(self, ndarray inbuf, ndarray outbuf):
	# typecast: buf.data is a double pointer (not char*)
    cdef double * inbufp 
    inbufp = <double*> inbuf.data
    cdef double * outbufp
    outbufp = <double*> outbuf.data
    cdef int c
    cdef int buf_len 
    buf_len = inbuf.dimensions[0]
    for c from 0 <= c < buf_len:
    	# dereferencing
        outbufp[0] = outbufp[0] + inbufp[0]
        # increment pointers
        outbufp = outbufp + 1
        inbufp = inbufp + 1
    	