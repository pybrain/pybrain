from pyrex_header cimport ndarray, import_array
import_array()


def Modulereset(self):
    self.time = 0
    self.seqlen = 0
    cdef ndarray inbuf
    cdef ndarray outbuf
    cdef ndarray inerr
    cdef ndarray outerr
 
    inbuf = self.inputbuffer
    outbuf = self.outputbuffer
    inerr = self.inputerror
    outerr = self.outputerror
    
    cdef double * inbufp 
    cdef double * outbufp
    cdef double * inerrp
    cdef double * outerrp
 
    inbufp = <double*> inbuf.data
    outbufp = <double*> outbuf.data
    inerrp = <double*> inerr.data
    outerrp = <double*> outerr.data

    cdef int c
    cdef int buf_len 
    
    buf_len = inbuf.dimensions[0] * inbuf.dimensions[1]
    for c from 0 <= c < buf_len:
        inbufp[0] = 0.0
        inbufp = inbufp + 1
        inerrp[0] = 0.0
        inerrp = inerrp + 1
    
    buf_len = outbuf.dimensions[0] * outbuf.dimensions[1]
    for c from 0 <= c < buf_len:
        outbufp[0] = 0.0
        outbufp = outbufp + 1
        outerrp[0] = 0.0
        outerrp = outerrp + 1
    
    