from pyrex_header cimport ndarray, import_array
import_array()

def SharedFullConnectionforward(self, int time, desttime = None):
    if desttime == None:
        desttime = time
    SharedFullConnection_forwardImplementation(self, self.inmod.outputbuffer[time, self.inSliceFrom:self.inSliceTo],
                                         self.outmod.inputbuffer[desttime, self.outSliceFrom:self.outSliceTo])

                                         
def SharedFullConnection_forwardImplementation(self, ndarray inbuf, ndarray outbuf):
    # replaces: outbuf += dot(reshape(self.params, (self.outdim, self.indim)), inbuf)
    cdef ndarray params
    cdef double * paramp 
    cdef double * inbufp 
    cdef double * outbufp
    cdef int co
    cdef int ci
    cdef int outdim
    cdef int indim
    cdef int buf_len 
    params = self.mother._params
    paramp = <double*> params.data
    outbufp = <double*> outbuf.data
    indim = self.indim
    outdim = self.outdim
    for co from 0 <= co < outdim:
        inbufp = <double*> inbuf.data    
        for ci from 0 <= ci < indim:
            outbufp[0] = outbufp[0] +  inbufp[0] * paramp[0]
            inbufp = inbufp + 1
            paramp = paramp + 1
        outbufp = outbufp + 1
        