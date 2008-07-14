from pyrex_header cimport ndarray, import_array, tanh
import_array()


def TanhLayer_forwardImplementation(self, ndarray inbuf, ndarray outbuf):
    cdef double * inbufp 
    inbufp = <double*> inbuf.data
    cdef double * outbufp
    outbufp = <double*> outbuf.data
    cdef int c
    cdef int buf_len 
    buf_len = inbuf.dimensions[0]
    for c from 0 <= c < buf_len:
        outbufp[0] = tanh(inbufp[0])
        outbufp = outbufp + 1
        inbufp = inbufp + 1
    	
def TanhLayerforward(self, time = None):
    if time == None:
        time = self.time
    TanhLayer_forwardImplementation(self, self.inputbuffer[time], self.outputbuffer[time])
    self.time = time + 1
    if time >= self.seqlen:
        self.seqlen = self.time
    if time + 1 >= self.outputbuffer.shape[0]:
        self._growBuffers()