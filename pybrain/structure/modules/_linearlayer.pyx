from pyrex_header cimport ndarray, import_array
import_array()


def LinearLayer_forwardImplementation(self, ndarray inbuf, ndarray outbuf):
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
        outbufp[0] = inbufp[0]
        outbufp = outbufp + 1
        inbufp = inbufp + 1
    

def LinearLayer_backwardImplementation(self, ndarray outerr, ndarray inerr, outbuf, inbuf):
	# typecast: buf.data is a double pointer (not char*)
    cdef double * inerrp 
    inerrp = <double*> inerr.data
    cdef double * outerrp
    outerrp = <double*> outerr.data
    cdef int c
    cdef int buf_len 
    buf_len = inerr.dimensions[0]
    for c from 0 <= c < buf_len:
    	# dereferencing
        inerrp[0] = outerrp[0]
        outerrp = outerrp + 1
        inerrp = inerrp + 1        
    	
    	
def LinearLayerforward(self, time = None):
    if time == None:
        time = self.time
    LinearLayer_forwardImplementation(self, self.inputbuffer[time], self.outputbuffer[time])
    self.time = time + 1
    if time >= self.seqlen:
        self.seqlen = self.time
    if time + 1 >= self.outputbuffer.shape[0]:
        self._growBuffers()
    
    
def LinearLayerbackward(self, time = None):
    if time == None:
        time = self.time 
        time = time - 1
    self.time = time
    LinearLayer_backwardImplementation(self, self.outputerror[time], self.inputerror[time],0,0) 
