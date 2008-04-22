from pyrex_header cimport ndarray, import_array
import_array()

def FullConnection_forwardImplementation(self, ndarray inbuf, ndarray outbuf):
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
    params = self.params
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
        
def FullConnection_backwardImplementation(self, ndarray outerr, ndarray inerr, ndarray inbuf):
    # Replace: 
    #          inerr += dot(reshape(self.params, (self.outdim, self.indim)).T, outerr)
    #          ds = self.derivs
    #          ds += outer(inbuf, outerr).T.flatten()
    cdef ndarray params
    cdef ndarray derivs
    cdef double * paramp 
    cdef double * derivsp 
    cdef double * inbufp 
    cdef double * inerrp 
    cdef double * outerrp
    cdef int co
    cdef int ci
    cdef int outdim
    cdef int indim
    cdef int buf_len 
    indim = self.indim
    outdim = self.outdim

    params = self.params
    paramp = <double*> params.data
    outerrp = <double*> outerr.data    
    for co from 0 <= co < outdim:
        inerrp = <double*> inerr.data    
        for ci from 0 <= ci < indim:
            inerrp[0] = inerrp[0] + outerrp[0] * paramp[0]
            inerrp = inerrp + 1 
            paramp = paramp + 1
        outerrp = outerrp + 1

    derivs = self.derivs
    derivsp = <double*> derivs.data
    outerrp = <double*> outerr.data    
    for co from 0 <= co < outdim:
        inbufp = <double*> inbuf.data    
        for ci from 0 <= ci < indim:
            derivsp[0] = derivsp[0] + outerrp[0] * inbufp[0]
            inbufp = inbufp + 1
            derivsp = derivsp + 1
        outerrp = outerrp + 1 
