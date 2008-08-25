from pyrex_header cimport ndarray, import_array
import_array()


def Network_forwardImplementation(self, ndarray inbuf, ndarray outbuf):
# Replaces the following:
#        index = 0
#        t = self.time
#        for m in self.inmodules:
#            m.inputbuffer[t] = inbuf[index:index + m.indim]
#            index += m.indim
#        if t > 0:
#            for c in self.recurrentConns:
#                c.forward(t-1, t)
#        for m in self.modules:
#            m.forward(t)
#            for c in self.connections[m]:
#                c.forward(t)              
#        index = 0
#        for m in self.outmodules:
#            outbuf[index:index + m.outdim] = m.outputbuffer[t]
#            index += m.outdim

        cdef int index 
        cdef int t
        cdef int moutdim
        cdef int mindim

        index = 0
        t = self.time
        for m in self.inmodules:
            moutdim = m.indim
            m.inputbuffer[t] = inbuf[index:index + mindim]
            index = index + mindim
        
        if t > 0:
            for c in self.recurrentConns:
                c.forward(t-1, t)
        
        for m in self.modulesSorted:
            m.forward(t)
            for c in self.connections[m]:
                c.forward(t)
                
        index = 0
        for m in self.outmodules:
            moutdim = m.outdim
            outbuf[index:index + moutdim] = m.outputbuffer[t]
            index = index + moutdim