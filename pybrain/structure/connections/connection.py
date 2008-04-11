__author__ = 'Daan Wierstra and Tom Schaul'

from pybrain.structure.parametercontainer import ParameterContainer
from pybrain.utilities import Named, abstractMethod
from pybrain.structure.moduleslice import ModuleSlice


class Connection(ParameterContainer, Named):
    """ a connection links 2 modules, more precisely: the output of the first module
    to the input of the second. It can potentially transform the information on the way. 
    It also transmits errors backwards between the same modules. """
    
    inmod = None
    outmod = None
    
    def __init__(self, inmod, outmod, name = None,
                 inSliceFrom = 0, inSliceTo = None, outSliceFrom = 0, outSliceTo = None):
        """ every connection requires an input and an output module. Optionally, it is possible to define slices on the buffers.
            @param inmod: input module
            @param outmod: output module
            @param inSliceFrom: starting index on the buffer of inmod (default = 0)
            @param inSliceTo: ending index on the buffer of inmod (default = last)
            @param outSliceFrom: starting index on the buffer of outmod (default = 0)
            @param outSliceTo: ending index on the buffer of outmod (default = last)
        """        
        self._name = name
        self.inSliceFrom = inSliceFrom
        self.outSliceFrom = outSliceFrom
        if inSliceTo:
            self.inSliceTo = inSliceTo
        else:
            self.inSliceTo = inmod.outdim
        if outSliceTo:
            self.outSliceTo = outSliceTo
        else:
            self.outSliceTo = outmod.indim
            
        if isinstance(inmod, ModuleSlice):
            self.inmod = inmod.base
            self.inSliceFrom += inmod.outOffset
            self.inSliceTo += inmod.outOffset
        else:
            self.inmod = inmod
        self.setArgs(inmod = self.inmod)
        
        if isinstance(outmod, ModuleSlice):
            self.outmod = outmod.base
            self.outSliceFrom += outmod.inOffset
            self.outSliceTo += outmod.inOffset
        else:
            self.outmod = outmod
        self.setArgs(outmod = self.outmod)
                    
        self.indim = self.inSliceTo - self.inSliceFrom 
        self.outdim = self.outSliceTo - self.outSliceFrom 
        
        # arguments for for xml
        if self.inSliceFrom > 0:
            self.setArgs(inSliceFrom = self.inSliceFrom)
        if self.outSliceFrom > 0:
            self.setArgs(outSliceFrom = self.outSliceFrom)
        if self.inSliceTo < self.inmod.outdim:
            self.setArgs(inSliceTo = self.inSliceTo)
        if self.outSliceTo < self.outmod.indim:
            self.setArgs(outSliceTo = self.outSliceTo)
        
        
    def forward(self, time, desttime = None):
        """ propagate the information from the incoming module's output buffer, adding it to 
            the outgoing node's input buffer, and possibly transforming it on the way. 
            @param time: timestep used for inmod
            @param desttime: timestep used for outmod (default = time)
        """
        if not desttime:
            desttime = time
        self._forwardImplementation(self.inmod.outputbuffer[time, self.inSliceFrom:self.inSliceTo],
                                    self.outmod.inputbuffer[desttime, self.outSliceFrom:self.outSliceTo])

    def backward(self, time, desttime = None):
        """ propagate the error found at the outgoing module, adding it to the incoming module' 
            output-error buffer and doing the inverse transformation of forward propagation. 
            If appropriate, also compute the parameter derivatives.
            @param time: timestep used for inmod
            @param desttime: timestep used for outmod (default = time)
        """
        if not desttime:
            desttime = time
        self._backwardImplementation(self.outmod.inputerror[desttime, self.outSliceFrom:self.outSliceTo],
                                     self.inmod.outputerror[time, self.inSliceFrom:self.inSliceTo],
                                     self.inmod.outputbuffer[time, self.inSliceFrom:self.inSliceTo])

    def _forwardImplementation(self, inbuf, outbuf):
        abstractMethod()
    
    def _backwardImplementation(self, outerr, inerr, inbuf):
        abstractMethod()

    def __repr__(self):
        """ A simple representation (this should probably be expanded by subclasses). """
        params = {
            'class': self.__class__.__name__,
            'name': self.name,
            'inmod': self.inmod.name,
            'outmod': self.outmod.name
        }
        return "<%(class)s '%(name)s': '%(inmod)s' -> '%(outmod)s'>" % params
