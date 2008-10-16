#! /usr/bin/env python2.5
# -*- coding: utf-8 -*

"""The structs of arac are defined here in order to be used with ctypes.

Some functionality for bridging arac with pybrain is also included here.
"""


__author__ = 'Justin S Bayer, bayer.justin@googlemail.com'


import collections
import gc

from ctypes import Structure, Union, cast, CDLL, pointer, addressof
from ctypes import c_int, c_double, POINTER, CFUNCTYPE
from itertools import izip

import scipy

from arac.util import is_power_of_two
from arac.lib import libarac
from pybrain.structure import LinearLayer, BiasUnit, SigmoidLayer, LSTMLayer, \
    MDLSTMLayer, IdentityConnection, FullConnection, TanhLayer, SoftmaxLayer, \
    MdrnnLayer
from pybrain.structure.connections.shared import SharedFullConnection
from pybrain.structure.connections.permutation import PermutationConnection
from pybrain.utilities import garbagecollect


c_double_p = POINTER(c_double)
c_int_p = POINTER(c_int)
c_mapfunc_p = CFUNCTYPE(c_double, c_double)


class c_parameter_container(Structure):
    """ctypes representation of the arac ParameterContainer struct."""
        
    _fields_ = [
        ('size', c_int),
        ('contents_p', c_double_p),
        ('error_p', c_double_p),
    ]
    
    def __init__(self, contents=None, errors=None):
        """Create a ParameterContainer struct from two scipy arrays.
        
        Arrays have to be of the same size. Size for the struct is infered by 
        that size."""
        self.size = c_int(0)
        if contents is not None:
            if contents.dtype != scipy.float64:
                raise ValueError("Only float64 arrays allowed.")
            self.size = c_int(scipy.size(contents))
            self.contents_p = contents.ctypes.data_as(c_double_p)
        if errors is not None:
            if errors.dtype != scipy.float64:
                raise ValueError("Only float64 arrays allowed.")
            self.error_p = errors.ctypes.data_as(c_double_p)


class c_identity_layer(Structure):
    """ctypes representation of the arac IdentityLayer struct."""
    
    
class c_sigmoid_layer(Structure):
    """ctypes representation of the arac SigmoidLayer struct."""
    
    
class c_bias_layer(Structure):
    """ctypes representation of the arac BiasLayer struct."""
    
    
class c_mdlstm_layer(Structure):
    """ctypes representation of the arac MdLstmLayer struct."""

    _fields_ = [
        ('timedim', c_int),

        ('peephole_input_weights', c_parameter_container),
        ('peephole_forget_weights', c_parameter_container),
        ('peephole_output_weights', c_parameter_container),

        ('input_squashed_p', c_double_p),
        ('input_gate_squashed_p', c_double_p),
        ('input_gate_unsquashed_p', c_double_p),
        ('output_gate_squashed_p', c_double_p),
        ('output_gate_unsquashed_p', c_double_p),
        ('forget_gate_unsquashed_p', c_double_p),
        ('forget_gate_squashed_p', c_double_p),

        ('gate_squasher', c_mapfunc_p),
        ('gate_squasher_prime', c_mapfunc_p),
        ('cell_squasher', c_mapfunc_p),
        ('cell_squasher_prime', c_mapfunc_p),
        ('output_squasher', c_mapfunc_p),
        ('output_squasher_prime', c_mapfunc_p)
    ]

    def __init__(self):
        self.gate_squasher = cast(libarac.sigmoid, c_mapfunc_p)
        self.gate_squasher_prime = cast(libarac.sigmoid_prime, c_mapfunc_p)
        self.cell_squasher = cast(libarac.tanh, c_mapfunc_p)
        self.cell_squasher_prime = cast(libarac.tanh_prime, c_mapfunc_p)
        self.output_squasher = cast(libarac.tanh_, c_mapfunc_p)
        self.output_squasher_prime = cast(libarac.tanh_prime, c_mapfunc_p)

    
class c_lstm_layer(Structure):
    """ctypes representation of the arac LstmLayer struct."""


class c_softmax_layer(Structure):
    """ctypes representation of the arac SoftmaxLayer struct."""


class c_tanh_layer(Structure):
    """ctypes representation of the arac TanhLayer struct."""
    
    
class c_mdrnn_layer(Structure):
    """ctypes representation of the arac MdrnnLayer struct."""


class c_any_layer(Union):
    """ctypes representation of the arac AnyLayer union."""

    _fields_ = [
        ('bias_layer_p', POINTER(c_bias_layer)),
        ('identity_layer_p', POINTER(c_identity_layer)),
        ('lstm_layer_p', POINTER(c_lstm_layer)),
        ('mdlstm_layer_p', POINTER(c_mdlstm_layer)),
        ('sigmoid_layer_p', POINTER(c_sigmoid_layer)),
        ('softmax_layer_p', POINTER(c_softmax_layer)),
        ('tanh_layer_p', POINTER(c_tanh_layer)),
        ('mdrnn_layer_p', POINTER(c_mdrnn_layer)),
    ]


class c_identity_connection(Structure):
    """ctypes representation of the arac IdentityConnection."""
    

class c_full_connection(Structure):
    """ctypes representation of the arac FullConnection."""
    
    _fields_ = [
        ('weights', c_parameter_container),
    ]

    # FIXME: This is an ugly workaround that keeps objects from being collected.
    registry = []
    def __init__(self):
        self.registry.append(self)
        

class c_permutation_connection(Structure):
    """ctypes representation of the arac PermutationConnection."""
    
    _fields_ = [
        ('permutation_p', c_double_p),
        ('invpermutation_p', c_double_p),
        ('blocksize', c_int)
    ]
    
    
class c_any_connection(Union):
    """ctypes representation of the arac AnyConnection union."""
    
    _fields_ = [
        ('identity_connection_p', POINTER(c_identity_connection)),
        ('full_connection_p', POINTER(c_full_connection)),
        ('permutation_connection_p', POINTER(c_permutation_connection)),
    ]
    
    
class c_layer(Structure):
    """ctypes representation of the arac Layer struct."""
    
    # Mappings from pybrain classes to string identifer which make it possible 
    # to map the classes to the corresponding arac structs.
    layer_mappings = {
        LinearLayer: 'identity',
        SigmoidLayer: 'sigmoid',
        MDLSTMLayer: 'mdlstm',
        LSTMLayer: 'lstm',
        BiasUnit: 'bias',
        TanhLayer: 'tanh',
        SoftmaxLayer: 'softmax',
        MdrnnLayer: 'mdrnn',
    }
    
    def __init__(self, input_dim, output_dim, 
                 input_, output, inerror, outerror):
        """Initialize the representation of a layer struct.
        
        The struct is not fully functional afterwards. You have to set the type
        and the type specific union internal.
        """

        self.inputs = c_parameter_container(input_, inerror)
        self.outputs = c_parameter_container(output, outerror)

        self.outgoing_n = 0
        self.incoming_n = 0
        
    @classmethod
    def from_layer(cls, layer):
        """Return an arac layer from a pybrain layer, both sharing the same 
        data."""
        klass = layer.__class__
        struct = cls(input_dim=layer.indim,
                     output_dim=layer.outdim,
                     input_=layer.inputbuffer[0],
                     output=layer.outputbuffer[0],
                     inerror=layer.inputerror[0],
                     outerror=layer.outputerror[0])
                     
        # We try to retrieve the specific methods for the current layer type. If
        # that does not work, the layer type is not supported; in that case the
        # function raises an exception.
        try:
            name = cls.layer_mappings[klass]
        except KeyError:
            raise ValueError("Unsupported Layer class %s in arac" % klass)
        maker = getattr(struct, 'make_%s_layer' % name, None)
        if not maker:
            raise ValueError("Unknown Layer %s in arac" % name)
        maker(layer)
        return struct
            
    def make_bias_layer(self, layer):
        """Make this a BiasUnit."""
        bias_layer = c_bias_layer()
        self.type = 0
        self.internal.bias_layer_p = pointer(bias_layer)
            
    def make_identity_layer(self, layer):
        """Make this an IdentityLayer."""
        identity_layer = c_identity_layer()
        self.type = 1
        self.internal.identity_layer_p = pointer(identity_layer)
        
    def make_sigmoid_layer(self, layer):
        """Make this a SigmoidLayer."""
        sigmoid_layer = c_sigmoid_layer()
        self.type = 2
        self.internal.sigmoid_layer_p = pointer(sigmoid_layer)

    def make_softmax_layer(self, layer):
        """Make this a SoftmaxLayer."""
        softmax_layer = c_softmax_layer()
        self.type = 6
        self.internal.softmax_layer_p = pointer(softmax_layer)

    def make_tanh_layer(self, layer):
        """Make this a TanhLayer."""
        tanh_layer = c_tanh_layer()
        self.type = 3
        self.internal.tanh_layer_p = pointer(tanh_layer)
        
    def make_lstm_layer(self, layer):
        """Make this an LstmLayer."""
        lstm_layer = c_lstm_layer()
        lstm_layer.states = \
            c_parameter_container(layer.state, layer.stateError)
        self.__lstm_layer = lstm_layer      # TODO: necessary?

        # Create the encapsulated MdLstmLayer and store it in the class, so it
        # is not garbage collected.
        self.__mdlstmlayer = MDLSTMLayer(layer.dim)

        mdlstmlayer = c_layer.from_layer(self.__mdlstmlayer)
        lstm_layer.mdlstm_p = pointer(mdlstmlayer)
        # Set other variables
        self.type = 5
        self.internal.lstm_layer_p = pointer(lstm_layer)
        
    def make_mdlstm_layer(self, layer):
        """Make this an MdLstmLayer."""
        mdlstm_layer = c_mdlstm_layer()
        self.__mdlstm_layer = mdlstm_layer      # TODO: necessary?

        mdlstm_layer.timedim = layer.dimensions
        mdlstm_layer.input_squashed_p = layer.state.ctypes.data_as(c_double_p)
        mdlstm_layer.input_gate_squashed_p = \
            layer.ingate.ctypes.data_as(c_double_p)
        mdlstm_layer.input_gate_unsquashed_p = \
            layer.ingatex.ctypes.data_as(c_double_p)
        mdlstm_layer.output_gate_squashed_p = \
            layer.outgate.ctypes.data_as(c_double_p)
        mdlstm_layer.output_gate_unsquashed_p = \
            layer.outgatex.ctypes.data_as(c_double_p)
        mdlstm_layer.forget_gate_squashed_p = \
            layer.forgetgate.ctypes.data_as(c_double_p)
        mdlstm_layer.forget_gate_unsquashed_p = \
            layer.forgetgatex.ctypes.data_as(c_double_p)

        self.type = 4
        self.internal.mdlstm_layer_p = pointer(mdlstm_layer)
        # TODO: add peephole weights
        # ('peephole_input_weights', c_parameter_container),
        # ('peephole_forget_weights', c_parameter_container),
        # ('peephole_output_weights', c_parameter_container),

    @garbagecollect
    def make_mdrnn_layer(self, layer):
        """Make this an MdrnnLayer."""
        # The internal layers need timesteps for their calculations.
        self._timestep = c_int(0)
        self._timestep_p = pointer(self._timestep)

        mdrnn_layer = c_mdrnn_layer()
        self.internal.mdrnn_layer_p = pointer(mdrnn_layer)
        self.type = 7

        # Initialize several parameters that should not be recalculated every
        # time.
        mdrnn_layer.timedim = layer.timedim
        mdrnn_layer.hiddendim = layer.hiddendim
        mdrnn_layer.shape_p = (c_int * layer.timedim)(*layer.shape)
        mdrnn_layer.blockshape_p = (c_int * layer.timedim)(*layer.blockshape)

        # Set parameters.
        # TODO: Fix this flaw in nomenclature: the arac's word for blocksize is
        # indim,
        self.inputs.size = layer.indim
        self.outputs.size = layer.outdim
        mdrnn_layer.indim = layer.blocksize
        mdrnn_layer.outdim = layer.outsize
        mdrnn_layer.sequence_length = layer.sequenceLength

        # Initialize buffers.
        mdrnn_layer.cell_states_p = layer.cellStates.ctypes.data_as(c_double_p)
        self.outputs.contents_p = layer.outputbuffer.ctypes.data_as(c_double_p)

        # Initialize layers for internal use. These have to be saved in
        # order to be not garbage collected.
        mdrnn_layer.hiddenlayer_p = \
            pointer(c_layer.from_layer(layer.hiddenlayer))
        mdrnn_layer.hiddenlayer_p.contents.timestep_p = self._timestep_p

        mdrnn_layer.outlayer_p = pointer(c_layer.from_layer(layer.outlayer))
        mdrnn_layer.outlayer_p.contents.timestep_p = self._timestep_p

        mdrnn_layer.swipe_inlayer_p = pointer(c_layer.from_layer(layer.inlayer))
        mdrnn_layer.swipe_inlayer_p.contents.timestep_p = self._timestep_p

        predlayers = [c_layer.from_layer(pred) for pred in layer.predlayers]

        mdrnn_layer.bias_p = pointer(c_layer.from_layer(layer.bias))

        # Give the internal layers a timestep - it will be zero all the time,
        # but it will need one anyway.
        for pred in predlayers:
            pred.timestep_p = self._timestep_p

        # Initialize connections between the internal layers.
        # FullConnection from the swiping inlayer to the hiddenlayer. This
        # connection is only partial, as it only connects with the input part
        # of the mdlstm cell.
        con = c_connection(mdrnn_layer.swipe_inlayer_p.contents,
                           mdrnn_layer.hiddenlayer_p.contents)
        con.make_full_connection(params=layer.inParams,
                                 errors=scipy.array((0.)))
        con.inlayerstart = 0
        con.inlayerstop = layer.blocksize + 1   # Exlusive slices
        con.outlayerstart = 0
        con.outlayerstop = layer.hiddendim * (3 + layer.timedim)

        mdrnn_layer.swipe_inlayer_p.contents.add_outgoing_connection(con)

        # FullConnections from the hidden layers to the hidden layers. This is
        # realized by connecting with the bogus predecessing layers.
        for pred, params in izip(predlayers * layer.timedim, layer.predParams):
            con = c_connection(pred, mdrnn_layer.hiddenlayer_p.contents)
            con.make_full_connection(params=params,
                                     errors=scipy.array((0.)))
            con.inlayerstart = 0
            con.inlayerstop = layer.outsize + 1
            con.outlayerstart = 0
            con.outlayerstop = \
                layer.hiddendim * (3 + layer.timedim)
            pred.add_outgoing_connection(con)

        layer_array_type = (c_layer * layer.timedim)
        mdrnn_layer.swipe_predlayers_p = layer_array_type(*predlayers)

        # FullConnection from the hidden layer to the output layer. Only the
        # first half of the output buffer is used.
        con = c_connection(mdrnn_layer.hiddenlayer_p.contents,
                           mdrnn_layer.outlayer_p.contents)
        con.make_full_connection(params=layer.outParams,
                                 errors=scipy.array((0.)))
        con.inlayerstart = 0
        con.inlayerstop = layer.hiddendim + 1
        con.outlayerstart = 0
        con.outlayerstop = layer.outsize + 1
        
        mdrnn_layer.hiddenlayer_p.contents.add_outgoing_connection(con)

        # FullConnection from the bias to the hidden layer.
        biashiddenparams = (3 + layer.timedim) * layer.hiddendim
        biasoutparams = layer.outsize
        con = c_connection(mdrnn_layer.bias_p.contents,
                           mdrnn_layer.hiddenlayer_p.contents)
        con.make_full_connection(params=layer.biasParams[:biashiddenparams])
        con.inlayerstart = 0
        con.inlayerstop = 2
        con.outlayerstart = 0
        con.outlayerstop = biashiddenparams + 1

        mdrnn_layer.bias_p.contents.add_outgoing_connection(con)

        # FullConnection from the bias to the out layer.
        con = c_connection(mdrnn_layer.bias_p.contents,
                           mdrnn_layer.outlayer_p.contents)
        con.make_full_connection(params=layer.biasParams[biashiddenparams:])
        con.inlayerstart = 0
        con.inlayerstop = 2
        con.outlayerstart = 0
        con.outlayerstop = biasoutparams + 1

        mdrnn_layer.bias_p.contents.add_outgoing_connection(con)

    def add_outgoing_connection(self, con):
        """Add a c_connection to this layer as an outgoing connection."""
        if self.outgoing_n == 0:
            typ = c_connection * 1
            self.outgoing_p = typ(con)
        elif is_power_of_two(self.outgoing_n):
            # We have to make a bigger array
            typ = c_connection * (self.outgoing_n * 2)
            self.outgoing_p = \
                typ(*(list(self.outgoing_p[:self.outgoing_n]) + [con]))
        else:
            # Explicitly set a stop in the slice, otherwise it will not stop
            self.outgoing_p[self.outgoing_n] = con
        self.outgoing_n += 1
        
    def add_incoming_connection(self, con):
        """Add a c_connection to this layers as an incoming connection."""
        if self.incoming_n == 0:
            typ = c_connection * 1
            self.incoming_p = typ(con)
        elif is_power_of_two(self.incoming_n):
            # We have to make a bigger array
            typ = c_connection * (self.incoming_n * 2)
            self.incoming_p = \
                typ(*(list(self.incoming_p[:self.incoming_n]) + [con]))
        else:
            # Explicitly set a stop in the slice, otherwise it will not stop
            self.incoming_p[self.incoming_n] = con
        self.incoming_n += 1

    
class c_connection(Structure): 
    """ctypes representation of the arac Connetion struct."""
    
    # Mapping of pybrain classes to arac identifiers which make it possible to
    # map the classes to the corresponding arac structs.
    connection_mappings = {
        IdentityConnection: 'identity',
        FullConnection: 'full',
        SharedFullConnection: 'full',
        PermutationConnection: 'permutation',
    }
    
    def __init__(self, inlayer=None, outlayer=None):
        """Create a c_connection by two c_layers."""
        if inlayer is not None:
            self.inlayer_p = c_layer_p(inlayer)
        if outlayer is not None:
            self.outlayer_p = c_layer_p(outlayer)
        
    @classmethod
    def from_connection(cls, connection, inlayer, outlayer):
        """Return an arac connection from a pybrain connection, both sharing the
        same data."""
        klass = connection.__class__
        struct = cls(inlayer=inlayer,
                     outlayer=outlayer)
        struct.inlayerstart = connection.inSliceFrom
        struct.inlayerstop = connection.inSliceTo
        struct.outlayerstart = connection.outSliceFrom
        struct.outlayerstop = connection.outSliceTo

        # We try to retrieve the specific methods for the current layer type. If
        # that does not work, the layer type is not supported; in that case the
        # function raises an exception.
        try:
            name = cls.connection_mappings[klass]
        except KeyError:
            raise ValueError("Unsupported Connection class %s in arac" % klass)
        maker = getattr(struct, 'make_%s_connection' % name, None)
        if not maker:
            raise ValueError("Unknown Connection %s in arac" % name)
        maker(connection)
        return struct
        
    def make_identity_connection(self, connection=None):
        """Make this connection an identity connection."""
        self.identity_connection = c_identity_connection()
        self.type = 0
        self.internal.identity_connection_p = \
            pointer(self.identity_connection)

    def make_full_connection(self, connection=None, params=None, errors=None):
        """Make this connection a full connection.
        
        The weights for the FullConnection are taken from the passed pybrain
        connection or, if no connection is passed, from the supplied
        params/error arrays.
        """
        self.full_connection = c_full_connection()
        self.type = 1
        self.internal.full_connection_p = pointer(self.full_connection)

        if connection is not None:
            params = connection.params
            errors = connection.derivs
            
        self.full_connection.weights = c_parameter_container(params, errors)
    
    def make_permutation_connection(self, connection):
        """Make this connection a permutation connection."""
        self.permutation_connection = c_permutation_connection()
        self.type = 2
        self.internal.permutation_connection_p = (
            pointer(self.permutation_connection))
        self.permutation_connection.permutation_p = (
            connection.permutation.ctypes.data_as(c_double_p)) 
        self.permutation_connection.invpermutation_p = (
            connection.invpermutation.ctypes.data_as(c_double_p)) 
        self.permutation_connection.blocksize = connection.blocksize


# Some shortcuts.
c_layer_p = POINTER(c_layer)
c_connection_p = POINTER(c_connection)

# Exact fields for the representation of the arac c struct. These have to be set
# later, since the upper class declarations are forward declarations. (Due to 
# cyclic dependencies.)

c_layer._fields_ = [
    ('inputs', c_parameter_container),
    ('outputs', c_parameter_container),
    
    ('incoming_n', c_int),
    ('outgoing_n', c_int),
    ('incoming_p', c_connection_p),
    ('outgoing_p', c_connection_p),
    
    ('type', c_int),
    ('internal', c_any_layer),
    
    ('timestep_p', c_int_p),
    ('seqlen_p', c_int_p),
]


c_connection._fields_ = [
    ('inlayer_p', c_layer_p),
    ('outlayer_p', c_layer_p),

    ('recurrent', c_int),

    ('inlayerstart', c_int),
    ('inlayerstop', c_int),
    ('outlayerstart', c_int),
    ('outlayerstop', c_int),

    ('type', c_int),
    ('internal', c_any_connection),
]


c_lstm_layer._fields_ = [
    ('mdlstm_p', c_layer_p),
    ('states', c_parameter_container),
]


c_mdrnn_layer._fields_ = [
    ('timedim', c_int),
    ('shape_p', c_int_p),
    ('blockshape_p', c_int_p),
    ('sequence_length', c_int),
    ('indim', c_int),
    ('hiddendim', c_int),
    ('outdim', c_int),
    ('cell_states_p', c_double_p),
    ('hiddenlayer_p', c_layer_p),
    ('outlayer_p', c_layer_p),
    ('swipe_inlayer_p', c_layer_p),
    ('swipe_predlayers_p', c_layer_p),
    ('bias_p', c_layer_p),
]

