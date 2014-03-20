__author__ = 'Tom Schaul and Thomas Rueckstiess'


from itertools import chain
import logging
from sys import exit as errorexit
from pybrain.structure.networks.feedforward import FeedForwardNetwork
from pybrain.structure.networks.recurrent import RecurrentNetwork
from pybrain.structure.modules import BiasUnit, SigmoidLayer, LinearLayer, LSTMLayer
from pybrain.structure.connections import FullConnection, IdentityConnection

try:
    from arac.pybrainbridge import _RecurrentNetwork, _FeedForwardNetwork
except ImportError as e:
    logging.info("No fast networks available: %s" % e)


class NetworkError(Exception): pass


def buildNetwork(*layers, **options):
    """Build arbitrarily deep networks.

    `layers` should be a list or tuple of integers, that indicate how many
    neurons the layers should have. `bias` and `outputbias` are flags to
    indicate whether the network should have the corresponding biases; both
    default to True.

    To adjust the classes for the layers use the `hiddenclass` and  `outclass`
    parameters, which expect a subclass of :class:`NeuronLayer`.

    If the `recurrent` flag is set, a :class:`RecurrentNetwork` will be created,
    otherwise a :class:`FeedForwardNetwork`.

    If the `fast` flag is set, faster arac networks will be used instead of the
    pybrain implementations."""
    # options
    opt = {'bias': True,
           'hiddenclass': SigmoidLayer,
           'outclass': LinearLayer,
           'outputbias': True,
           'peepholes': False,
           'recurrent': False,
           'fast': False,
    }
    for key in options:
        if key not in opt.keys():
            raise NetworkError('buildNetwork unknown option: %s' % key)
        opt[key] = options[key]

    if len(layers) < 2:
        raise NetworkError('buildNetwork needs 2 arguments for input and output layers at least.')

    # Bind the right class to the Network name
    network_map = {
        (False, False): FeedForwardNetwork,
        (True, False): RecurrentNetwork,
    }
    try:
        network_map[(False, True)] = _FeedForwardNetwork
        network_map[(True, True)] = _RecurrentNetwork
    except NameError:
        if opt['fast']:
            raise NetworkError("No fast networks available.")
    if opt['hiddenclass'].sequential or opt['outclass'].sequential:
        if not opt['recurrent']:
            # CHECKME: a warning here?
            opt['recurrent'] = True
    Network = network_map[opt['recurrent'], opt['fast']]
    n = Network()
    # linear input layer
    n.addInputModule(LinearLayer(layers[0], name='in'))
    # output layer of type 'outclass'
    n.addOutputModule(opt['outclass'](layers[-1], name='out'))
    if opt['bias']:
        # add bias module and connection to out module, if desired
        n.addModule(BiasUnit(name='bias'))
        if opt['outputbias']:
            n.addConnection(FullConnection(n['bias'], n['out']))
    # arbitrary number of hidden layers of type 'hiddenclass'
    for i, num in enumerate(layers[1:-1]):
        layername = 'hidden%i' % i
        if issubclass(opt['hiddenclass'], LSTMLayer):
            n.addModule(opt['hiddenclass'](num, peepholes=opt['peepholes'], name=layername))
        else:
            n.addModule(opt['hiddenclass'](num, name=layername))
        if opt['bias']:
            # also connect all the layers with the bias
            n.addConnection(FullConnection(n['bias'], n[layername]))
    # connections between hidden layers
    for i in range(len(layers) - 3):
        n.addConnection(FullConnection(n['hidden%i' % i], n['hidden%i' % (i + 1)]))
    # other connections
    if len(layers) == 2:
        # flat network, connection from in to out
        n.addConnection(FullConnection(n['in'], n['out']))
    else:
        # network with hidden layer(s), connections from in to first hidden and last hidden to out
        n.addConnection(FullConnection(n['in'], n['hidden0']))
        n.addConnection(FullConnection(n['hidden%i' % (len(layers) - 3)], n['out']))

    # recurrent connections
    if issubclass(opt['hiddenclass'], LSTMLayer):
        if len(layers) > 3:
            errorexit("LSTM networks with > 1 hidden layers are not supported!")
        n.addRecurrentConnection(FullConnection(n['hidden0'], n['hidden0']))

    n.sortModules()
    return n


def _buildNetwork(*layers, **options):
    """This is a helper function to create different kinds of networks.

    `layers` is a list of tuples. Each tuple can contain an arbitrary number of
    layers, each being connected to the next one with IdentityConnections. Due
    to this, all layers have to have the same dimension. We call these tuples
    'parts.'

    Afterwards, the last layer of one tuple is connected to the first layer of
    the following tuple by a FullConnection.

    If the keyword argument bias is given, BiasUnits are added additionally with
    every FullConnection.

    Example:

        _buildNetwork(
            (LinearLayer(3),),
            (SigmoidLayer(4), GaussianLayer(4)),
            (SigmoidLayer(3),),
        )
    """
    bias = options['bias'] if 'bias' in options else False

    net = FeedForwardNetwork()
    layerParts = iter(layers)
    firstPart = iter(layerParts.next())
    firstLayer = firstPart.next()
    net.addInputModule(firstLayer)

    prevLayer = firstLayer

    for part in chain(firstPart, layerParts):
        new_part = True
        for layer in part:
            net.addModule(layer)
            # Pick class depending on whether we entered a new part
            if new_part:
                ConnectionClass = FullConnection
                if bias:
                    biasUnit = BiasUnit('BiasUnit for %s' % layer.name)
                    net.addModule(biasUnit)
                    net.addConnection(FullConnection(biasUnit, layer))
            else:
                ConnectionClass = IdentityConnection
            new_part = False
            conn = ConnectionClass(prevLayer, layer)
            net.addConnection(conn)
            prevLayer = layer
    net.addOutputModule(layer)
    net.sortModules()
    return net


