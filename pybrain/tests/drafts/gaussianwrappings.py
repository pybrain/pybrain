from pybrain import GaussianLayer, Network, LinearLayer, SigmoidLayer, FullConnection, IdentityConnection
from pybrain.tools.shortcuts import buildSimpleNetwork


gauss = GaussianLayer(2)

variant = 3

if variant == 1:
    # removing an output module
    ffn = buildSimpleNetwork(2, 3, 2)    
    out = ffn.outmodules.pop()
elif variant == 2:
    # all in one network
    ffn = Network()
    ffn.addInputModule(LinearLayer(2, name = 'in'))
    ffn.addModule(SigmoidLayer(3, name = 'h'))
    out = LinearLayer(2, name = 'out')
    ffn.addModule(out)
    ffn.addConnection(FullConnection(ffn['in'], ffn['h']))
    ffn.addConnection(FullConnection(ffn['h'], ffn['out']))
else:
    # nested networks
    out = buildSimpleNetwork(2, 3, 2)  
    ffn = Network()
    ffn.addInputModule(out)
    
ffn.addOutputModule(gauss)
ffn.addConnection(IdentityConnection(out, gauss))

ffn.sortModules()

ffn.activate([4, 3])
ffn.backward()

print "ffn:", ffn.getDerivatives()
print "gauss:", gauss.getDerivatives()

print ffn
