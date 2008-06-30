""" Test the symmetry handling of a capturegamenetwork. """

__author__ = 'Tom Schaul, tom@idsia.ch'

# TODO: transform into a unittest.

from scipy import zeros, ravel, randn

from pybrain.structure.networks.custom import CaptureGameNetwork

size = 6
hsize = 5

net = CaptureGameNetwork(size = size, hsize = hsize, simpleborders = True)


print 'all params:', net.params
net._printPredefined()

input = zeros((size,size,2))

""" An empty input should give a symmetrical output, with respect to the borders. """
x = net.activate(ravel(input))
x.resize(size,size)
print x
print
""" test the rotational symmetry: 
rotating the input by 90 degrees should rotate the output the same way. """
r = randn(size)
inp1 = input.copy()
inp1[0,0:size,0] += r
x1 = net.activate(ravel(inp1))
x1.resize(size,size)

# now the rotated:
inp2 = input.copy()
inp2[0:size,0,0] += r[::-1]
x2 = net.activate(ravel(inp2))
x2.resize(size,size)


#print inp1[:,:,0]
print x1
print
#print inp2[:,:,0]
print x2
print 

print x1[0, :]
print '='
print x2[:, 0][::-1]
print

""" verify translational symmetries: if the borderconnections ar zero, 
a presented pattern should give the same result, independently of the location. """

net.predefined['borderconn']._params *= 0
inp3 = input.copy()
inp3[1,1] += 10
x3 = net.activate(ravel(inp3))
x3.resize(size,size)
inp4 = input.copy()
inp4[2,2] += 10
x4 = net.activate(ravel(inp4))
x4.resize(size,size)
print x3[:-1,:-1]
print '='
print x4[1:,1:]


""" and the latter should be true even if the network is rescaled. """
dsize = size*2
inp5 = zeros((dsize, dsize, 2))
inp5[size,size] += 10
net2 = net.resizedTo(dsize)

x5 = net2.activate(ravel(inp5))
x5.resize(dsize,dsize)

print '='
print x5[size-1:dsize-2,size-1:dsize-2]
