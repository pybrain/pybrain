from __future__ import print_function

from pybrain.structure import FeedForwardNetwork
from pybrain.tools.validation    import ModuleValidator,Validator
from pybrain.utilities           import percentError
from pybrain.tools.customxml     import NetworkReader
from pybrain.datasets            import SupervisedDataSet
import numpy
import pylab
import os

def myplot(trns,ctrns = None,tsts = None,ctsts = None,iter = 0):
  plotdir = os.path.join(os.getcwd(),'plot')
  pylab.clf()
  try:
    assert len(tsts) > 1
    tstsplot = True
  except:
    tstsplot = False
  try:
    assert len(ctsts) > 1
    ctstsplot = True
  except:
    ctstsplot = False
  try:
    assert len(ctrns) > 1
    ctrnsplot = True
  except:
    ctrnsplot = False
  if tstsplot:
    pylab.plot(tsts['input'],tsts['target'],c='b')
  pylab.scatter(trns['input'],trns['target'],c='r')
  if ctrnsplot:
    pylab.scatter(trns['input'],ctrns,c='y')
  if tstsplot and ctstsplot:
    pylab.plot(tsts['input'], ctsts,c='g')
 
  pylab.xlabel('x')
  pylab.ylabel('y')
  pylab.title('Neuron Number:'+str(nneuron))
  pylab.grid(True)
  plotname = os.path.join(plotdir,('jpq2layers_plot'+ str(iter)))
  pylab.savefig(plotname)


# set-up the neural network
nneuron = 5
mom = 0.98
netname="LSL-"+str(nneuron)+"-"+str(mom)
mv=ModuleValidator()
v = Validator()


#create the test DataSet
x = numpy.arange(0.0, 1.0+0.01, 0.01)
s = 0.5+0.4*numpy.sin(2*numpy.pi*x)
tsts = SupervisedDataSet(1,1)
tsts.setField('input',x.reshape(len(x),1))
tsts.setField('target',s.reshape(len(s),1))
#read the train DataSet from file
trndata = SupervisedDataSet.loadFromFile(os.path.join(os.getcwd(),'trndata'))

myneuralnet = os.path.join(os.getcwd(),'myneuralnet.xml')
if os.path.isfile(myneuralnet):
  n = NetworkReader.readFrom(myneuralnet,name=netname)
  #calculate the test DataSet based on the trained Neural Network
  ctsts = mv.calculateModuleOutput(n,tsts)
  tserr = v.MSE(ctsts,tsts['target'])
  print('MSE error on TSTS:',tserr)
  myplot(trndata,tsts = tsts,ctsts = ctsts)

  pylab.show()
