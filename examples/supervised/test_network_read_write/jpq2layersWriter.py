from __future__ import print_function

from pybrain.structure import FeedForwardNetwork
from pybrain.structure           import LinearLayer, SigmoidLayer
from pybrain.structure           import BiasUnit,TanhLayer
from pybrain.structure           import FullConnection
from pybrain.datasets            import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer, RPropMinusTrainer
from pybrain.tools.validation    import ModuleValidator,Validator
from pybrain.utilities           import percentError
from pybrain.tools.customxml     import NetworkWriter
import numpy
import pylab
import os

def myplot(trns,ctrns,tsts = None,ctsts = None,iter = 0):
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
  if tstsplot:
    pylab.plot(tsts['input'],tsts['target'],c='b')
  pylab.scatter(trns['input'],trns['target'],c='r')
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
n=FeedForwardNetwork(name=netname)
inLayer = LinearLayer(1,name='in')
hiddenLayer = SigmoidLayer(nneuron,name='hidden0')
outLayer = LinearLayer(1,name='out')
biasinUnit = BiasUnit(name="bhidden0")
biasoutUnit = BiasUnit(name="bout")
n.addInputModule(inLayer)
n.addModule(hiddenLayer)
n.addModule(biasinUnit)
n.addModule(biasoutUnit)
n.addOutputModule(outLayer)
in_to_hidden = FullConnection(inLayer,hiddenLayer)
bias_to_hidden = FullConnection(biasinUnit,hiddenLayer)
bias_to_out = FullConnection(biasoutUnit,outLayer)
hidden_to_out = FullConnection(hiddenLayer,outLayer)
n.addConnection(in_to_hidden)
n.addConnection(bias_to_hidden)
n.addConnection(bias_to_out)
n.addConnection(hidden_to_out)

n.sortModules()
n.reset()

#read the initail weight values from myparam2.txt
filetoopen = os.path.join(os.getcwd(),'myparam2.txt')
if os.path.isfile(filetoopen):
  myfile = open('myparam2.txt','r')
  c=[]
  for line in myfile:
    c.append(float(line))
  n._setParameters(c)
else:
  myfile = open('myparam2.txt','w')
  for i in n.params:
    myfile.write(str(i)+'\n')
myfile.close()

#activate the neural networks
act = SupervisedDataSet(1,1)
act.addSample((0.2,),(0.880422606518061,))
n.activateOnDataset(act)
#create the test DataSet
x = numpy.arange(0.0, 1.0+0.01, 0.01)
s = 0.5+0.4*numpy.sin(2*numpy.pi*x)
tsts = SupervisedDataSet(1,1)
tsts.setField('input',x.reshape(len(x),1))
tsts.setField('target',s.reshape(len(s),1))

#read the train DataSet from file
trndata = SupervisedDataSet.loadFromFile(os.path.join(os.getcwd(),'trndata'))

#create the trainer

t = BackpropTrainer(n, learningrate = 0.01 ,
                    momentum = mom)
#train the neural network from the train DataSet

cterrori=1.0
print("trainer momentum:"+str(mom))
for iter in range(25):
  t.trainOnDataset(trndata, 1000)
  ctrndata = mv.calculateModuleOutput(n,trndata)
  cterr = v.MSE(ctrndata,trndata['target'])
  relerr = abs(cterr-cterrori)
  cterrori = cterr
  print('iteration:',iter+1,'MSE error:',cterr)
  myplot(trndata,ctrndata,iter=iter+1)
  if cterr < 1.e-5 or relerr < 1.e-7:
    break
#write the network using xml file     
myneuralnet = os.path.join(os.getcwd(),'myneuralnet.xml')
if os.path.isfile(myneuralnet):
    NetworkWriter.appendToFile(n,myneuralnet)
else:
    NetworkWriter.writeToFile(n,myneuralnet)
    
#calculate the test DataSet based on the trained Neural Network
ctsts = mv.calculateModuleOutput(n,tsts)
tserr = v.MSE(ctsts,tsts['target'])
print('MSE error on TSTS:',tserr)
myplot(trndata,ctrndata,tsts,ctsts)

pylab.show()
