Simple neural network to test the NetworkWriter and NetworkReader modules

The Neural network consists of 5 Sigmoid neurons
the function to train the network against is

y = 0.5 + 0.4*sin(2.0*pi*x)

for x in the range [0,1]

The file param2.txt if it exists in the folder contains the initial values of
the network.params

the train data set is read from trndata file


First execute jpq2layersWriter.py to write the neural network after convergence
 -The trainer is looping till convergence with a maximum of 25 iterations and
  at each iteration a plot is save in the plot directory
 -The train network is written on a xml file: myneuralnet.xml
 -the test data set is calculated based on the trained Neural Network
 -the train data set and test data set are then plot with the indice 0

Second execute jpq2layerReader.py

