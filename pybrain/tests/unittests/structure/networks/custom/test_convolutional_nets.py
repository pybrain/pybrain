"""

Let's build a convolutional network designed for board games:
    >>> from pybrain.structure.networks.custom.convboard import ConvolutionalBoardNetwork
    >>> from scipy import array, ravel, var
    >>> N = ConvolutionalBoardNetwork(4, 3, 5)
    >>> print(N.paramdim)
    97

This is what a typical input would look like (on a 4x4 board)

    >>> input = [[[0,0],[0,0],[0,0],[0,0]],\
                 [[0,0],[0,0],[0,0],[0,0]],\
                 [[0,0],[1,1],[0,0],[0,1]],\
                 [[0,0],[1,0],[1,1],[0,1]],\
                 ]

We let the network process the input:

    >>> res = N.activate(ravel(array(input)))
    >>> res = res.reshape(4,4)
    >>> inp =  N['pad'].inputbuffer[0].reshape(6,6,2)[:,:,0]

The input of the first features (e.g. white stone presence) is in the middle, like we set it:

    >>> print(inp[1:5,1:5])
    [[ 0.  0.  0.  0.]
     [ 0.  0.  0.  0.]
     [ 0.  1.  0.  0.]
     [ 0.  1.  1.  0.]]

The rest of that array has been padded with an arbitrary, but identical bias weight:

    >>> var(inp[0,:]) < 1e-20
    True

    >>> inp[0,0] != 0.0
    True

On the output, all the values should be distinct, except for two in the middle above
because a cluster-size of 3x3 makes their input look identical.

    The prior output was 0.0. Need to test this differently due to
    print precision differences in 2 and 3.
    >>> import sys
    >>> res[0,1] - res[0,2] < sys.float_info.epsilon
    True

    >>> res[0,1] == res[0,3]
    False

    >>> res[1,1] == res[0,0]
    False

    >>> res[0,2] == res[3,2]
    False

Now let's use the network, and play a game with it:

    >>> from pybrain.rl.environments.twoplayergames import CaptureGameTask
    >>> t = CaptureGameTask(4)
    >>> tmp = t(N)

"""

__author__ = 'Tom Schaul, tom@idsia.ch'

from pybrain.tests import runModuleTestSuite

if __name__ == '__main__':
    runModuleTestSuite(__import__('__main__'))








