# try importing the external dependencies
try:
    from matplotlib.mlab import rk4
except ImportError:
    raise ImportError('This environment needs the matplotlib library installed.')

from pybrain.rl.environments.cartpole.cartpole import CartPoleEnvironment, CartPoleLinEnvironment
from pybrain.rl.environments.cartpole.renderer import CartPoleRenderer
from pybrain.rl.environments.cartpole.balancetask import BalanceTask, EasyBalanceTask, DiscreteBalanceTask, DiscreteNoHelpTask, JustBalanceTask, LinearizedBalanceTask, DiscretePOMDPTask
from pybrain.rl.environments.cartpole.doublepole import DoublePoleEnvironment
from pybrain.rl.environments.cartpole.nonmarkovpole import NonMarkovPoleEnvironment
from pybrain.rl.environments.cartpole.nonmarkovdoublepole import NonMarkovDoublePoleEnvironment