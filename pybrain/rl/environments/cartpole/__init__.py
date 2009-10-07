# try importing the external dependencies
try:
    from matplotlib.mlab import rk4 
except ImportError:
    raise ImportError('This environment needs the matplotlib library installed.')

from cartpole import CartPoleEnvironment, CartPoleLinEnvironment
from renderer import CartPoleRenderer
from balancetask import BalanceTask, EasyBalanceTask, DiscreteBalanceTask, DiscreteNoHelpTask, JustBalanceTask, LinearizedBalanceTask
from doublepole import DoublePoleEnvironment
from nonmarkovpole import NonMarkovPoleEnvironment
from nonmarkovdoublepole import NonMarkovDoublePoleEnvironment