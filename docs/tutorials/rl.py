############################################################################
# PyBrain Tutorial "Reinforcement Learning - CartPole"
# 
# Author: Thomas Rueckstiess, ruecksti@in.tum.de
############################################################################

"""
A reinforcement learning (RL) task in pybrain always consists of a few
components that interact with each other: Environment, Agent, Task, and
Experiment. We will go through each of them, create the instance and explain
what they do.

But first of all, we need to import the RL components from pybrain:
""" 
from pybrain.rl import PolicyGradientAgent, ENAC
from pybrain.rl.environments.cartpole import BalanceTask, CartPoleEnvironment
from pybrain import buildNetwork

"""
The Environment is the world, in which the agent acts. It receives input
with .performAction() and returns an output with .getSensors(). All
environments in PyBrain are located under rl/environments. One of these 
environments is the cart-pole balancing, which we will use for this tutorial.
Its states consist of cart position, cart velocity, pole angle, and
pole angular velocity. It can receive one scalar value, the force with which
the cart is pushed, either to the left (negative value) or right (positive).
Let's create the instance.
"""
environment = CartPoleEnvironment()

"""
Next, we need an agent. The agent is where the learning happens. It can
interact with the environment with its .getAction() and .integrateObservation()
methods. For continuous problems, like the CartPole, we need a policy gradient
agent. Each agent needs a controller, that maps the current state to an action.
We will use a linear controller, which can be created in PyBrain with the 
buildNetwork() shortcut function. We need 4 inputs and 1 output. 
Each agent also has a learner component. There are several learners
for policy gradient agents, which we won't cover in this tutorial. Let's just
use the ENAC learning algorithm now and create the agent.
"""
controller = buildNetwork(4, 1, bias=False)
agent = PolicyGradientAgent(controller, ENAC())

"""
So far, there is no connection between the agent and the environment. In fact,
in PyBrain, there is one component that connects environment and agent: the
task. A task also specifies what the goal is in an environment and how the
agent is rewarded for its actions. For episodic experiments, the Task also
decides when an episode is over. Environments usually bring along their own
tasks. The CartPoleEnvironment for example has a BalanceTask, that we will use.
"""
task = BalanceTask()


"""... to be continued """