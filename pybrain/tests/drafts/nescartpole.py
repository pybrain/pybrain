from pybrain.rl.environments.functions.episodicevaluators import CartPoleEvaluator
from pybrain.tools.shortcuts import buildSimpleNetwork
from pybrain.rl.learners.blackboxoptimizers import NaturalEvolutionStrategies, CMAES


f = CartPoleEvaluator(buildSimpleNetwork(4, 1, 1, True))

n = NaturalEvolutionStrategies(f, lr = 0.0001, ranking = 'smooth')
c = CMAES(f, silent = False)
print n.optimize()
