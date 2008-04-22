from pybrain.rl.environments.functions.episodicevaluators import CartPoleEvaluator
from pybrain import buildNetwork
from pybrain.rl.learners.blackboxoptimizers import NaturalEvolutionStrategies, CMAES


f = CartPoleEvaluator(buildNetwork(4, 1, 1))

n = NaturalEvolutionStrategies(f, lr = 0.0001, ranking = 'smooth')
c = CMAES(f, silent = False)
print n.optimize()
