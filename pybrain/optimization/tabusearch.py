from pybrain.optimization.optimizer import TabuOptimizer
from pybrain.optimization.hillclimber import HillClimber
from pybrain.optimization.randomsearch import RandomSearch
class TabuHillClimber(TabuOptimizer, HillClimber):
    """Applies the tabu proccess in addition to a hill climbing search."""
class TabuRandomSearch(TabuOptimizer, RandomSearch):
    """Applies the tabu proccess in addition to a random search."""
