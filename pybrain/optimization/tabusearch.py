from pybrain.optimization.optimizer import TabuOptimizer
from pybrain.optimization.hillclimber import HillClimber

class TabuHillClimber(TabuOptimizer, HillClimber):
    """Applies the tabu proccess in addition to a hill climbing search."""
