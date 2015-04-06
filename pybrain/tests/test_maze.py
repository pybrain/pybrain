import sys

import numpy as np

from pybrain.rl.environments.mazes import Maze, MDPMazeTask
from pybrain.rl.learners.valuebased import ActionValueTable
from pybrain.rl.agents import LearningAgent
from pybrain.rl.learners import Q  # , SARSA # (State-Action-Reward-State-Action)
from pybrain.rl.experiments import Experiment

def test_maze():
    # simplified version of the reinforcement learning tutorial example
    structure = np.array([[1, 1, 1, 1, 1],
                          [1, 0, 0, 0, 1],
                          [1, 0, 1, 0, 1],
                          [1, 0, 1, 0, 1],
                          [1, 1, 1, 1, 1]])
    shape = np.array(structure.shape)
    environment = Maze(structure,  tuple(shape - 2))
    controller = ActionValueTable(shape.prod(), 4)
    controller.initialize(1.)
    learner = Q()
    agent = LearningAgent(controller, learner)
    task = MDPMazeTask(environment)
    experiment = Experiment(task, agent)

    for i in range(30):
        experiment.doInteractions(30)
        agent.learn()
        agent.reset()

    controller.params.reshape(shape.prod(), 4).max(1).reshape(*shape)
    # (0, 0) is upper left and (0, N) is upper right, so flip matrix upside down to match NESW action order 
    greedy_policy = np.argmax(controller.params.reshape(shape.prod(), 4),1)
    greedy_policy = np.flipud(np.array(list('NESW'))[greedy_policy].reshape(shape))
    maze = np.flipud(np.array(list(' #'))[structure])
    print('Maze map:')
    print('\n'.join(''.join(row) for row in maze))
    print('Greedy policy:')
    print('\n'.join(''.join(row) for row in greedy_policy))
    assert '\n'.join(''.join(row) for row in greedy_policy) == 'NNNNN\nNSNNN\nNSNNN\nNEENN\nNNNNN'


if __name__ == "__main__":
    try:
        test_maze()
    except:
        from traceback import format_exc
        sys.exit(format_exc())


