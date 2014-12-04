from __future__ import print_function

"""This module provides functionality to use pybrain with rlglue and to use it
for the rlcompetition.

The whole module has quite a hacky feel, which is due to the fact that
communication with subprocesses is not always easy and less often intended by
the original author.

So make changes with care and don't be surprised if you see some rude lines of
code.
"""


__author__ = 'Justin Bayer, bayerj@in.tum.de'


import exceptions
import logging
import os

from signal import SIGKILL #@UnresolvedImport
from subprocess import Popen, PIPE

from rlglue.agent.ClientAgent import ClientAgent #@UnresolvedImport
from rlglue.network.Network import kRetryTimeout as CLIENT_TIMEOUT #@UnresolvedImport
from rlglue.network.Network import kDefaultPort as DEFAULT_PORT #@UnresolvedImport
from rlglue.network.Network import kLocalHost as DEFAULT_HOST #@UnresolvedImport
from rlglue.types import Action as RLGlueAction #@UnresolvedImport
from scipy import array

from pybrain.structure.modules.module import Module
from pybrain.rl.agents import LearningAgent
from pybrain.utilities import threaded
from pybrain.tools.benchmark import BenchmarkDataSet


class RLGlueError(Exception): pass
class RLCompetitionNotFound(RLGlueError): pass


def adaptAgent(agent_klass):
    """Return a factory function that instantiates a pybrain agent and adapts
    it to the rlglue framework interface.

    :type   agent_klass:    subclass of some pybrain agent
    :key  agent_klass:    Some class that is to be adapted to the rlglue
                            framework

    """
    # TODO: return a real class instead of a function, so docstrings and such
    # are not lost.
    def inner(*args, **kwargs):
        return RlglueAgentAdapter(agent_klass, *args, **kwargs)
    return inner


def adaptAgentObject(agent_object):
    """Return an object that adapts a pybrain agent to the rlglue interface.
    """
    # This is pretty hacky: We first take a bogus agent with a bogus module
    # for our function adaptAgent to work, then substitue the bogus agent with
    # our actual agent.
    agent = adaptAgent(LearningAgent)(Module(1, 1))
    agent.agent = agent_object
    return agent


class RlglueAgentAdapter(object):
    """Wrapper class to use pybrain agents with the RLGlue library."""

    def __init__(self, klass, *args, **kwargs):
        """
        Create an object that adapts an object of class klass to the
        protocol of rlglue agents.

        :type   klass:    subclass of some pybrain agent
        :key  klass:    Some class that is to be adapted to the rlglue
                          framework
        """
        if not issubclass(klass, LearningAgent):
            raise ValueError("Supply a LearningAgent as first argument")

        self.agent = klass(*args, **kwargs)

        # TODO: At the  moment, learning is done after a certain amount of
        # steps - this is somehow logic of the agent, and not of the wrapper
        # Maybe there are some changes in the agent API needed.
        self.learnCycle = 1
        self.episodeCount = 1

    def agent_init(self, task_specification=None):
        """Give the agent a specification of the action and state space.

        Since pybrain agents are not using task specifications
        (they are already set up to the problem domain) the task_specification
        parameter is only there for API consistency, but it will be ignored.

        The specification for the specifications can be found here:
        http://rlai.cs.ualberta.ca/RLBB/TaskSpecification.html

        :type task_specification:   string
        """
        # This is (for now) actually a dummy method to satisfy the
        # RLGlue interface. It is the programmer's job to check wether an
        # experiment fits the agent object.
        self.agent.reset()

    def agent_start(self, firstObservation):
        """
        Return an action depending on the first observation.

        :type firstObservation:     Observation
        """
        self._integrateObservation(firstObservation)
        return self._getAction()

    def agent_step(self, reward, observation):
        """
        Return an action depending on an observation and a reward.

        :type reward:               number
        :type firstObservation:     Observation
        """
        self._giveReward(reward)
        self._integrateObservation(observation)
        return self._getAction()

    def agent_end(self, reward):
        """
        Give the last reward to the agent.

        :type reward: number
        """
        self._giveReward(reward)
        self.agent.newEpisode()
        self.episodeCount += 1
        if self.episodeCount % self.learnCycle == 0:
            self.agent.learn()
            self.agent.reset()

    def agent_cleanup(self):
        """This is called when an episode ends.

        Should be in one ratio to agent_init."""

    def agent_freeze(self):
        """Tell the agent to end training.

        Learning and exploration is stopped."""
        self.agent.disableTraining()

    def agent_message(self, message):
        # Originally thought to enable dynamic methods for agents, but this
        # does not make a lot of sense in a dynamic language (and in OO?)
        print(("Message:", message))

    def _getAction(self):
        """
        Return a RLGlue action that is made out of a numpy array yielded by
        the hold pybrain agent.
        """
        action = RLGlueAction()
        action.doubleArray = self.agent.getAction().tolist()
        action.intArray = []
        return action

    def _integrateObservation(self, observation):
        """
        Take an RLGlue observation and convert it into a numpy array to feed
        it into the pybrain agent.

        :type observation:     Observation
        """
        observation = array(observation.doubleArray)
        self.agent.integrateObservation(observation)

    def _giveReward(self, reward):
        self.agent.giveReward(reward)


class RLCExperiment(object):
    """Class to abstract a subprocess that runs an rl-competition experiment
    on a given port."""

    def __init__(self, path, port=None, autoreconnect=None):
        """Instantiate an object with the given variables."""
        if os.name not in ('posix', 'mac'):
            raise NotImplementedError(
                    "Killing processes under win32 not supported")
        self.path = path
        self.port = port
        self.autoreconnect = autoreconnect
        self.running = False

    def start(self):
        """Start the experiment."""
        self.running = True
        env = {}
        if self.port:
            env['RLGLUE_PORT'] = self.port
        if self.autoreconnect:
            env['RLGLUE_AUTORECONNECT'] = self.autoreconnect

        cwd = self.path[:self.path.rfind("/") + 1]
        self.process = Popen(self.path, env=env, shell=True, cwd=cwd,
                             bufsize=0, stdin=PIPE, stdout=PIPE, stderr=PIPE)

        # We have to fetch some PIDs from the subprocess' output to kill the
        # processes afterwards.
        rlg_pidstr = self.process.stdout.readline()
        de_pidstr = self.process.stdout.readline()
        get_pid = lambda s: int(s[s.find("PID=") + 4:].strip())
        self.rlglue_pid = get_pid(rlg_pidstr)
        self.dynenv_pid = get_pid(de_pidstr)

        logging.info("Environment started (%i, %i. %i)" %
                     (self.process.pid, self.rlglue_pid, self.dynenv_pid))

        # We need to consume the processes standard output and error output,
        # otherwise the program will wait for it to be consumed. This is
        # actually pretty nasty, but it has to be done.
        @threaded(lambda x: None, True)
        def consume(flo):
            while self.running:
                flo.read(512)

        consume(self.process.stdout)
        consume(self.process.stderr)

    def stop(self):
        """Stop the experiment."""
        # There is some bad bad process shooting going on here. It seems as if
        # the rl-competition software has some hickups when the process is
        # started via subprocess and does not always end its childprocesses
        # properly. Luckily, their pids are printed out, and we can grab those
        # and kill the processes.

        # Shoot child processes mercilessly
        self.running = False
        self._killProcess(self.rlglue_pid)
        self._killProcess(self.dynenv_pid)
        self._killProcess(self.process.pid)

        logging.info("Environment ended.")

    def _killProcess(self, pid):
        try:
            os.kill(pid, SIGKILL) #@UndefinedVariable
        except exceptions.OSError:
            # Explicitly silence if the process has already been killed
            pass

    def __del__(self):
        if self.running:
            self.stop()


class RlCompBenchmark(object):
    """Class to run benchmarks of pybrain agents on the rl-competition 2007
    environments.
    """

    port = DEFAULT_PORT
    overwrite = False

    def __init__(self, agents, port=None):
        self.agents = agents
        if port: self.port = port

    def run(self):
        """Run the benchmark. All agents are tested loop times against the
        environment and statistics for each run are saved into the benchmark
        directory benchmarkDir.

        """
        # Create benchmark directory: the desired name plus the current date
        # and time.
        try:
            os.makedirs(self.benchmarkDir)
        except OSError as e:
            if not "File exists" in str(e):
                raise e

        for name, agent_klass in self.agents:
            todo = range(self.loops)
            if not self.overwrite:
                # If overwrite is set to false, we will only do the experiments
                # that have not been done.

                # index gets the index of a benchmark file out of the filename.
                index = lambda x: int(x[x.rfind("-") + 1:])
                done = set(index(i) for i in os.listdir(self.benchmarkDir)
                           if i.startswith("%s-" % name))
                todo = (i for i in todo if i not in done)
            for j in todo:
                logging.info("Starting agent %s's loop #%i" % (name, j + 1))
                # Make a clean copy of the agent for every run
                # Start subprocess that gives us the experiment
                agent = agent_klass()
                stats = self.testAgent(agent)
                # Dump stats to the given directory
                self.saveStats(name + "-%i" % j, stats)

    def testAgent(self, agent):
        """Test an agent once on the experiment and return a benchmark
        dataset."""
        return testAgent(self.path, agent, self.port)

    def saveStats(self, name, dataset):
        """Save the given dataset to a"""
        filename = os.path.join(self.benchmarkDir, name)
        dataset.saveToFile(filename, arraysonly=True)
        logging.info("Saved statistics to %s" % filename)


def testAgent(path, agent, port=DEFAULT_PORT):
    """Test an agent once on a rlcompetition experiment.

    Path specifies the executable file of the rl competition.
    """
    agent = adaptAgentObject(BenchmarkingAgent(agent))

    experiment = RLCExperiment(path, str(port))
    experiment.start()
    # This method is provided by rlglue and makes a client to be runnable
    # over the network.
    clientAgent = ClientAgent(agent)
    clientAgent.connect(DEFAULT_HOST, port, CLIENT_TIMEOUT)
    logging.info("Agent connected")
    clientAgent.runAgentEventLoop()
    clientAgent.close()
    logging.info("Agent finished")
    experiment.stop()

    return agent.agent.benchmark


# This class is defined here and not in benchmarks, since use of it is
# discouraged when not interacting with the rlglue framework. When using
# pybrain environments, other ways should be found.
class BenchmarkingAgent(object):
    """Agent that is used as a middleware to record benchmarks into a
    BenchmarkDataSet.
    """

    def __init__(self, agent):
        """Return a wrapper around the given agent."""
        if hasattr(agent, 'benchmark') or  hasattr(agent, 'agent'):
            raise ValueError("Wrapped agent must not define a benchmark or" +
                             "an agent attribute.")
        self.agent = agent
        self.benchmark = BenchmarkDataSet()

        # For episodewide statistics
        self.__rewards = []

    def giveReward(self, reward):
        self.agent.giveReward(reward)
        self.__rewards.append(reward)

    def newEpisode(self):
        episodeLength = len(self.__rewards)
        avgReward = sum(self.__rewards) / episodeLength
        self.benchmark.appendLinked(avgReward, episodeLength)
        self.__rewards = []
        return self.agent.newEpisode()

    def __getattribute__(self, key):
        try:
            return super(BenchmarkingAgent, self).__getattribute__(key)
        except AttributeError:
            agent = super(BenchmarkingAgent, self).__getattribute__('agent')
            return getattr(agent, key)

    def __setattribute__(self, key, value):
        if hasattr(self, key):
            setattr(self, key, value)
        else:
            agent = super(BenchmarkingAgent, self).__getattribute__('agent')
            setattr(agent, key, value)


