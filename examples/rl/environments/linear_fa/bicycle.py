from __future__ import print_function

"""An attempt to implement Randlov and Alstrom (1998). They successfully
use reinforcement learning to balance a bicycle, and to control it to drive
to a specified goal location. Their work has been used since then by a few
researchers as a benchmark problem.

We only implement the balance task. This implementation differs at least
slightly, since Randlov and Alstrom did not mention anything about how they
annealed/decayed their learning rate, etc. As a result of differences, the
results do not match those obtained by Randlov and Alstrom.

"""

__author__ = 'Chris Dembia, Bruce Cam, Johnny Israeli'

from scipy import asarray
from numpy import sin, cos, tan, sqrt, arcsin, arctan, sign, clip, argwhere
from matplotlib import pyplot as plt

import pybrain.rl.environments
from pybrain.rl.environments.environment import Environment
from pybrain.rl.learners.valuebased.linearfa import SARSALambda_LinFA
from pybrain.rl.agents.linearfa import LinearFA_Agent
from pybrain.rl.experiments import EpisodicExperiment
from pybrain.utilities import one_to_n

class BicycleEnvironment(Environment):
    """Randlov and Alstrom's bicycle model. This code matches nearly exactly
    some c code we found online for simulating Randlov and Alstrom's
    bicycle. The bicycle travels at a fixed speed.

    """

    # For superclass.
    indim = 2
    outdim = 10

    # Environment parameters.
    time_step = 0.01

    # Goal position and radius
    # Lagouakis (2002) uses angle to goal, not heading, as a state
    max_distance = 1000.

    # Acceleration on Earth's surface due to gravity (m/s^2):
    g = 9.82

    # See the paper for a description of these quantities:
    # Distances (in meters):
    c = 0.66
    dCM = 0.30
    h = 0.94
    L = 1.11
    r = 0.34
    # Masses (in kilograms):
    Mc = 15.0
    Md = 1.7
    Mp = 60.0
    # Velocity of a bicycle (in meters per second), equal to 10 km/h:
    v = 10.0 * 1000.0 / 3600.0

    # Derived constants.
    M = Mc + Mp # See Randlov's code.
    Idc = Md * r**2
    Idv = 1.5 * Md * r**2
    Idl = 0.5 * Md * r**2
    Itot = 13.0 / 3.0 * Mc * h**2 + Mp * (h + dCM)**2
    sigmad = v / r

    def __init__(self):
        Environment.__init__(self)
        self.reset()
        self.actions = [0.0, 0.0]
        self._save_wheel_contact_trajectories = False

    def performAction(self, actions):
        self.actions = actions
        self.step()

    def saveWheelContactTrajectories(self, opt):
        self._save_wheel_contact_trajectories = opt

    def step(self):
        # Unpack the state and actions.
        # -----------------------------
        # Want to ignore the previous value of omegadd; it could only cause a
        # bug if we assign to it.

        (theta, thetad, omega, omegad, _,
                xf, yf, xb, yb, psi) = self.sensors
        (T, d) = self.actions

        # For recordkeeping.
        # ------------------
        if self._save_wheel_contact_trajectories:
            self.xfhist.append(xf)
            self.yfhist.append(yf)
            self.xbhist.append(xb)
            self.ybhist.append(yb)

        # Intermediate time-dependent quantities.
        # ---------------------------------------
        # Avoid divide-by-zero, just as Randlov did.
        if theta == 0:
            rf = 1e8
            rb = 1e8
            rCM = 1e8
        else:
            rf = self.L / np.abs(sin(theta))
            rb = self.L / np.abs(tan(theta))
            rCM = sqrt((self.L - self.c)**2 + self.L**2 / tan(theta)**2)

        phi = omega + np.arctan(d / self.h)

        # Equations of motion.
        # --------------------
        # Second derivative of angular acceleration:
        omegadd = 1 / self.Itot * (self.M * self.h * self.g * sin(phi)
                - cos(phi) * (self.Idc * self.sigmad * thetad
                    + sign(theta) * self.v**2 * (
                        self.Md * self.r * (1.0 / rf + 1.0 / rb)
                        + self.M * self.h / rCM)))
        thetadd = (T - self.Idv * self.sigmad * omegad) / self.Idl

        # Integrate equations of motion using Euler's method.
        # ---------------------------------------------------
        # yt+1 = yt + yd * dt.
        # Must update omega based on PREVIOUS value of omegad.
        omegad += omegadd * self.time_step
        omega += omegad * self.time_step
        thetad += thetadd * self.time_step
        theta += thetad * self.time_step

        # Handlebars can't be turned more than 80 degrees.
        theta = np.clip(theta, -1.3963, 1.3963)

        # Wheel ('tyre') contact positions.
        # ---------------------------------

        # Front wheel contact position.
        front_temp = self.v * self.time_step / (2 * rf)
        # See Randlov's code.
        if front_temp > 1:
            front_temp = sign(psi + theta) * 0.5 * np.pi
        else:
            front_temp = sign(psi + theta) * arcsin(front_temp)
        xf += self.v * self.time_step * -sin(psi + theta + front_temp)
        yf += self.v * self.time_step * cos(psi + theta + front_temp)

        # Rear wheel.
        back_temp = self.v * self.time_step / (2 * rb)
        # See Randlov's code.
        if back_temp > 1:
            back_temp = np.sign(psi) * 0.5 * np.pi
        else:
            back_temp = np.sign(psi) * np.arcsin(back_temp)
        xb += self.v * self.time_step * -sin(psi + back_temp)
        yb += self.v * self.time_step * cos(psi + back_temp)

        # Preventing numerical drift.
        # ---------------------------
        # Copying what Randlov did.
        current_wheelbase = sqrt((xf - xb)**2 + (yf - yb)**2)
        if np.abs(current_wheelbase - self.L) > 0.01:
            relative_error = self.L / current_wheelbase - 1.0
            xb += (xb - xf) * relative_error
            yb += (yb - yf) * relative_error

        # Update heading, psi.
        # --------------------
        delta_y = yf - yb
        if (xf == xb) and delta_y < 0.0:
            psi = np.pi
        else:
            if delta_y > 0.0:
                psi = arctan((xb - xf) / delta_y)
            else:
                psi = sign(xb - xf) * 0.5 * np.pi - arctan(delta_y / (xb - xf))

        self.sensors = np.array([theta, thetad, omega, omegad, omegadd,
                xf, yf, xb, yb, psi])

    def reset(self):
        theta = 0
        thetad = 0
        omega = 0
        omegad = 0
        omegadd = 0
        xf = 0
        yf = self.L
        xb = 0
        yb = 0

        psi = np.arctan((xb - xf) / (yf - yb))
        self.sensors = np.array([theta, thetad, omega, omegad, omegadd,
                xf, yf, xb, yb, psi])

        self.xfhist = []
        self.yfhist = []
        self.xbhist = []
        self.ybhist = []

    def getSteer(self):
        return self.sensors[0]
    def getTilt(self):
        return self.sensors[2]
    def get_xfhist(self):
        return self.xfhist
    def get_yfhist(self):
        return self.yfhist
    def get_xbhist(self):
        return self.xbhist
    def get_ybhist(self):
        return self.ybhist
    def getSensors(self):
        return self.sensors

class BalanceTask(pybrain.rl.environments.EpisodicTask):
    """The rider is to simply balance the bicycle while moving with the
    speed perscribed in the environment. This class uses a continuous 5
    dimensional state space, and a discrete state space.

    This class is heavily guided by
    pybrain.rl.environments.cartpole.balancetask.BalanceTask.

    """
    max_tilt = np.pi / 6.
    nactions = 9

    def __init__(self, max_time=1000.0):
        super(BalanceTask, self).__init__(BicycleEnvironment())
        self.max_time = max_time
        # Keep track of time in case we want to end episodes based on number of
        # time steps.
        self.t = 0

    @property
    def indim(self):
        return 1

    @property
    def outdim(self):
        return 5

    def reset(self):
        super(BalanceTask, self).reset()
        self.t = 0

    def performAction(self, action):
        """Incoming action is an int between 0 and 8. The action we provide to
        the environment consists of a torque T in {-2 N, 0, 2 N}, and a
        displacement d in {-.02 m, 0, 0.02 m}.

        """
        self.t += 1
        assert round(action[0]) == action[0]

        # -1 for action in {0, 1, 2}, 0 for action in {3, 4, 5}, 1 for
        # action in {6, 7, 8}
        torque_selector = np.floor(action[0] / 3.0) - 1.0
        T = 2 * torque_selector
        # Random number in [-1, 1]:
        p = 2.0 * np.random.rand() - 1.0
        # -1 for action in {0, 3, 6}, 0 for action in {1, 4, 7}, 1 for
        # action in {2, 5, 8}
        disp_selector = action[0] % 3 - 1.0
        d = 0.02 * disp_selector + 0.02 * p
        super(BalanceTask, self).performAction([T, d])

    def getObservation(self):
        (theta, thetad, omega, omegad, omegadd,
                xf, yf, xb, yb, psi) = self.env.getSensors()
        return self.env.getSensors()[0:5]

    def isFinished(self):
        # Criterion for ending an episode. From Randlov's paper:
        # "When the agent can balance for 1000 seconds, the task is considered
        # learned."
        if np.abs(self.env.getTilt()) > self.max_tilt:
            return True
        elapsed_time = self.env.time_step * self.t
        if elapsed_time > self.max_time:
            return True
        return False

    def getReward(self):
        # -1 reward for falling over; no reward otherwise.
        if np.abs(self.env.getTilt()) > self.max_tilt:
            return -1.0
        return 0.0

class LinearFATileCoding3456BalanceTask(BalanceTask):
    """An attempt to exactly implement Randlov's function approximation. He
    discretized (tiled) the state space into 3456 bins. We use the same action
    space as in the superclass.

    """
    # From Randlov, 1998:
    theta_bounds = np.array(
            [-0.5 * np.pi, -1.0, -0.2, 0, 0.2, 1.0, 0.5 * np.pi])
    thetad_bounds = np.array(
            [-np.inf, -2.0, 0, 2.0, np.inf])
    omega_bounds = np.array(
            [-BalanceTask.max_tilt, -0.15, -0.06, 0, 0.06, 0.15,
                BalanceTask.max_tilt])
    omegad_bounds = np.array(
            [-np.inf, -0.5, -0.25, 0, 0.25, 0.5, np.inf])
    omegadd_bounds = np.array(
            [-np.inf, -2.0, 0, 2.0, np.inf])
    # http://stackoverflow.com/questions/3257619/numpy-interconversion-between-multidimensional-and-linear-indexing
    nbins_across_dims = [
            len(theta_bounds) - 1,
            len(thetad_bounds) - 1,
            len(omega_bounds) - 1,
            len(omegad_bounds) - 1,
            len(omegadd_bounds) - 1]
    # This array, when dotted with the 5-dim state vector, gives a 'linear'
    # index between 0 and 3455.
    magic_array = np.cumprod([1] + nbins_across_dims)[:-1]

    @property
    def outdim(self):
        # Used when constructing LinearFALearner's.
        return 3456

    def getBin(self, theta, thetad, omega, omegad, omegadd):
        bin_indices = [
                np.digitize([theta], self.theta_bounds)[0] - 1,
                np.digitize([thetad], self.thetad_bounds)[0] - 1,
                np.digitize([omega], self.omega_bounds)[0] - 1,
                np.digitize([omegad], self.omegad_bounds)[0] - 1,
                np.digitize([omegadd], self.omegadd_bounds)[0] - 1,
                ]
        return np.dot(self.magic_array, bin_indices)

    def getBinIndices(self, linear_index):
        """Given a linear index (integer between 0 and outdim), returns the bin
        indices for each of the state dimensions.

        """
        return linear_index / self.magic_array % self.nbins_across_dims

    def getObservation(self):
        (theta, thetad, omega, omegad, omegadd,
                xf, yf, xb, yb, psi) = self.env.getSensors()
        state = one_to_n(self.getBin(theta, thetad, omega, omegad, omegadd),
                self.outdim)
        return state

class SARSALambda_LinFA_ReplacingTraces(SARSALambda_LinFA):
    """Randlov used replacing traces, but this doesn't exist in PyBrain's
    SARSALambda.

    """
    def _updateEtraces(self, state, action, responsibility=1.):
        self._etraces *= self.rewardDiscount * self._lambda * responsibility
        # This assumes that state is an identity vector (like, from one_to_n).
        self._etraces[action] = clip(self._etraces[action] + state, -np.inf, 1.)
        # Set the trace for all other actions in this state to 0:
        action_bit = one_to_n(action, self.num_actions)

        for argstate in argwhere(state == 1) :
        	self._etraces[argwhere(action_bit != 1), argstate] = 0.


task = LinearFATileCoding3456BalanceTask()
env = task.env

# The learning is very sensitive to the learning rate decay.
learner = SARSALambda_LinFA_ReplacingTraces(task.nactions, task.outdim,
        learningRateDecay=2000)
learner._lambda = 0.95

task.discount = learner.rewardDiscount

agent = LinearFA_Agent(learner)
agent.logging = False

exp = EpisodicExperiment(task, agent)

performance_agent = LinearFA_Agent(learner)
performance_agent.logging = False
performance_agent.greedy = True
performance_agent.learning = False

env.saveWheelContactTrajectories(True)
plt.ion()
plt.figure(figsize=(8, 4))

ax1 = plt.subplot(1, 2, 1)
ax2 = plt.subplot(1, 2, 2)

def update_wheel_trajectories():
    front_lines = ax2.plot(env.get_xfhist(), env.get_yfhist(), 'r')
    back_lines = ax2.plot(env.get_xbhist(), env.get_ybhist(), 'b')
    plt.axis('equal')

perform_cumrewards = []
for irehearsal in range(7000):

    # Learn.
    # ------
    r = exp.doEpisodes(1)
    # Discounted reward.
    cumreward = exp.task.getTotalReward()
    #print 'cumreward: %.4f; nsteps: %i; learningRate: %.4f' % (
    #        cumreward, len(r[0]), exp.agent.learner.learningRate)

    if irehearsal % 50 == 0:
        # Perform (no learning).
        # ----------------------
        # Swap out the agent.
        exp.agent = performance_agent
    
        # Perform.
        r = exp.doEpisodes(1)
        perform_cumreward = task.getTotalReward()
        perform_cumrewards.append(perform_cumreward)
        print('PERFORMANCE: cumreward:', perform_cumreward, 'nsteps:', len(r[0]))
    
        # Swap back the learning agent.
        performance_agent.reset()
        exp.agent = agent
    
        ax1.cla()
        ax1.plot(perform_cumrewards, '.--')
        # Wheel trajectories.
        update_wheel_trajectories()
    
        plt.pause(0.001)
