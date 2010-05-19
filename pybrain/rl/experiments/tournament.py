__author__ = 'Tom Schaul, tom@idsia.ch'

from pybrain.rl.environments.twoplayergames.twoplayergame import TwoPlayerGame
from pybrain.utilities import Named


class Tournament(Named):
    """ the tournament class is a specific kind of experiment, that takes a pool of agents
    and has them compete against each other in a TwoPlayerGame. """

    # do all moves need to be checked for legality?
    forcedLegality = False

    def __init__(self, env, agents):
        assert isinstance(env, TwoPlayerGame)
        self.startcolor = env.startcolor
        self.env = env
        self.agents = agents
        for a in agents:
            a.game = self.env
        self.reset()

    def reset(self):
        # a dictionnary attaching a list of outcomes to a player-couple-key
        self.results = {}
        self.rounds = 0
        self.numGames = 0

    def _produceAllPairs(self):
        """ produce a list of all pairs of agents (assuming ab <> ba)"""
        res = []
        for a in self.agents:
            for b in self.agents:
                if a != b:
                    res.append((a, b))
        return res

    def _oneGame(self, p1, p2):
        """ play one game between two agents p1 and p2."""
        self.numGames += 1
        self.env.reset()
        players = (p1, p2)
        p1.color = self.startcolor
        p2.color = -p1.color
        p1.newEpisode()
        p2.newEpisode()
        i = 0
        while not self.env.gameOver():
            p = players[i]
            i = (i + 1) % 2 # alternate
            act = p.getAction()

            if self.forcedLegality:
                tries = 0
                while not self.env.isLegal(*act):
                    tries += 1
                    # CHECKME: maybe the legality check is too specific?
                    act = p.getAction()
                    if tries > 50:
                        raise Exception('No legal move produced!')

            self.env.performAction(act)

        if players not in self.results:
            self.results[players] = []
        wincolor = self.env.getWinner()
        if wincolor == p1.color:
            winner = p1
        else:
            winner = p2
        self.results[players].append(winner)

    def organize(self, repeat=1):
        """ have all agents play all others in all orders, and repeat. """
        for dummy in range(repeat):
            self.rounds += 1
            for p1, p2 in self._produceAllPairs():
                self._oneGame(p1, p2)
        return self.results

    def eloScore(self, startingscore=1500, k=32):
        """ compute the elo score of all the agents, given the games played in the tournament.
        Also checking for potentially initial scores among the agents ('elo' variable). """
        # initialize
        elos = {}
        for a in self.agents:
            if 'elo' in a.__dict__:
                elos[a] = a.elo
            else:
                elos[a] = startingscore
        # adjust ratings
        for i, a1 in enumerate(self.agents[:-1]):
            for a2 in self.agents[i + 1:]:
                # compute score (in favor of a1)
                s = 0
                outcomes = self.results[(a1, a2)] + self.results[(a2, a1)]
                for r in outcomes:
                    if r == a1:
                        s += 1.
                    elif r == self.env.DRAW:
                        s += 0.5
                # what score would have been estimated?
                est = len(outcomes) / (1. + 10 ** ((elos[a2] - elos[a1]) / 400.))
                delta = k * (s - est)
                elos[a1] += delta
                elos[a2] -= delta
        for a, e in elos.items():
            a.elo = e
        return elos

    def __str__(self):
        s = 'Tournament results (' + str(self.rounds) + ' rounds, ' + str(self.numGames) + ' games):\n'
        for p1, p2 in self._produceAllPairs():
            wins = len(filter(lambda x: x == p1, self.results[(p1, p2)]))
            losses = len(filter(lambda x: x == p2, self.results[(p1, p2)]))
            s += ' ' * 3 + p1.name + ' won ' + str(wins) + ' times and lost ' + str(losses) + ' times against ' + p2.name + '\n'
        return s
