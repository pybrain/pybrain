__author__ = 'Tom Schaul, tom@idsia.ch'

import socket

from captureplayer import CapturePlayer
from pybrain.rl.environments.twoplayergames import CaptureGame

# TODO: allow partially forced random moves.

class ClientCapturePlayer(CapturePlayer):
    """ A wrapper class for using external code to play the capture game,
    interacting via a TCP socket. """

    verbose = False

    def __init__(self, game, color=CaptureGame.BLACK, player='AtariGreedy', **args):
        '''player: AtariGreedy, AtariMinMax, AtariAlphaBeta possible'''
        CapturePlayer.__init__(self, game, color, **args)
        # build connection
        host = "127.0.0.1"
        port = 6524
        self.player = player
        try:
            self.theSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.theSocket.connect((host, port))
            if self.verbose:
                print("Connected to server")
        except:
            print('Failed to connect')

        #define player
        self.theSocket.send(player + '-' + str(color) + '\n')
        if self.verbose:
            print('Sending:', player + '-' + str(color))
        accept = ""
        while len (accept) < 2:
            accept = self.theSocket.recv(1000)
        assert accept == 'OK'


    def getAction(self):
        # build a java string
        if self.color == CaptureGame.BLACK:
            js = '1-'
        else:
            js = '2-'
        for i, p in enumerate(self.game._iterPos()):
            if i % self.game.size == 0:
                js += '-'
            if self.game.b[p] == CaptureGame.BLACK:
                js += '1'
            elif self.game.b[p] == CaptureGame.WHITE:
                js += '2'
            else:
                js += '0'

        # get the suggested move from the java player:
        if self.verbose:
            print('Sending:', js)
        self.theSocket.send(js + '\n')
        jr = ""
        if self.verbose:
            print('Waiting for server',)
        while len (jr) < 2:
            jr = self.theSocket.recv(1000)
            if self.verbose:
                print('.',)
        if self.verbose:
            print(" received.", jr)

        chosen = eval(jr)
        assert self.game.isLegal(self.color, chosen)
        return [self.color, chosen]


