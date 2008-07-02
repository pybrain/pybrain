__author__ = 'Tom Schaul, tom@idsia.ch'

import socket
import thread
from random import choice

from pybrain.rl.agents.capturegameplayers.clientwrapper import ClientCapturePlayer
from pybrain.rl.environments.twoplayergames import CaptureGame

if __name__ == '__main__':
    s1, s2 = socket.socketpair()
    
    def randomServer():
        while True:
            #print 'Waiting for input',
            js = ''
            while len(js) < 2:    
                js = s2.recv(1000)
                #print ',',
            print 'recieved:', js,
            r = '('+str(choice(range(4)))+','+str(choice(range(4)))+')'
            print 'sending', r,
            s2.send(r+'\n')
            print 'sent.'
            
    thread.start_new_thread(randomServer, ())
    
    g = CaptureGame(4)
    a = ClientCapturePlayer(g, verbose = False)
    a.theSocket = s1
    g.doMove(*a.getAction())
    g.doMove(*a.getAction())
    g.doMove(*a.getAction())
    print
    print g