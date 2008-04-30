__author__ = 'Tom Schaul, tom@idsia.ch'

from randomplayer import RandomCapturePlayer

class ModuleDecidingPlayer(RandomCapturePlayer):
    """ A Capture-game player that plays according to the rules, but choosing its moves
    according to the output of a module that takes as input the current state of the board. """

    def __init__(self, game, color, module):
        RandomCapturePlayer.__init__(self, game, color)
        self.module = module
        
    def getAction(self):
        """ get suggested action, return them if they are legal, otherwise choose randomly. """        
        # TODO:
        #return [self.color, choice(self.game.getLegals(self.color))]



# LearningAgent-version...
#
#from pybrain.rl.agents import PolicyGradientAgent
#
#class LearningCapturePlayer(RandomCapturePlayer, PolicyGradientAgent):
#    
#    tmp = None
#    filtered = True
#    
#    def __init__(self, game, color, learningrate = 0.01, momentum = 0, verbose = False, 
#                 illegalpunish = 0.9, nbillegals = None, net = None, merciless = False, **args):        
#        RandomCapturePlayer.__init__(self, game, color)        
#        self.merciless = merciless
#        self.illegalpunish = illegalpunish
#        self.lr = learningrate
#        self.verbose = verbose
#        if nbillegals == None:
#            self.nbillegals = (self.game.size/2)**2
#        else:
#            self.nbillegals = nbillegals
#        self.illegals = 0
#        self.legals = 0
#        if net == None:
#            net = CaptureSwipingNet(self.game.size, **args)  
#        self.net = net
#        self.trainer = PolicyGradientTrainer(self.net)
#        self.trainer.setLearningRate(learningrate)
#        if momentum > 0:
#            self.trainer.kpointer.setMomentum(True, momentum)
#        self.integrateObservation()
#        
#    def getAction(self):
#        """ get suggested actions, return them if they are legal, otherwise return a pass """        
#        legal = self.game.getLegals(self.color)
#        for i in range(self.nbillegals): #CHECKME
#            res = self.suggestAction()
#            if res in legal:
#                if self.verbose: print res,
#                self.legals += 1
#                # CHECKME: 
#                #self.giveReward(self.illegalpunish)
#                return [self.color, res]            
#            elif self.merciless:
#                # lose the game
#                self.illegals += 1
#                if self.verbose:
#                    print 'resigning after an illegal move'
#                return [self.color, 'resign']
#            else:
#                # punishment for an illegal move
#                self.illegals += 1
#                self.giveReward(-self.illegalpunish)
#                if self.verbose:
#                    print self.game,
#                    print 'ILLEGAL:', res
#                    print 'thermal softmax input (x1000)', map(lambda x: int(x*1000), self.net['smax'].getInput())
#                    print 'softmax input (x1000)', map(lambda x: int(x*1000), self.net['smax']['softmax'].getInput())
#                    print 'temp', self.net['smax'].getTemperature()
#        # if too many illegal moves have been done - just choose a random legal one!
#        if self.verbose:
#            print 'Too many illegal moves:', i
#            print self.game
#        self.legals += 1
#        return [self.color, choice(legal)]
#        
#    def integrateObservation(self, obs = None):
#        board = self.game.getBoardArray()
#        if self.verbose: print board
#        self.trainer.integrateObservation(board)
#    
#    def suggestAction(self):
#        """ let the network suggest a move. """
#        outp = self.trainer.getAction()        
#        if self.filtered:            
#            m = SoftmaxLayer(self.game.size**2)
#            k = ones(self.game.size**2)*-1000
#            for p in self.game.getLegals(self.color):
#                k[p[0]*self.game.size+p[1]] = 0
#            k += self.net['smax'].getInput() 
#            k *= self.net['smax'].t            
#            m.addToInput(k)
#            outp = m.forwardPass()                        
#        index = maxIndex(outp)        
#        return (index/self.game.size, index%self.game.size)        
#    
#    def giveReward(self, r):
#        """ punishment for an illegal move, a rewarding/punishing final winning/losing score at
#        the end of a game, and 0 otherwise. """
#        #print r,
#        self.trainer.giveReward(r)
#                
#    def newEpisode(self):
#        self.trainer.newEpisode()        
#        
#    def enableTraining(self):
#        if self.tmp != None:
#            self.illegalpunish = self.tmp
#        self.trainer.setLearningRate(self.lr)
#        self.net.defrost()        
#        
#    def disableTraining(self):
#        self.tmp = self.illegalpunish
#        self.illegalpunish = 0
#        self.lr = self.trainer.getLearningRate()
#        self.trainer.setLearningRate(0.)
#        self.net.freeze()
#        
#        
 