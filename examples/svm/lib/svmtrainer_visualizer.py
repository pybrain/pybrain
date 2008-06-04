
__author__ = 'Michael Isik'


from visual import PlotPanel2D, PlotPanel3D, IdleGui
from matplotlib.numerix import arange, sin, cos, pi
import wx
import pylab
from numpy import array, empty, zeros, where
import numpy
import scipy.linalg
import threading
from lib.debug import dbg,tracedm


class SVMInputSpace2DViewer(PlotPanel2D):
    @tracedm
    def __init__(self, trainer, parent):
        PlotPanel2D.__init__(self, parent)

        self._trainer = trainer
        self._points  = []
        self._texts   = []
        self._old_i1  = 0
        self._old_i2  = 0
        self._resolution = 90

    @tracedm
    def update(self):
        if self._trainer.stepno < 0: return
        self._trainer.updateBeta()
        if len(self._points) == 0 :
            self._plotPoints()
        self._updatePoints()
        title = "Step %d" % (self._trainer.stepno)
        self.subplot.set_title( title )
        self._plotMesh()


    @tracedm
    def _plotPoints(self):
        self._points   = []
        self._texts    = []
        self._wstexts  = []
        for i in range(self._trainer.module._kernel.l):
            x  = self._trainer.module._kernel._X[i]
            t  = self._trainer.module._kernel._Y[i]
            point = self.subplot.plot([x[0]],[x[1]],"g"+self._targetToShape(t))
            self._points.append( point )
            y     = self._trainer.module.rawOutput(x)
            text  = self.subplot.text( x[0], x[1] ,"%.2f"%y )
            self._texts.append( text )


    @tracedm
    def _updatePoints(self):
        for i in range(self._trainer.module._kernel.l):
            x = self._trainer.module._kernel._X[i]
            y = self._trainer.module.rawOutput(x)
            pylab.setp( self._texts[i], text="%0.2f"%y )
        i1  = self._trainer._i
        i2  = self._trainer._j
        oi1 = self._old_i1
        oi2 = self._old_i2
        self._old_i1  = i1
        self._old_i2  = i2
        pylab.setp(self._points[oi1], color = 'g')
        pylab.setp(self._points[oi2], color = 'g')
        pylab.setp(self._points[i1],  color = 'r')
        pylab.setp(self._points[i2],  color = 'r')

        for i in range(self._trainer.module._kernel.l):
            x = self._trainer.module._kernel._X[i]
            t = self._trainer.module._kernel._Y[i]
            a = self._trainer.module._alpha[i]
            y = self._trainer.module.rawOutput(x)



    @tracedm
    def _targetToShape(self, t):
        if t==1: return "o"
        else:    return "^"

    @tracedm
    def _plotMesh(self):
        try:
            self._maxx1
        except AttributeError:
            x = self._trainer.module._kernel._X
            self._maxx1,self._maxx2 = x.max( axis=0 )
            self._minx1,self._minx2 = x.min( axis=0 )
            self._pad = (self._maxx1+self._maxx2-self._minx1-self._minx2) / 7
            pad = self._pad
            self._minx1 -= pad; self._maxx1 += pad
            self._minx2 -= pad; self._maxx2 += pad
            self.subplot.set_autoscale_on( False )
            self.subplot.set_xlim(self._minx1,self._maxx1)
            self.subplot.set_ylim(self._minx2,self._maxx2)

        n = self._resolution
        x1    = numpy.linspace( self._minx1, self._maxx1, n )
        x2    = numpy.linspace( self._minx2, self._maxx2, n )
        X1,X2 = numpy.meshgrid(x1,x2);


        Y = empty(X1.shape,float)
        for i in range(len(X1)):
            row1 = X1[i]
            row2 = X2[i]
            for j in range(len(row1)):
                Y[i][j] = self._trainer.module.rawOutput([row1[j],row2[j]])
        Y =  1 / ( 1. + numpy.exp( - Y * 20. ) ) - 0.5

        self.subplot.pcolormesh( X1, X2, Y, shading='flat' )




class SVMFeatureSpace3DViewer(PlotPanel3D):
    @tracedm
    def __init__(self, trainer, parent):
        PlotPanel3D.__init__(self, parent)

        self._trainer = trainer
        self._points = []
        self._texts  = []
        self._plane  = None
        self._title  = None
        self._old_i1  = 0
        self._old_i2  = 0

    @tracedm
    def update(self):
        if self._trainer.stepno < 0: return
        self._trainer.updateBeta()

        # if no points were drawn yet
        if len(self._points) == 0:
            # draw the initial points
            self._plotPoints()
            self.subplot.set_autoscale_on( False )

        self._updatePoints()
        self._plotPlane()

#        title = "Step %d" % (self._trainer.stepno)
#        if self._title != None:
#            pylab.setp( self._title, text=title )
#        else:
#            self._title = self.subplot.text3D( 0, 0, 0, title)


    @tracedm
    def _plotPoints(self):
        X   = self._trainer.module._kernel._X
        Y   = self._trainer.module._kernel._Y
        phi = self._trainer.module._kernel.phi

        self._points = []
        self._texts  = []
        xf = []
        for i in range(self._trainer.module._kernel.l):
            xp = phi(X[i])
            xf.append(xp)
            point = self.subplot.plot3D([xp[0]],[xp[1]],[xp[2]],"g"+self._targetToShape(Y[i]))
            self._points.append( point )
        xf = array(xf)
        self._maxx1, self._maxx2, self._maxx3 = xf.max(axis=0)
        self._minx1, self._minx2, self._minx3 = xf.min(axis=0)
        pad = (self._maxx1 + self._maxx2 + self._maxx3 - self._minx1 - self._minx2 - self._minx3) / 7
        self._maxx1+=pad;self._maxx2+=pad;self._maxx3+=pad
        self._minx1-=pad;self._minx2-=pad;self._minx3-=pad



    @tracedm
    def _updatePoints(self):
        i1  = self._trainer._i
        i2  = self._trainer._j
        oi1 = self._old_i1
        oi2 = self._old_i2
        if i1 == None or i2 == None: return
        self._old_i1  = i1
        self._old_i2  = i2

        pylab.setp( self._points[oi1] , color = 'g' )
        pylab.setp( self._points[oi2] , color = 'g' )
        pylab.setp( self._points[i1]  , color = 'r' )
        pylab.setp( self._points[i2]  , color = 'r' )


    @tracedm
    def _plotPlane(self):
        if self._plane != None:
            self.subplot.collections.remove(self._plane)

        [ w1, w2, w3 ]  = self._trainer.calculateW()

        if abs(w1)<0.1 and abs(w2)<0.1 and abs(w3)<0.1: return


        n = 30.
        x1    = numpy.linspace(self._minx1,self._maxx1,n)
        x2    = numpy.linspace(self._minx2,self._maxx2,n)
        X1,X2 = numpy.meshgrid(x1,x2)

        b  = self._trainer.module._beta
        X3 = array( ( b - X1*w1 - X2*w2 ) / w3 )

        # abschneiden?
        #max_idx = where( ( X3 > self._maxx3 ) )
        #min_idx = where( ( X3 < self._minx3 ) )
        #X3[max_idx] = self._maxx3
        #X3[min_idx] = self._minx3

        self._plane = self.subplot.plot_wireframe( X1, X2, X3 )



    def _targetToShape(self, t):
        if t==1: return "o"
        else:    return "^"








# custom event ids
ID_RUN      = wx.NewId()
ID_STEP     = wx.NewId()
ID_STOP     = wx.NewId()
ID_PAUSE    = wx.NewId()
ID_VISUAL   = wx.NewId()
ID_SNAPSHOT = wx.NewId()
ID_INCRES   = wx.NewId()
ID_DECRES   = wx.NewId()
class SVMTrainerVisualizer(IdleGui):
    @tracedm
    def __init__(self, trainer):
        self._title = "SVM Visualization"
        IdleGui.__init__(self)

        # setup the step event hook
        self._trainer = trainer
        self._trainer._onStep = self._onTrainerStep

        # threading stuff
        self._step_semaphore = threading.Semaphore()
        self._step_semaphore.acquire()
        self._lock = threading.Lock()


        # initialize gui state
        self._do_visualize = True
        self._run          = False


        # initialize gui elements
        self._status_text=""

        self._toolbar = self.CreateToolBar( wx.TB_HORIZONTAL | wx.NO_BORDER | wx.TB_FLAT )
        self._toolbar.AddSimpleTool( ID_PAUSE, wx.Bitmap("images/pause.png" ,wx.BITMAP_TYPE_PNG ), "Pause", "Pause" )
        self._toolbar.AddSimpleTool( ID_RUN,   wx.Bitmap("images/run.png"   ,wx.BITMAP_TYPE_PNG ), "Run", "Run" )
        self._toolbar.AddSimpleTool( ID_STEP,  wx.Bitmap("images/step.png"  ,wx.BITMAP_TYPE_PNG ), "Step", "Step" )
        self._toolbar.AddSeparator()
        self._toolbar.AddCheckTool( ID_VISUAL, wx.Bitmap("images/eye.gif"  , wx.BITMAP_TYPE_GIF ), wx.NullBitmap, "Toggle Visualization", "Toggle Visualization" )
        self._toolbar.ToggleTool( ID_VISUAL, True )
        self._toolbar.AddSeparator()
        self._toolbar.AddSimpleTool( ID_SNAPSHOT,  wx.Bitmap("images/camera.gif" ,wx.BITMAP_TYPE_GIF ), "Dump Image", "Dump Image" )
        self._toolbar.AddSeparator()
        self._toolbar.AddSimpleTool( ID_DECRES,  wx.Bitmap("images/lowres.png"  ,wx.BITMAP_TYPE_PNG ), "Decrease Resolution", "Decrease Resolution" )
        self._toolbar.AddSimpleTool( ID_INCRES,  wx.Bitmap("images/highres.png" ,wx.BITMAP_TYPE_PNG ), "Increase Resolution", "Increase Resolution" )
        self._toolbar.Realize()
        self.SetBackgroundColour(wx.Colour(255,255,200))

        # integrate observers into gui
        self._observers = []
        kernel = trainer.module._kernel
        if kernel.getXDim() == 2:
            self._observers.append(SVMInputSpace2DViewer(self._trainer, self))
#            self._observers[-1].SetBackgroundColour(wx.Colour(255,100,200))

        if kernel.isExplicit() and kernel.getFeatureSpaceDim() == 3:
            self._observers.append(SVMFeatureSpace3DViewer(self._trainer, self))
#            self._observers[-1].SetBackgroundColour(wx.Colour(55,100,200))
        if not len( self._observers ):
            raise Exception("No visualization available. Only possible with 2-dimensional input space and/or 3-dimensional feature space!")

        pylab.cool()

        self._sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.SetSizer(self._sizer)

        for observer in self._observers:
            observer.canvas.Bind(wx.EVT_LEFT_DCLICK, self._onStepCommand)
            observer.canvas.Bind(wx.EVT_MOUSEWHEEL,  self._onStepCommand)
            observer.canvas.Bind(wx.EVT_KEY_DOWN,    self._onKeyDown)
            self._sizer.Add(observer, wx.ID_ANY, wx.EXPAND)
            self._sizer.SetItemMinSize(observer,600, 600)

        # setup gui event handlers
        self.Bind( wx.EVT_MENU, self._onRun,         id = ID_RUN      )
        self.Bind( wx.EVT_MENU, self._onPause,       id = ID_PAUSE    )
        self.Bind( wx.EVT_MENU, self._onStepCommand, id = ID_STEP     )
        self.Bind( wx.EVT_MENU, self._onVisual,      id = ID_VISUAL   )
        self.Bind( wx.EVT_MENU, self._onSnapshot,    id = ID_SNAPSHOT )
        self.Bind( wx.EVT_MENU, self._onIncRes,      id = ID_INCRES   )
        self.Bind( wx.EVT_MENU, self._onDecRes,      id = ID_DECRES   )
        self.Bind( wx.EVT_LEFT_DCLICK, self._onStepCommand            )



        # refresh layout
        self._sizer.Fit(self)
        self.Layout()

        # start the gui
        self.start()
        self.doUpdateObservers()
        self._observers[0].SetFocus()





    @tracedm
    def _onTrainerStep(self):


        self.doUpdateObservers()

        self._status_text = "Processing Step %d" % self._trainer.stepno
        if self._do_visualize:
            self._waitForIdle()

        if not self._run:
            self._step_semaphore.acquire()


    @tracedm
    def doUpdateObservers(self):
        self._invoke(self.update)

    @tracedm
    def update(self):
        if self._do_visualize:
            self.lockTrainer()
            for observer in self._observers:
                observer.update()
                observer._setSize()
            self.unlockTrainer()



    @tracedm
    def _onVisual(self,evt):
        self._do_visualize = self._toolbar.GetToolState(ID_VISUAL)
        dbg("set do_visualize to: ", self._do_visualize)
        if self._do_visualize:
            self.update()




    @tracedm
    def lockTrainer(self):
        self._lock.acquire()

    @tracedm
    def unlockTrainer(self):
        self._lock.release()


    @tracedm
    def _onKeyDown(self,evt):
        code=evt.GetKeyCode()
        if   code == wx.WXK_RETURN:
            self._onStepCommand(evt)
        elif code == 86: # V
            self._toolbar.ToggleTool( ID_VISUAL,
                not self._toolbar.GetToolState(ID_VISUAL) )
            self._onVisual(None)
        elif code == 82: # R
            self._onRun(None)
        elif code == 80: # P
            self._onPause(None)

    @tracedm
    def _onStepCommand(self,evt):
#        self.SetStatusText("Step")
        self._step_semaphore.release()

    @tracedm
    def _onRun(self,evt):
        self._run = True
#        self.SetStatusText("Run")
        self._step_semaphore.release()

    @tracedm
    def _onPause(self,evt):
        self._run = False
#        self.SetStatusText("Pause")

    @tracedm
    def _onSnapshot(self,evt):
        for i,o in enumerate(self._observers):
            filename = "fig_%d_%d.png" % (i,self._trainer.stepno)
            print "saving fig to: ", filename
            o.saveFig(filename)

    @tracedm
    def _onIncRes(self,evt):
        for o in self._observers:
            try:
                o._resolution *= 1.1
                self.SetStatusText("New Resolution %d" % o._resolution)
            except AttributeError: pass

    @tracedm
    def _onDecRes(self,evt):
        for o in self._observers:
            try:
                o._resolution *= 0.9
                o._resolution = max(o._resolution,4)
                self.SetStatusText("New Resolution %d" % o._resolution)
            except AttributeError: pass






