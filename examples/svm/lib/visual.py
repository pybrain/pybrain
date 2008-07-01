
__author__ = 'Michael Isik'

import matplotlib
matplotlib.interactive(False)
matplotlib.use('WXAgg')



from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg
from matplotlib.figure  import Figure
import wx

from pybrain.examples.svm.lib.debug import dbg,tracedm



class NoRepaintCanvas(FigureCanvasWxAgg):
    @tracedm
    def __init__(self, *args, **kwargs):
        FigureCanvasWxAgg.__init__(self, *args, **kwargs)
        self._drawn = 0

    @tracedm
    def _onPaint(self, evt):
        self.repaint()

    @tracedm
    def repaint(self):
        if not self._isRealized:
            self.realize()
        if self._drawn < 2:
            self.draw(repaint = False)
            self._drawn += 1
        self.gui_repaint(drawDC=wx.PaintDC(self))


class AbstractPlotPanel(wx.Panel):
    @tracedm
    def __init__(self,parent):
        wx.Panel.__init__(self,parent,wx.ID_ANY, None, None, wx.NO_FULL_REPAINT_ON_RESIZE)
        self.Bind(wx.EVT_SIZE, self._onSize)
        self.Bind(wx.EVT_IDLE, self._onIdle)
        dpi=None
        self.figure  = Figure(None, dpi)
        self.canvas = NoRepaintCanvas(self, -1, self.figure)

    @tracedm
    def _setSize(self, pixels = None):
        if not pixels:
            pixels = self.GetClientSize()
        dbg( "setting size to: ",pixels )
        self.canvas.SetSize(pixels)
        self.figure.set_size_inches(
            pixels[0]/self.figure.get_dpi(),
            pixels[1]/self.figure.get_dpi()
        )

    @tracedm
    def _onSize(self, event):
        self._resizeflag = True

    # zzzzzzzzz on idle schon hier verwendet
    @tracedm
    def _onIdle(self, evt):
        if self._resizeflag:
            self._resizeflag = False
            self._setSize()

    @tracedm
    def saveFig(self, filename):
        self.figure.savefig( filename )



class PlotPanel2D(AbstractPlotPanel):
    @tracedm
    def __init__(self, parent):
        AbstractPlotPanel.__init__(self, parent)

        self.subplot = self.figure.add_subplot(111)
        self._setSize()

class PlotPanel3D(AbstractPlotPanel):
    @tracedm
    def __init__(self, parent):
        AbstractPlotPanel.__init__(self, parent)

        try: axes3d
        except NameError:
            import matplotlib.axes3d as axes3d
            global axes3d
        self.subplot = axes3d.Axes3D(self.figure)
        self._setSize()






import threading

EVT_INVOKE_TYPE = wx.NewEventType()
EVT_INVOKE      = wx.PyEventBinder(EVT_INVOKE_TYPE)
class InvokeEvent(wx.PyEvent):
    @tracedm
    def __init__(self, func, *args, **kwargs):
        wx.PyEvent.__init__(self)
        self.SetEventType(EVT_INVOKE_TYPE)
        self.__func = func
        self.__args = args
        self.__kwargs = kwargs

    @tracedm
    def invoke(self):
        self.__func(*self.__args, **self.__kwargs)



class SimpleGui(wx.Frame, threading.Thread):
    _sizer = None
    @tracedm
    def __init__(self):
        self._status_text = ''
        threading.Thread.__init__(self)
        self._app = wx.PySimpleApp()
        try: self._title
        except AttributeError: self._title="Your Ad Here"
        wx.Frame.__init__( self, None, wx.ID_ANY, self._title) #, style=wx.DEFAULT_FRAME_STYLE | wx.NO_FULL_REPAINT_ON_RESIZE )

        # setup invoking system
        self._invoke_cond = threading.Condition()
        self.Bind(EVT_INVOKE, self._onInvoke)

        # set some event handlers
        self.Bind(wx.EVT_CLOSE, self._onClose)


        self.CreateStatusBar()

        if self._sizer is None:
            self._sizer = wx.BoxSizer(wx.HORIZONTAL)
            self.SetSizer(self._sizer)

        # refresh layout
        self._sizer.Fit(self)
        self.Layout()


    def _refreshLayout(self):
        self._sizer.Fit(self)
        self.Layout()


    @tracedm
    def run(self):
        self.Show()
        self._app.MainLoop()


    # zzz wrong place?
    @tracedm
    def doUpdateObservers(self):
        self._invoke(self.update)


    @tracedm
    def _onClose(self,evt):
        exit(0)

    # zzz wrong place?
    @tracedm
    def _onTimer(self,evt):
        # we want idle events all the time
        self.SetStatusText(self._status_text)
        wx.PostEvent( self, DummyEvent() )



    @tracedm
    def _invoke(self, func, *args, **kwargs):
        """ Initiates the execution of func with parameters args and kwargs
            within the gui thread by sending an OnInvoke event. The execution
            is synchronous, hence this function will not return until func
            returned.
        """
        self._invoke_cond.acquire()
        wx.PostEvent(self,InvokeEvent(func,*args,**kwargs))
        self._invoke_cond.wait()
        self._invoke_cond.release()

    @tracedm
    def _onInvoke(self,evt):
        """ Gets called on receiving the OnInvoke event. Executes the events
            payload.
        """
        self._invoke_cond.acquire()
        evt.invoke()
        self._invoke_cond.notify()
        self._invoke_cond.release()









EVT_DUMMY_TYPE = wx.NewEventType()
EVT_DUMMY      = wx.PyEventBinder(EVT_DUMMY_TYPE)
class DummyEvent(wx.PyEvent):
    @tracedm
    def __init__(self):
        wx.PyEvent.__init__(self)
        self.SetEventType(EVT_DUMMY_TYPE)


class IdleGui(SimpleGui):
    @tracedm
    def __init__(self):
        SimpleGui.__init__(self)

        self._idle_condition = threading.Condition()
        self.Bind( wx.EVT_IDLE, self._onIdle)

        # initialize timer
        TIMER_ID = 777
        self.timer = wx.Timer(self, TIMER_ID)
        self.timer.Start(500)
        wx.EVT_TIMER(self, TIMER_ID, self._onTimer)


    @tracedm
    def _onIdle(self,evt):
        self._idle_condition.acquire()
        self._idle_condition.notifyAll()
        self._idle_condition.release()


    @tracedm
    def _waitForIdle(self):
        self._idle_condition.acquire()
        self._idle_condition.wait()
        self._idle_condition.release()








