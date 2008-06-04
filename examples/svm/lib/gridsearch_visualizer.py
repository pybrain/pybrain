
__author__ = 'Michael Isik'

from visual import PlotPanel3D, SimpleGui
import wx

from lib.debug import dbg,tracedm

from numpy import array, append, sort, meshgrid


def vectorCompare(x,y):
    for i in range(len(x)):
        if   x[i] < y[i]: return -1
        elif x[i] > y[i]: return 1
    return 0

class GridSearchViewer(PlotPanel3D):
    @tracedm
    def __init__(self, grid_search, parent):
        PlotPanel3D.__init__(self, parent)

        self._grid_search = grid_search

    @tracedm
    def update(self):
        perfs = self._grid_search._performances
        xy = array(perfs.keys(),float)
        z  = array(perfs.values(),float)
        if not len(xy): return

#        xy.sort(vectorCompare)
#        print xy
#        exit(0)



        z   = z.reshape(len(z),1)
        xyz = list(append(xy, z, axis=1))
#        print xyz
#        xy.sort(vectorCompare,axis=0)
        xyz.sort(vectorCompare)
        xyz=array(xyz)
#        print xyz

        x = xyz[:,0]
        y = xyz[:,1]
        z = xyz[:,2]
        xdim = len(set(x))
        ydim = len(set(y))
        x = x.reshape(xdim,ydim)
        y = y.reshape(xdim,ydim)
        z = z.reshape(xdim,ydim)
#        print points
#        print values
        #points.sort(vectorCompare)
#        x = sort(list(set(data[:,0])))
#        y = sort(list(set(data[:,1])))
#        X,Y = meshgrid(x,y)
#        print x
#        print y
#        print z
#        print perfs
#        print "update"

        self.subplot.cla()
        self.subplot.contour3D( x,y,z )
        for p in xyz:
            #print p
            self.subplot.plot3D([p[0]],[p[1]],[p[2]],"go")


#        self.subplot.plot_surface( x,y,z )




#import time
class GridSearchVisualizer(SimpleGui):
    @tracedm
    def __init__(self, grid_search):
        SimpleGui.__init__(self)

        # setup the step event hook
        self._grid_search = grid_search
        self._grid_search._onStep = self._onGridSearchStep


        # initialize gui elements
        self._status_text=""
        self.CreateStatusBar()


        # integrate observers into gui
        self._observer = GridSearchViewer(self._grid_search, self)
        observer = self._observer

        self._sizer.Add(observer, wx.ID_ANY, wx.EXPAND)
        self._sizer.SetItemMinSize(observer,600, 600)


        self._refreshLayout()

        # refresh layout
        self._sizer.Fit(self)
        self.Layout()

        # start the gui
        self.start()
        self.doUpdate()
        self._observer.SetFocus()


    def _onGridSearchStep(self):
        self.doUpdate()

    @tracedm
    def doUpdate(self):
        self._invoke(self._update)

    @tracedm
    def _update(self):
        self._observer.update()
        self._observer._setSize()




