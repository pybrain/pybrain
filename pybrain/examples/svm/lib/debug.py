
__author__ = 'Michael Isik'

import threading
import types



class Global:
    verbosity = 2
#    debug_enabled = True
    debug_enabled = False

def setVerbosity(v) : Global.verbosity  = v
def incVerbosity( ) : Global.verbosity += 1
def getVerbosity( ) : return Global.verbosity
def enableDebugging(v=True) : Global.debug_enabled=v


class Colors:
    black   = "\033[30m"
    red     = "\033[31m"
    green   = "\033[32m"
    yellow  = "\033[33m"
    blue    = "\033[34m"
    magenta = "\033[35m"
    cyan    = "\033[36m"
    white   = "\033[37m"


class ThreadInfo:
    def __init__(self,color):
        self.level = 0
        self.color = color

class DebugObject:
    colors = [
        Colors.black,
        Colors.red,
        Colors.green,
        Colors.yellow,
        Colors.blue,
        Colors.magenta,
        Colors.cyan
        ]

    def __init__(self):
        self._threadinfos = {}

    def _getThreadInfo(self):
        t = threading.currentThread()
        if not self._threadinfos.has_key(t):
            self._threadinfos[t] = ThreadInfo( DebugObject.colors[len(self._threadinfos)] )
        return self._threadinfos[t]


    def _getPrefix(self):
        ti = self._getThreadInfo()
        return ti.color + ( "  " * ti.level )

    def out(self, *args):
#        return
        print self._getPrefix(),
        for arg in args:
            print arg,
        print
#        print self._getPrefix(),*args,**kwargs
    def _funcin(self,funcname):
        self.out("+++ "+funcname)
        ti = self._getThreadInfo()
        ti.level+=1
    def _funcout(self,funcname):
        ti = self._getThreadInfo()
        ti.level-=1
        self.out("--- "+funcname)
    def printPrefix(self):
        print self._getPrefix(),


debug_object = DebugObject()


def doNothing(*args,**kwargs):pass

# import these
if Global.debug_enabled:
    dbg = debug_object.out
    pfx = debug_object.printPrefix
else:
    dbg = doNothing
    pfx = doNothing



def tracedm(*args,**kwargs):
    par_idx  = []
    show_ret = False
    if   kwargs.has_key('v') : v = kwargs['v']
    else                     : v = 2

    if len(args)==1 and not kwargs and type(args[0]) is types.FunctionType:
        be_decorator = True
        func         = args[0]
    else:
        be_decorator = False
        for i in args:
            if   i >=  0: par_idx.append(i)
            elif i == -1: show_ret = True

    def decorator(func):
        par_names = func.func_code.co_varnames
        if not Global.debug_enabled: return func

        def wrapper(self,*args,**kwargs):
            if Global.verbosity < v:
                return func(self,*args,**kwargs)
            else:
                funcname = self.__class__.__name__+"::"+func.__name__
                params=", ".join( par_names[i] + "=" + str(args[i-1]) for i in par_idx)
                debug_object._funcin(funcname+"("+params+")")
                
                try:
                    ret = func(self,*args,**kwargs)
                except:
                    debug_object._funcout(funcname+"()"+" exception")
                    raise
                    
                ret_str=""
                if show_ret:
                    ret_str  = " = " + str(ret)
                debug_object._funcout(funcname+"()"+ret_str)
                return ret

        return wrapper


    if be_decorator:
        return decorator(func)
    else:
        return decorator





