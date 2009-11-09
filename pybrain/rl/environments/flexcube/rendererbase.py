__author__ = 'Frank Sehnke, sehnke@in.tum.de'

import threading

class Renderer(threading.Thread):
    """ The general interface for a class displays what is happening in an environment. 
        The renderer is executed as concurrent thread. Start the renderer with the function
        start() inherited from Thread, and check with isAlive(), if the thread is running.
    """
    
    def __init__(self):
        """ initializes some variables and parent init functions """ 
        threading.Thread.__init__(self)   
    
    def updateData(self):
        """ overwrite this class to update whatever data the renderer needs to display the current
            state of the world. """
        pass
    
    def _render(self): 
        """ Here, the render methods are called. This function has to be implemented by subclasses. """
        pass 
    
    def start(self):
        """ wrapper for Thread.start(). only calls start if thread has not been started before. """
        if not self.isAlive():
            threading.Thread.start(self)
        
    def run(self):
        """ Don't call this function on its own. Use start() instead. """
        self._render()

    def stop(self):  
        """ stop signal requested. stop current thread.
            @note: only if possible. OpenGL glutMainLoop is not stoppable. 
        """
        pass 

    

