__author__ = 'Thomas Rueckstiess, ruecksti@in.tum.de'

from environment import Environment

class GraphicalEnvironment(Environment):
    """ Special type of environment that has graphical output and therefore needs a renderer.
    """

    def __init__(self):
        self.renderer = None

    def setRenderer(self, renderer):
        """ set the renderer, which is an object of or inherited from class Renderer.

            :key renderer: The renderer that should display the Environment
            :type renderer: L{Renderer}
            :see: Renderer
        """
        self.renderer = renderer

    def getRenderer(self):
        """ returns the current renderer.

            :return: the current renderer
            :rtype: L{Renderer}
        """
        return self.renderer

    def hasRenderer(self):
        """ tells you whether or not a Renderer has been set previously

            :return: True if a renderer was set, False otherwise
            :rtype: Boolean
        """
        return (self.getRenderer() != None)
