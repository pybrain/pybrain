__author__ = 'Frank Sehnke, sehnke@in.tum.de'

from environment import Environment

class GraphicalEnvironment(Environment):
    """ Special type of environment that has graphical output and therefore needs a renderer.
    """

    def __init__(self):
        self.renderInterface = None

    def setRenderInterface(self, renderer):
        """ set the renderer, which is an object of or inherited from class Renderer.

            :arg renderer: The renderer that should display the Environment
            :type renderer: L{Renderer}
            .. seealso:: :class:`Renderer`
        """
        self.renderInterface = renderer

    def getRenderInterface(self):
        """ returns the current renderer.

            :return: the current renderer
            :rtype: L{Renderer}
        """
        return self.renderInterface

    def hasRenderInterface(self):
        """ tells you, if a Renderer has been set previously or not

            :return: True if a renderer was set, False otherwise
            :rtype: Boolean
        """
        return (self.getRenderInterface() != None)
