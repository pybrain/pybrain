from __future__ import print_function

__author__ = 'Tom Schaul, tom@idsia.ch'


from .handling import XMLHandling

# those imports are necessary for the eval() commands to find the right classes
import pybrain #@UnusedImport
from scipy import array #@UnusedImport


try:
    import arac.pybrainbridge #@UnusedImport
except ImportError:
    pass


class NetworkReader(XMLHandling):
    """ A class that can take read a network from an XML file """

    mothers = {}
    modules = {}

    @staticmethod
    def readFrom(filename, name = None, index = 0):
        """ append the network to an existing xml file

        :key name: if this parameter is specified, read the network with this name
        :key index: which network in the file shall be read (if there is more than one)
        """
        r = NetworkReader(filename, newfile = False)
        if name:
            netroot = r.findNamedNode('Network', name)
        else:
            netroot = r.findNode('Network', index)

        return r.readNetwork(netroot)

    def readNetwork(self, node):
        # TODO: why is this necessary?
        import pybrain.structure.networks.custom #@Reimport @UnusedImport
        nclass = eval(str(node.getAttribute('class')))
        argdict = self.readArgs(node)
        n = nclass(**argdict)
        n.name = node.getAttribute('name')

        for mnode in self.getChildrenOf(self.getChild(node, 'Modules')):
            m, inmodule, outmodule = self.readModule(mnode)
            if inmodule:
                n.addInputModule(m)
            elif outmodule:
                n.addOutputModule(m)
            else:
                n.addModule(m)

        mconns = self.getChild(node, 'MotherConnections')
        if mconns:
            for mcnode in self.getChildrenOf(mconns):
                m = self.readBuildable(mcnode)
                self.mothers[m.name] = m

        for cnode in self.getChildrenOf(self.getChild(node, 'Connections')):
            c, recurrent = self.readConnection(cnode)
            if recurrent:
                n.addRecurrentConnection(c)
            else:
                n.addConnection(c)

        n.sortModules()
        return n

    def readModule(self, mnode):
        if mnode.nodeName == 'Network':
            m = self.readNetwork(mnode)
        else:
            m = self.readBuildable(mnode)
        self.modules[m.name] = m
        inmodule = mnode.hasAttribute('inmodule')
        outmodule = mnode.hasAttribute('outmodule')
        return m, inmodule, outmodule

    def readConnection(self, cnode):
        c = self.readBuildable(cnode)
        recurrent = cnode.hasAttribute('recurrent')
        return c, recurrent

    def readBuildable(self, node):
        mclass = node.getAttribute('class')
        argdict = self.readArgs(node)
        try:
            m = eval(mclass)(**argdict)
        except:
            print(('Could not construct', mclass))
            print(('with arguments:', argdict))
            return None
        m.name = node.getAttribute('name')
        self.readParams(node, m)
        return m

    def readArgs(self, node):
        res = {}
        for c in self.getChildrenOf(node):
            val = c.getAttribute('val')
            if val in self.modules:
                res[str(c.nodeName)] = self.modules[val]
            elif val in self.mothers:
                res[str(c.nodeName)] = self.mothers[val]
            elif val != '':
                res[str(c.nodeName)] = eval(val)
        return res

    def readParams(self, node, m):
        pnode = self.getChild(node, 'Parameters')
        if pnode:
            params = eval(pnode.firstChild.data.strip())
            m._setParameters(params)
