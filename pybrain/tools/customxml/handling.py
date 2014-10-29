__author__ = 'Tom Schaul, tom@idsia.ch'

from xml.dom.minidom import parse, getDOMImplementation
from pybrain.utilities import fListToString
from scipy import zeros
import string

class XMLHandling:
    """ general purpose methods for reading, writing and editing XML files.
    This class should wrap all the XML-specific code, and then be subclassed
    by specialized readers/writers that use its methods.

    The priority is on readability and usability for the subclasses, not efficiency.
    """

    def __init__(self, filename, newfile):
        """ :key newfile: is the file to be read or is it a new file? """
        self.filename = filename
        if not newfile:
            self.dom = parse(filename)
            if self.dom.firstChild.nodeName != 'PyBrain':
                raise Exception('Not a correct PyBrain XML file')
        else:
            domimpl = getDOMImplementation()
            self.dom = domimpl.createDocument(None, 'PyBrain', None)
        self.root = self.dom.documentElement

    def save(self):
        file = open(self.filename, 'w')
        file.write(self.dom.toprettyxml())
        file.close()

    def readAttrDict(self, node, transform = None):
        """ read a dictionnary of attributes
        :key transform: optionally function transforming the attribute values on reading """
        args = {}
        for name, val in node.attributes.items():
            name = str(name)
            if transform != None:
                args[name] = transform(val, name)
            else:
                args[name] = val
        return args

    def writeAttrDict(self, node, adict, transform = None):
        """ read a dictionnary of attributes

        :key transform: optionally transform the attribute values on writing """
        for name, val in adict.items():
            if val != None:
                if transform != None:
                    node.setAttribute(name, transform(val, name))
                else:
                    node.setAttribute(name, val)

    def newRootNode(self, name):
        return self.newChild(self.root, name)

    def newChild(self, node, name):
        """ create a new child of node with the provided name. """
        elem = self.dom.createElement(name)
        node.appendChild(elem)
        return elem

    def addTextNode(self, node, text):
        tmp = self.dom.createTextNode(text)
        node.appendChild(tmp)

    def getChild(self, node, name):
        """ get the child with the given name """
        for n in node.childNodes:
            if name and n.nodeName == name:
                return n

    def getChildrenOf(self, node):
        """ get the element children """
        return filter(lambda x: x.nodeType == x.ELEMENT_NODE, node.childNodes)

    def findNode(self, name, index = 0, root = None):
        """ return the toplevel node with the provided name (if there are more, choose the
        index corresponding one). """
        if root == None:
            root = self.root
        for n in root.childNodes:
            if n.nodeName == name:
                if index == 0:
                    return n
                index -= 1
        return None

    def findNamedNode(self, name, nameattr, root = None):
        """ return the toplevel node with the provided name, and the fitting 'name' attribute. """
        if root == None:
            root = self.root
        for n in root.childNodes:
            if n.nodeName == name:
# modif JPQ
#                if 'name' in n.attributes:
                if n.attributes['name']:
# modif JPQ
#                    if n.attributes['name'] == nameattr:
                    if n.attributes['name'].value == nameattr:
                        return n
        return None

    def writeDoubles(self, node, l, precision = 6):
        self.addTextNode(node, fListToString(l, precision)[2:-1])

    def writeMatrix(self, node, m, precision = 6):
        for i, row in enumerate(m):
            r = self.newChild(node, 'row')
            self.writeAttrDict(r, {'number':str(i)})
            self.writeDoubles(r, row, precision)

    def readDoubles(self, node):
        dstrings = string.split(node.firstChild.data)
        return map(lambda s: float(s), dstrings)

    def readMatrix(self, node):
        rows = []
        for c in self.getChildrenOf(node):
            rows.append(self.readDoubles(c))
        if len(rows) == 0:
            return None
        res = zeros((len(rows), len(rows[0])))
        for i, r in enumerate(rows):
            res[i] = r
        return res


def baseTransform(val):
    """ back-conversion: modules are encoded by their name
    and classes by the classname """
    from pybrain.structure.modules.module import Module
    from inspect import isclass

    if isinstance(val, Module):
        return val.name
    elif isclass(val):
        return val.__name__
    else:
        return str(val)
