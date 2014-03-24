__author__ = 'Thomas Rueckstiess, ruecksti@in.tum.de'

import sys

class XMLstruct:
    """
    Defines an XML tag structure. Tags are added at the current level
    using the insert() method, with up() providing a way to get to the
    parent tag.

    $Id:xmltools.py 150 2007-04-11 13:42:47Z ruecksti $
    """

    _tab = "\t"

    def __init__(self, name, attr=None):
        """create a new tag at the topmost level using given name
        and (optional) attribute dictionary"""
        # XML tag structure is a dictionary containing all attributes plus
        # two special tags:
        #   myName = name of the tag
        #   Icontain = list of XML tags this one encloses
        self.tag = {}
        self.tag['myName'] = name
        if attr is not None:
            self.tag.update(attr)

        # to traverse the XML hierarchy, store the current tag and
        # the previously visited ones. Start at the top level.
        self.top()

    def __iter__(self):
        """returns the top-level list of tags (including internal tags 'Icontain' and 'myName')"""
        return self.tag


    def insert(self, name, attr=None):
        """Insert a new tag into the current one. The name can be either the
        new tag name or an XMLstruct object (in which case attr is ignored).
        Unless name is None, we descend into the new tag as a side effect.
        A dictionary is expected for attr."""
        if not self.current.hasSubtag():
            self.current.tag['Icontain'] = []
        if name == None:
            # empty subtag list inserted, return now
            # (this produces <tag></tag> in the output)
            return
        elif type(name) == str:
            # create a new subtag with given name and attributes
            newtag = XMLstruct(name, attr)
        else:
            # input assumed to be a tag structure
            newtag = name
        self.current.tag['Icontain'].append(newtag)
        self.stack.append(self.current)
        self.current = newtag


    def insertMulti(self, attrlist):
        """Inserts multiple subtags at once. A list of XMLstruct objects
        must be given; the tag hierarchy is not descended into."""
        if not self.current.hasSubtag():
            self.current.tag['Icontain'] = []
        self.current.tag['Icontain'] += attrlist


    def downTo(self, name, stack=None, current=None):
        """Traverse downward from current tag, until given named tag is found. Returns
        true if found and sets stack and current tag correspondingly."""
        if stack is None:
            stack = self.stack
            current = self.current
        if self.name == name:
            return(True)
        else:
            if not self.hasSubtag():
                return(False)
            else:
                # descend one level
                stack.append(self)
                found = self.getSubtag(name)
                if found is None:
                    # no subtag of correct name found; recursively check whether
                    # any of the subtags contain the looked for tag
                    for subtag in self.tag['Icontain']:
                        found = subtag.downTo(name, stack, current)
                        if found:
                            # everything was updated recursively, need only return
                            return(True)
                    # nothing found, revert stack and return
                    stack.pop()
                    return(False)

                else:
                    current.setCurrent(found)
                    return(True)


    def up(self, steps=1):
        """traverse upward a number of steps in tag stack"""
        for _ in range(steps):
            if self.stack != []:
                self.current = self.stack.pop()

    def top(self):
        """traverse upward to root level"""
        self.stack = []
        self.current = self

    def setCurrent(self, tag):
        self.current = tag

    def getName(self):
        """return tag's name"""
        return(self.tag['myName'])

    def getCurrentSubtags(self):
        if self.current.hasSubtag():
            return(self.current.tag['Icontain'])
        else:
            return([])

    def hasSubtag(self, name=None):
        """determine whether current tag contains other tags, and returns
        the tag with a matching name (if name is given) or True (if not)"""
        if self.tag.has_key('Icontain'):
            if name is None:
                return(True)
            else:
                for subtag in self.tag['Icontain']:
                    if subtag.name == name: return(True)
        return(False)


    def getSubtag(self, name=None):
        """determine whether current tag contains other tags, and returns
        the tag with a matching name (if name is given) or None (if not)"""
        if self.tag.has_key('Icontain'):
            for subtag in self.tag['Icontain']:
                if subtag.name == name: return(subtag)
        return(None)

    def nbAttributes(self):
        """return number of user attributes the current tag has"""
        nAttr = len(self.tag.keys()) - 1
        if self.hasSubtag():
            nAttr -= 1
        return nAttr


    def scale(self, sc, scaleset=set([]), exclude=set([])):
        """for all tags not in the exclude set, scale all attributes whose names are in scaleset by the given factor"""
        if self.name not in exclude:
            for name, val in self.tag.iteritems():
                if name in scaleset:
                    self.tag[name] = val * sc
        if self.hasSubtag():
            for subtag in self.tag['Icontain']:
                subtag.scale(sc, scaleset, exclude)


    def write(self, file, depth=0):
        """parse XML structure recursively and append to the output fileID,
        increasing the offset (tabs) while descending into the tree"""
        if not self.tag.has_key('myName'):
            print("Error parsing XML structure: Tag name missing!")
            sys.exit(1)
        # find number of attributes (disregarding special keys)
        nAttr = self.nbAttributes()
        endmark = '/>'
        if self.hasSubtag():
            endmark = '>'
        # print(start tag, with attributes if present)
        if nAttr > 0:
            file.write(self._tab * depth + "<" + self.tag['myName'] + " " + \
                ' '.join([name + '="' + str(val) + '"' for name, val in self.tag.iteritems() \
                if name != 'myName' and name != 'Icontain']) + endmark + '\n')
        else:
            file.write(self._tab * depth + "<" + self.tag['myName'] + ">\n")
        # print(enclosed tags, if any)
        if self.hasSubtag():
            for subtag in self.tag['Icontain']:
                subtag.write(file, depth=depth + 1)
            # finalize tag
            file.write(self._tab * depth + "</" + self.tag['myName'] + ">\n")


