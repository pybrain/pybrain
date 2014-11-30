from __future__ import print_function

__author__ = 'Michael Isik'

class KWArgDsc(object):
    def __init__(self, name, **kwargs):
        self.name = name
        self.private = False
        self.mandatory = False
        keys = ['private', 'default', 'mandatory']
        for key in keys:
            if key in kwargs:
                setattr(self, key, kwargs[key])

        assert not (self.mandatory and self.hasDefault())

    def hasDefault(self):
        return hasattr(self, 'default')



class KWArgsProcessor(object):
    def __init__(self, obj, kwargs):
#       self.argDscs = []
        self._object = obj
        self._obj_kwargs = kwargs

    def add(self, name, **kwargs):
        kwargDsc = KWArgDsc(name, **kwargs)
#        self.argDscs.append(ad)

        # determine attribute name
        name = kwargDsc.name
        if kwargDsc.private:
            attrname = "_" + name
        else:
            attrname = name

        # set the objects attribute
        if name in self._obj_kwargs:
            # set attribute supplied value
            setattr(self._object, attrname, self._obj_kwargs[name])
        elif kwargDsc.hasDefault():
            # set attribute to default value
            setattr(self._object, attrname, kwargDsc.default)
        elif kwargDsc.mandatory:
            raise KeyError('Mandatory Keyword argument "%s" missing!' % name)
        # del kwargs[name]



if __name__ == '__main__':
    class C(object):

        b = property(lambda self: self._b) # b will be readonly

        def __init__(self, **kwargs):
            kp = KWArgsProcessor(self, kwargs)
            kp.add('simple')

            kp.add('a', default=33)
            kp.add('b', private=True, default=55)
            kp.add('c', default=self.a + self._b)
            kp.add('m', mandatory=True)

        def __str__(self):
            return str(dict(self.__dict__))

    c1 = C(m=1)
    print(('c1 =', c1))

    c2 = C(m=1, a=1, b=2)
    print(('c2 =', c2))

    c3 = C(m=1, simple="hallo", a=11, b=22, c=55)
    print(('c3 =', c3))


    print(("\nc3.b = ", c3.b))

    try:
        C() # will raise KeyError because mandatory keyword argument "m" is missing
    except KeyError as k:
        print(k)



