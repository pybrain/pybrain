

__author__ = 'Michael Isik'



class KWArgDsc(object):
    def __init__(self, name, **kwargs):
        self.name = name
        self.private = False
        keys=['private','default']
        for key in keys:
            if kwargs.has_key(key):
                setattr(self,key,kwargs[key])


class KWArgsProcessor(object):
    def __init__(self, obj, kwargs):
 #       self.argDscs = []
        self._object  = obj
        self._obj_kwargs  = kwargs

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
        if self._obj_kwargs.has_key(name):
            # set attribute supplied value
            setattr(self._object, attrname, self._obj_kwargs[name])
        elif hasattr(kwargDsc, 'default'):
            # set attribute to default value
            setattr(self._object, attrname, kwargDsc.default)
        # del kwargs[name]



if __name__ == '__main__':
    class C(object):

        b = property(lambda self: self._b) # b will be readonly

        def __init__(self, **kwargs):
            kp = KWArgsProcessor(self, kwargs)
            kp.add('simple')

            kp.add('a', default=33)
            kp.add('b', private=True, default=55)
            kp.add('c', default=self.a+self._b)

        def __str__(self):
            return str(dict(self.__dict__))

    c1 = C()
    print 'c1 =',c1

    c2 = C(a=1, b=2)
    print 'c2 =',c2
    
    c3 = C(simple="hallo", a=11, b=22, c=55)
    print 'c3 =',c3


    print "\nc3.b = ", c3.b

