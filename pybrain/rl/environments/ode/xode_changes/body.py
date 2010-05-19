#@PydevCodeAnalysisIgnore

######################################################################
# Python Open Dynamics Engine Wrapper
# Copyright (C) 2004 PyODE developers (see file AUTHORS)
# All rights reserved.
#
# This library is free software; you can redistribute it and/or
# modify it under the terms of EITHER:
#   (1) The GNU Lesser General Public License as published by the Free
#       Software Foundation; either version 2.1 of the License, or (at
#       your option) any later version. The text of the GNU Lesser
#       General Public License is included with this library in the
#       file LICENSE.
#   (2) The BSD-style license that is included with this library in
#       the file LICENSE-BSD.
#
# This library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the files
# LICENSE and LICENSE-BSD for more details.
######################################################################

# XODE Importer for PyODE

"""
XODE Body and Mass Parser
@author: U{Timothy Stranex<mailto:timothy@stranex.com>}
"""

import ode
import errors, node, joint, transform, geom

class Body(node.TreeNode):
    """
    Represents an ode.Body object and corresponds to the <body> tag.
    """

    def __init__(self, name, parent, attrs):
        node.TreeNode.__init__(self, name, parent)
        world = parent.getFirstAncestor(ode.World)
        self.setODEObject(ode.Body(world.getODEObject()))

        enabled = attrs.get('enabled', 'true')
        if (enabled not in ['true', 'false']):
            raise errors.InvalidError("Enabled attribute must be either 'true'"\
                                      " or 'false'.")
        else:
            if (enabled == 'false'):
                self.getODEObject().disable()

        gravitymode = int(attrs.get('gravitymode', 1))
        if (gravitymode == 0):
            self.getODEObject().setGravityMode(0)

        self._mass = None
        self._transformed = False

    def takeParser(self, parser):
        """
        Handle further parsing. It should be called immediately after the <body>
        tag has been encountered.
        """

        self._parser = parser
        self._parser.push(startElement=self._startElement,
                          endElement=self._endElement)

    def _applyTransform(self):
        if (self._transformed): return

        t = self.getTransform()

        body = self.getODEObject()
        body.setPosition(t.getPosition())
        body.setRotation(t.getRotation())

        self._transformed = True

    def _startElement(self, name, attrs):
        nodeName = attrs.get('name', None)

        if (name == 'transform'):
            t = transform.Transform()
            t.takeParser(self._parser, self, attrs)
        else:
            self._applyTransform()

        if (name == 'torque'):
            self.getODEObject().setTorque(self._parser.parseVector(attrs))
        elif (name == 'force'):
            self.getODEObject().setForce(self._parser.parseVector(attrs))
        elif (name == 'finiteRotation'):
            mode = int(attrs['mode'])

            try:
                axis = (float(attrs['xaxis']),
                        float(attrs['yaxis']),
                        float(attrs['zaxis']))
            except KeyError:
                raise errors.InvalidError('finiteRotation element must have' \
                                          ' xaxis, yaxis and zaxis attributes')

            if (mode not in [0, 1]):
                raise errors.InvalidError('finiteRotation mode attribute must' \
                                          ' be either 0 or 1.')

            self.getODEObject().setFiniteRotationMode(mode)
            self.getODEObject().setFiniteRotationAxis(axis)
        elif (name == 'linearVel'):
            self.getODEObject().setLinearVel(self._parser.parseVector(attrs))
        elif (name == 'angularVel'):
            self.getODEObject().setAngularVel(self._parser.parseVector(attrs))
        elif (name == 'mass'):
            self._mass = Mass(nodeName, self)
            self._mass.takeParser(self._parser)
        elif (name == 'joint'):
            j = joint.Joint(nodeName, self)
            j.takeParser(self._parser)
        elif (name == 'body'):
            b = Body(nodeName, self, attrs)
            b.takeParser(self._parser)
        elif (name == 'geom'):
            g = geom.Geom(nodeName, self)
            g.takeParser(self._parser)
        elif (name == 'transform'): # so it doesn't raise ChildError
            pass
        else:
            raise errors.ChildError('body', name)

    def _endElement(self, name):
        if (name == 'body'):
            self._parser.pop()

            self._applyTransform()
            if (self._mass is not None):
                self.getODEObject().setMass(self._mass.getODEObject())

class Mass(node.TreeNode):
    """
    Represents an ode.Mass object and corresponds to the <mass> tag.
    """

    def __init__(self, name, parent):
        node.TreeNode.__init__(self, name, parent)

        mass = ode.Mass()
        mass.setSphere(1.0, 1.0)
        self.setODEObject(mass)

        body = self.getFirstAncestor(ode.Body)
        body.getODEObject().setMass(mass)

    def takeParser(self, parser):
        """
        Handle further parsing. It should be called immediately after the <mass>
        tag is encountered.
        """

        self._parser = parser
        self._parser.push(startElement=self._startElement,
                          endElement=self._endElement)

    def _startElement(self, name, attrs):
        nodeName = attrs.get('name', None)

        if (name == 'mass_struct'):
            pass
        elif (name == 'mass_shape'):
            self._parseMassShape(attrs)
        elif (name == 'transform'):
            # parse transform
            pass
        elif (name == 'adjust'):
            total = float(attrs['total'])
            self.getODEObject().adjust(total)
        elif (name == 'mass'):
            mass = Mass(nodeName, self)
            mass.takeParser(self._parser)
        else:
            raise errors.ChildError('mass', name)

    def _endElement(self, name):
        if (name == 'mass'):
            try:
                mass = self.getFirstAncestor(ode.Mass)
            except node.AncestorNotFoundError:
                pass
            else:
                mass.getODEObject().add(self.getODEObject())
            self._parser.pop()

    def _parseMassShape(self, attrs):
        density = attrs.get('density', None)
        mass = self.getODEObject()

        def start(name, attrs):
            if (name == 'sphere'):
                radius = float(attrs.get('radius', 1.0))
                if (density is not None):
                    mass.setSphere(float(density), radius)
            elif (name == 'box'):
                lx = float(attrs['sizex'])
                ly = float(attrs['sizey'])
                lz = float(attrs['sizez'])
                if (density is not None):
                    mass.setBox(float(density), lx, ly, lz)
            elif (name == 'cappedCylinder'):
                radius = float(attrs.get('radius', 1.0))
                length = float(attrs['length'])
                if (density is not None):
                    mass.setCappedCylinder(float(density), 3, radius, length)
            elif (name == 'cylinder'):
                radius = float(attrs.get('radius', 1.0))
                length = float(attrs['length'])
                if (density is not None):
                    mass.setCylinder(float(density), 3, radius, length)
            else:
                # FIXME: Implement remaining mass shapes.
                raise NotImplementedError()

        def end(name):
            if (name == 'mass_shape'):
                self._parser.pop()

        self._parser.push(startElement=start, endElement=end)

