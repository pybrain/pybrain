__author__ = 'Frank Sehnke, sehnke@in.tum.de'


class MArray:
    def __init__(self):
        self.field = {}
    def get(self, i):
        return self.field[i]
    def set(self, i, val):
        self.field[i] = val

class MassPoint:
    def __init__(self):
        self.pos = [0.0, 0.0, 0.0]
        self.vel = [0.0, 0.0, 0.0]
    def setPos(self, pos):
        self.pos = pos
    def setVel(self, vel):
        self.vel = vel

