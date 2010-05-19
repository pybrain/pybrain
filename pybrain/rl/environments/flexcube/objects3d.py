__author__ = 'Frank Sehnke, sehnke@in.tum.de'

from OpenGL.GL import * #@UnusedWildImport
from OpenGL.GLU import * #@UnusedWildImport
import math


class Objects3D:
    def normale(self, vect, centerOfGrav):
        vect = self.dumpVect(vect, 1.0 / 4.0)
        norm = self.difVect(vect, centerOfGrav)
        norm = self.normVect(norm, 1.0)
        return norm

    def drawCreature(self, cPoints, centerOfGrav):
        glBegin(GL_QUADS)

        #unten
        point = []
        summe = [0.0, 0.0, 0.0]
        for i in range(2):
            for k in range(2):
                grayI = i
                grayK = i ^ k
                point.append(cPoints[grayI * 4 + grayK])
                summe = self.addVect(summe, point[i * 2 + k])
        norm = self.normale(summe, centerOfGrav)
        glNormal(norm[0], norm[1], norm[2])
        for i in range(2):
            for k in range(2):
                glVertex3f(point[i * 2 + k][0], point[i * 2 + k][1], point[i * 2 + k][2]);

        #links
        point = []
        summe = [0.0, 0.0, 0.0]
        for j in range(2):
            for k in range(2):
                grayJ = j
                grayK = j ^ k
                point.append(cPoints[grayJ * 2 + grayK])
                summe = self.addVect(summe, point[j * 2 + k])
        norm = self.normale(summe, centerOfGrav)
        glNormal(norm[0], norm[1], norm[2])
        for j in range(2):
            for k in range(2):
                glVertex3f(point[j * 2 + k][0], point[j * 2 + k][1], point[j * 2 + k][2]);

        #rechts
        point = []
        summe = [0.0, 0.0, 0.0]
        for j in range(2):
            for k in range(2):
                grayJ = j
                grayK = j ^ k
                point.append(cPoints[4 + grayJ * 2 + grayK])
                summe = self.addVect(summe, point[j * 2 + k])
        norm = self.normale(summe, centerOfGrav)
        glNormal(norm[0], norm[1], norm[2])
        for j in range(2):
            for k in range(2):
                glVertex3f(point[j * 2 + k][0], point[j * 2 + k][1], point[j * 2 + k][2]);

        #oben
        point = []
        summe = [0.0, 0.0, 0.0]
        for i in range(2):
            for k in range(2):
                grayI = i
                grayK = i ^ k
                point.append(cPoints[grayI * 4 + 2 + grayK])
                summe = self.addVect(summe, point[i * 2 + k])
        norm = self.normale(summe, centerOfGrav)
        glNormal(norm[0], norm[1], norm[2])
        for i in range(2):
            for k in range(2):
                glVertex3f(point[i * 2 + k][0], point[i * 2 + k][1], point[i * 2 + k][2]);

        #vorne
        point = []
        summe = [0.0, 0.0, 0.0]
        for i in range(2):
            for j in range(2):
                grayI = i
                grayJ = i ^ j
                point.append(cPoints[grayI * 4 + grayJ * 2 + 1])
                summe = self.addVect(summe, point[i * 2 + j])
        norm = self.normale(summe, centerOfGrav)
        glNormal(norm[0], norm[1], norm[2])
        for i in range(2):
            for j in range(2):
                glVertex3f(point[i * 2 + j][0], point[i * 2 + j][1], point[i * 2 + j][2]);

        #hinten
        point = []
        summe = [0.0, 0.0, 0.0]
        for i in range(2):
            for j in range(2):
                grayI = i
                grayJ = i ^ j
                point.append(cPoints[grayI * 4 + grayJ * 2])
                summe = self.addVect(summe, point[i * 2 + j])
        norm = self.normale(summe, centerOfGrav)
        glNormal(norm[0], norm[1], norm[2])
        for i in range(2):
            for j in range(2):
                glVertex3f(point[i * 2 + j][0], point[i * 2 + j][1], point[i * 2 + j][2]);
        glEnd()

    def drawMirCreat(self, cPoints, centerOfGrav):
        glBegin(GL_QUADS)

        #unten
        point = []
        summe = [0.0, 0.0, 0.0]
        for i in range(2):
            for k in range(2):
                grayI = i
                grayK = i ^ k
                point.append(cPoints[grayI * 4 + grayK])
                summe = self.addVect(summe, point[i * 2 + k])
        norm = self.normale(summe, centerOfGrav)
        glNormal(norm[0], norm[1], norm[2])
        for i in range(2):
            for k in range(2):
                glVertex3f(point[i * 2 + k][0], -point[i * 2 + k][1], point[i * 2 + k][2])

        #links
        point = []
        summe = [0.0, 0.0, 0.0]
        for j in range(2):
            for k in range(2):
                grayJ = j
                grayK = j ^ k
                point.append(cPoints[grayJ * 2 + grayK])
                summe = self.addVect(summe, point[j * 2 + k])
        norm = self.normale(summe, centerOfGrav)
        glNormal(norm[0], norm[1], norm[2])
        for j in range(2):
            for k in range(2):
                glVertex3f(point[j * 2 + k][0], -point[j * 2 + k][1], point[j * 2 + k][2])

        #rechts
        point = []
        summe = [0.0, 0.0, 0.0]
        for j in range(2):
            for k in range(2):
                grayJ = j
                grayK = j ^ k
                point.append(cPoints[4 + grayJ * 2 + grayK])
                summe = self.addVect(summe, point[j * 2 + k])
        norm = self.normale(summe, centerOfGrav)
        glNormal(norm[0], norm[1], norm[2])
        for j in range(2):
            for k in range(2):
                glVertex3f(point[j * 2 + k][0], -point[j * 2 + k][1], point[j * 2 + k][2])

        #oben
        point = []
        summe = [0.0, 0.0, 0.0]
        for i in range(2):
            for k in range(2):
                grayI = i
                grayK = i ^ k
                point.append(cPoints[grayI * 4 + 2 + grayK])
                summe = self.addVect(summe, point[i * 2 + k])
        norm = self.normale(summe, centerOfGrav)
        glNormal(norm[0], norm[1], norm[2])
        for i in range(2):
            for k in range(2):
                glVertex3f(point[i * 2 + k][0], -point[i * 2 + k][1], point[i * 2 + k][2]);

        #vorne
        point = []
        summe = [0.0, 0.0, 0.0]
        for i in range(2):
            for j in range(2):
                grayI = i
                grayJ = i ^ j
                point.append(cPoints[grayI * 4 + grayJ * 2 + 1])
                summe = self.addVect(summe, point[i * 2 + j])
        norm = self.normale(summe, centerOfGrav)
        glNormal(norm[0], norm[1], norm[2])
        for i in range(2):
            for j in range(2):
                glVertex3f(point[i * 2 + j][0], -point[i * 2 + j][1], point[i * 2 + j][2])

        #hinten
        point = []
        summe = [0.0, 0.0, 0.0]
        for i in range(2):
            for j in range(2):
                grayI = i
                grayJ = i ^ j
                point.append(cPoints[grayI * 4 + grayJ * 2])
                summe = self.addVect(summe, point[i * 2 + j])
        norm = self.normale(summe, centerOfGrav)
        glNormal(norm[0], norm[1], norm[2])
        for i in range(2):
            for j in range(2):
                glVertex3f(point[i * 2 + j][0], -point[i * 2 + j][1], point[i * 2 + j][2])
        glEnd()

    def drawShadow(self, cPoints, centerOfGrav):
        glBegin(GL_QUADS)
        zPers = -0.5
        xPers = -0.33

        #schatten
        #unten
        point = []
        for i in range(2):
            for k in range(2):
                grayI = i
                grayK = i ^ k
                point.append([cPoints[grayI * 4 + grayK][0] + xPers * cPoints[grayI * 4 + grayK][1], -0.025, cPoints[grayI * 4 + grayK][2] + zPers * cPoints[grayI * 4 + grayK][1]])
        glNormal(0.0, 1.0, 0.0)
        for i in range(2):
            for k in range(2):
                glVertex3f(point[i * 2 + k][0], point[i * 2 + k][1], point[i * 2 + k][2])

        #links
        point = []
        summe = [0.0, 0.0, 0.0]
        for j in range(2):
            for k in range(2):
                grayJ = j
                grayK = j ^ k
                #point.append(cPoints[grayJ*2+grayK].pos)
                point.append([cPoints[grayJ * 2 + grayK][0] + xPers * cPoints[grayJ * 2 + grayK][1], -0.02, cPoints[grayJ * 2 + grayK][2] + zPers * cPoints[grayJ * 2 + grayK][1]])
                summe = self.addVect(summe, point[j * 2 + k])
        norm = self.normale(summe, centerOfGrav)
        glNormal(norm[0], norm[1], norm[2])
        for j in range(2):
            for k in range(2):
                glVertex3f(point[j * 2 + k][0], -point[j * 2 + k][1], point[j * 2 + k][2])

        #rechts
        point = []
        summe = [0.0, 0.0, 0.0]
        for j in range(2):
            for k in range(2):
                grayJ = j
                grayK = j ^ k
                #point.append(cPoints[4+grayJ*2+grayK].pos)
                point.append([cPoints[4 + grayJ * 2 + grayK][0] + xPers * cPoints[4 + grayJ * 2 + grayK][1], -0.015, cPoints[4 + grayJ * 2 + grayK][2] + zPers * cPoints[4 + grayJ * 2 + grayK][1]])
                summe = self.addVect(summe, point[j * 2 + k])
        norm = self.normale(summe, centerOfGrav)
        glNormal(norm[0], norm[1], norm[2])
        for j in range(2):
            for k in range(2):
                glVertex3f(point[j * 2 + k][0], -point[j * 2 + k][1], point[j * 2 + k][2])

        #oben
        point = []
        summe = [0.0, 0.0, 0.0]
        for i in range(2):
            for k in range(2):
                grayI = i
                grayK = i ^ k
                #point.append(cPoints[grayI*4+2+grayK].pos)
                point.append([cPoints[grayI * 4 + 2 + grayK][0] + xPers * cPoints[grayI * 4 + 2 + grayK][1], -0.01, cPoints[grayI * 4 + 2 + grayK][2] + zPers * cPoints[grayI * 4 + 2 + grayK][1]])
                summe = self.addVect(summe, point[i * 2 + k])
        norm = self.normale(summe, centerOfGrav)
        glNormal(norm[0], norm[1], norm[2])
        for i in range(2):
            for k in range(2):
                glVertex3f(point[i * 2 + k][0], -point[i * 2 + k][1], point[i * 2 + k][2]);

        #vorne
        point = []
        summe = [0.0, 0.0, 0.0]
        for i in range(2):
            for j in range(2):
                grayI = i
                grayJ = i ^ j
                #point.append(cPoints[grayI*4+grayJ*2+1].pos)
                point.append([cPoints[grayI * 4 + grayJ * 2 + 1][0] + xPers * cPoints[grayI * 4 + grayJ * 2 + 1][1], -0.005, cPoints[grayI * 4 + grayJ * 2 + 1][2] + zPers * cPoints[grayI * 4 + grayJ * 2 + 1][1]])
                summe = self.addVect(summe, point[i * 2 + j])
        norm = self.normale(summe, centerOfGrav)
        glNormal(norm[0], norm[1], norm[2])
        for i in range(2):
            for j in range(2):
                glVertex3f(point[i * 2 + j][0], -point[i * 2 + j][1], point[i * 2 + j][2]);

        #hinten
        point = []
        summe = [0.0, 0.0, 0.0]
        for i in range(2):
            for j in range(2):
                grayI = i
                grayJ = i ^ j
                #point.append(cPoints[grayI*4+grayJ*2].pos)
                point.append([cPoints[grayI * 4 + grayJ * 2][0] + xPers * cPoints[grayI * 4 + grayJ * 2][1], -0.0, cPoints[grayI * 4 + grayJ * 2][2] + zPers * cPoints[grayI * 4 + grayJ * 2][1]])
                summe = self.addVect(summe, point[i * 2 + j])
        norm = self.normale(summe, centerOfGrav)
        glNormal(norm[0], norm[1], norm[2])
        for i in range(2):
            for j in range(2):
                glVertex3f(point[i * 2 + j][0], -point[i * 2 + j][1], point[i * 2 + j][2])
        glEnd()

    def difVect(self, point1, point2):
        vect = [point1[0] - point2[0], point1[1] - point2[1], point1[2] - point2[2]]
        return vect

    def addVect(self, point1, point2):
        vect = [point1[0] + point2[0], point1[1] + point2[1], point1[2] + point2[2]]
        return vect

    def velDif(self, vect, dif, soll):
        zug = self.d * (soll - dif)
        dif = [vect[0] / dif * zug, vect[1] / dif * zug, vect[2] / dif * zug]
        return dif

    def dumpVect(self, vect, fakt):
        for i in range(3):
            vect[i] *= fakt
        return vect

    def normVect(self, vect, norm):
        summe = 0.0
        for i in range(3):
            summe += vect[i] * vect[i]
        vect = self.dumpVect(vect, norm / math.sqrt(summe))
        return vect

    def calcNormal(self, xVector, yVector):
        result = [0, 0, 0]
        result[0] = xVector[1] * yVector[2] - yVector[1] * xVector[2]
        result[1] = -xVector[0] * yVector[2] + yVector[0] * xVector[2]
        result[2] = xVector[0] * yVector[1] - yVector[0] * xVector[1]
        return [result[0], result[1], result[2]]

    def points2Vector(self, startPoint, endPoint):
        result = [0, 0, 0]
        result[0] = endPoint[0] - startPoint[0]
        result[1] = endPoint[1] - startPoint[1]
        result[2] = endPoint[2] - startPoint[2]
        return [result[0], result[1], result[2]]

