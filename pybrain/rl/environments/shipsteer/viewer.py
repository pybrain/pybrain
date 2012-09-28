__author__ = 'Frank Sehnke, sehnke@in.tum.de'

#@PydevCodeAnalysisIgnore
#########################################################################
# OpenGL viewer for the FlexCube Environment
#
# The FlexCube Environment is a Mass-Spring-System composed of 8 mass points.
# These resemble a cube with flexible edges.
#
# This viewer uses an UDP connection found in tools/networking/udpconnection.py
#
# The viewer recieves the position matrix of the 8 masspoints and the center of gravity.
# With this information it renders a Glut based 3d visualization of teh FlexCube
#
# Options:
# - serverIP: The ip of the server to which the viewer should connect
# - ownIP: The IP of the computer running the viewer
# - port: The starting port (2 adjacent ports will be used)
#
# Saving the images is possible by setting self.savePics=True.
# Changing the point and angle of view is possible by using the mouse
# while button 1 or 2 pressed.
#
# Requirements: OpenGL
#
#########################################################################

from OpenGL.GLUT import *
from OpenGL.GL import *
from OpenGL.GLE import *
from OpenGL.GLU import *
from time import sleep
from scipy import ones, array, cos, sin
from pybrain.tools.networking.udpconnection import UDPClient

class FlexCubeRenderer(object):
    #Options: ServerIP(default:localhost), OwnIP(default:localhost), Port(default:21560)
    def __init__(self, servIP="127.0.0.1", ownIP="127.0.0.1", port="21580"):
        self.oldScreenValues = None
        self.view = 0
        self.worldRadius = 400

        # Start of mousepointer
        self.lastx = 0
        self.lasty = 15
        self.lastz = 300
        self.zDis = 1

        # Start of cube
        self.cube = [0.0, 0.0, 0.0]
        self.bmpCount = 0
        self.actCount = 0
        self.calcPhysics = 0
        self.newPic = 1
        self.picCount = 0
        self.sensors = [0.0, 0.0, 0.0]
        self.centerOfGrav = array([0.0, 5.0, 0.0])

        self.savePics = False
        self.drawCounter = 0
        self.fps = 50
        self.dt = 1.0 / float(self.fps)
        self.step = 0

        self.client = UDPClient(servIP, ownIP, port)

    # If self.savePics=True this method saves the produced images
    def saveTo(self, filename, format="JPEG"):
        import Image # get PIL's functionality...
        width, height = 800, 600
        glPixelStorei(GL_PACK_ALIGNMENT, 1)
        data = glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE)
        image = Image.fromstring("RGB", (width, height), data)
        image = image.transpose(Image.FLIP_TOP_BOTTOM)
        image.save(filename, format)
        print('Saved image to ', filename)
        return image

    # the render method containing the Glut mainloop
    def _render(self):
        # Call init: Parameter(Window Position -> x, y, height, width)
        self.init_GL(self, 300, 300, 800, 600)
        self.quad = gluNewQuadric()
        glutMainLoop()

    # The Glut idle function
    def drawIdleScene(self):
        #recive data from server and update the points of the cube
        try: self.sensors = self.client.listen(self.sensors)
        except: pass
        if self.sensors == ["r", "r", "r"]: self.centerOfGrav = array([0.0, 5.0, 0.0])
        else:
            self.step += 1
            a = self.sensors[0] / 360.0 * 3.1428
            dir = array([cos(a), 0.0, -sin(a)])
            self.centerOfGrav += self.sensors[2] * dir * 0.02
            self.drawScene()
        if self.savePics:
            self.saveTo("./screenshots/image_jump" + repr(10000 + self.picCount) + ".jpg")
            self.picCount += 1
        else: sleep(self.dt)

    def drawScene(self):
        ''' This methode describes the complete scene.'''
        # clear the buffer
        if self.zDis < 10: self.zDis += 0.25
        if self.lastz > 100: self.lastz -= self.zDis
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()

        # Point of view
        glRotatef(self.lastx, 0.0, 1.0, 0.0)
        glRotatef(self.lasty, 1.0, 0.0, 0.0)
        #glRotatef(15, 0.0, 0.0, 1.0)
        # direction of view is aimed to the center of gravity of the cube
        glTranslatef(-self.centerOfGrav[0], -self.centerOfGrav[1] - 50.0, -self.centerOfGrav[2] - self.lastz)

        #Objects

        #Massstab
        for lk in range(41):
            if float(lk - 20) / 10.0 == (lk - 20) / 10:
                glColor3f(0.75, 0.75, 0.75)
                glPushMatrix()
                glRotatef(90, 1, 0, 0)
                glTranslate(self.worldRadius / 40.0 * float(lk) - self.worldRadius / 2.0, -40.0, -30)
                quad = gluNewQuadric()
                gluCylinder(quad, 2, 2, 60, 4, 1)
                glPopMatrix()
            else:
                if float(lk - 20) / 5.0 == (lk - 20) / 5:
                    glColor3f(0.75, 0.75, 0.75)
                    glPushMatrix()
                    glRotatef(90, 1, 0, 0)
                    glTranslate(self.worldRadius / 40.0 * float(lk) - self.worldRadius / 2.0, -40.0, -15.0)
                    quad = gluNewQuadric()
                    gluCylinder(quad, 1, 1, 30, 4, 1)
                    glPopMatrix()
                else:
                    glColor3f(0.75, 0.75, 0.75)
                    glPushMatrix()
                    glRotatef(90, 1, 0, 0)
                    glTranslate(self.worldRadius / 40.0 * float(lk) - self.worldRadius / 2.0, -40.0, -7.5)
                    quad = gluNewQuadric()
                    gluCylinder(quad, 0.5, 0.5, 15, 4, 1)
                    glPopMatrix()

        # Floor
        tile = self.worldRadius / 40.0
        glEnable (GL_BLEND)
        glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        glColor3f(0.8, 0.8, 0.5)
        glPushMatrix()
        glTranslatef(0.0, -3.0, 0.0)
        glBegin(GL_QUADS)
        glNormal(0.0, 1.0, 0.0)
        glVertex3f(-self.worldRadius, 0.0, -self.worldRadius)
        glVertex3f(-self.worldRadius, 0.0, self.worldRadius)
        glVertex3f(self.worldRadius, 0.0, self.worldRadius)
        glVertex3f(self.worldRadius, 0.0, -self.worldRadius)
        glEnd()
        glPopMatrix()

        #Water
        for xF in range(40):
            for yF in range(40):
                if float(xF + yF) / 2.0 == (xF + yF) / 2: glColor4f(0.7, 0.7, 1.0, 0.5)
                else: glColor4f(0.9, 0.9, 1.0, 0.5)
                glPushMatrix()
                glTranslatef(0.0, -0.03, 0.0)
                glBegin(GL_QUADS)
                glNormal(0.5 + sin(float(xF) + float(self.step) / 4.0) * 0.5, 0.5 + cos(float(xF) + float(self.step) / 4.0) * 0.5, 0.0)
                for i in range(2):
                    for k in range(2):
                        glVertex3f((i + xF - 20) * tile, sin(float(xF + i) + float(self.step) / 4.0) * 3.0, ((k ^ i) + yF - 20) * tile)
                glEnd()
                glPopMatrix()

        self.ship()
                  	
        # swap the buffer
        glutSwapBuffers()

    def ship(self):
        glColor3f(0.4, 0.1, 0.2)
        glPushMatrix()
        glTranslate(self.centerOfGrav[0] + 14, self.centerOfGrav[1], self.centerOfGrav[2])
        glRotatef(180 - self.sensors[0], 0.0, 1.0, 0.0)

        self.cuboid(0, 0, 0, 20, 5, 5)
        #bow of ship
        glBegin(GL_TRIANGLES)
        glNormal3fv(self.calcNormal(self.points2Vector([-5, 6, 2.5], [0, 5, 5]), self.points2Vector([-5, 6, 2.5], [0, 5, 0])))
        glVertex3f(-5, 6, 2.5), glVertex3f(0, 5, 0), glVertex3f(0, 5, 5)
        glNormal3fv(self.calcNormal(self.points2Vector([-5, 6, 2.5], [0, 0, 5]), self.points2Vector([-5, 6, 2.5], [0, 5, 5])))
        glVertex3f(-5, 6, 2.5), glVertex3f(0, 0, 5), glVertex3f(0, 5, 5)
        glNormal3fv(self.calcNormal(self.points2Vector([-5, 6, 2.5], [0, 0, 0]), self.points2Vector([-5, 6, 2.5], [0, 0, 5])))
        glVertex3f(-5, 6, 2.5), glVertex3f(0, 0, 5), glVertex3f(0, 0, 0)
        glNormal3fv(self.calcNormal(self.points2Vector([-5, 6, 2.5], [0, 5, 0]), self.points2Vector([-5, 6, 2.5], [0, 0, 0])))
        glVertex3f(-5, 6, 2.5), glVertex3f(0, 0, 0), glVertex3f(0, 5, 0)
        glEnd()
        # stern
        glPushMatrix()
        glRotatef(-90, 1.0, 0.0, 0.0)
        glTranslatef(15, -2.5, 0)
        gluCylinder(self.quad, 2.5, 2.5, 5, 10, 1)
        glTranslatef(0, 0, 5)
        gluDisk(self.quad, 0, 2.5, 10, 1)
        glPopMatrix()
        # deck
        if abs(self.sensors[0]) < 5.0: reward = (self.sensors[2] + 10.0) / 50.0
        else: reward = 0.2
        glColor3f(1.0 - reward, reward, 0)
        self.cuboid(5, 5, 1, 10, 8, 4)
        glPushMatrix()
        glRotatef(-90, 1.0, 0.0, 0.0)
        glTranslatef(13, -2.5, 5)
        glColor3f(1, 1, 1)
        gluCylinder(self.quad, 1, 0.8, 5, 20, 1)
        glPopMatrix()
        glPopMatrix()

    def cuboid(self, x0, y0, z0, x1, y1, z1):
        glBegin(GL_QUADS)
        glNormal(0, 0, 1)
        glVertex3f(x0, y0, z1); glVertex3f(x0, y1, z1); glVertex3f(x1, y1, z1); glVertex3f(x1, y0, z1)  #front
        glNormal(-1, 0, 0)
        glVertex3f(x0, y0, z0); glVertex3f(x0, y0, z1); glVertex3f(x0, y1, z1); glVertex3f(x0, y1, z0) # left
        glNormal(0, -1, 0)
        glVertex3f(x0, y0, z0); glVertex3f(x0, y0, z1); glVertex3f(x1, y0, z1); glVertex3f(x1, y0, z0) # bottom
        glNormal(0, 0, -1)
        glVertex3f(x0, y0, z0); glVertex3f(x1, y0, z0); glVertex3f(x1, y1, z0); glVertex3f(x0, y1, z0) # back
        glNormal(0, 1, 0)
        glVertex3f(x0, y1, z0); glVertex3f(x1, y1, z0); glVertex3f(x1, y1, z1); glVertex3f(x0, y1, z1) # top
        glNormal(1, 0, 0)
        glVertex3f(x1, y0, z0); glVertex3f(x1, y0, z1); glVertex3f(x1, y1, z1); glVertex3f(x1, y1, z0) # right
        glEnd()

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

    def resizeScene(self, width, height):
        '''Needed if window size changes.'''
        if height == 0: # Prevent A Divide By Zero If The Window Is Too Small
            height = 1

        glViewport(0, 0, width, height) # Reset The Current Viewport And Perspective Transformation
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45.0, float(width) / float(height), 0.1, 700.0)
        glMatrixMode(GL_MODELVIEW)

    def activeMouse(self, x, y):
        #Returns mouse coordinates while any mouse button is pressed.
        # store the mouse coordinate
        if self.mouseButton == GLUT_LEFT_BUTTON:
            self.lastx = x - self.xOffset
            self.lasty = y - self.yOffset
        if self.mouseButton == GLUT_RIGHT_BUTTON:
            self.lastz = y - self.zOffset
        # redisplay
        glutPostRedisplay()

    def passiveMouse(self, x, y):
        '''Returns mouse coordinates while no mouse button is pressed.'''
        pass

    def completeMouse(self, button, state, x, y):
        #Returns mouse coordinates and which button was pressed resp. released.
        self.mouseButton = button
        if state == GLUT_DOWN:
            self.xOffset = x - self.lastx
            self.yOffset = y - self.lasty
            self.zOffset = y - self.lastz
        # redisplay
        glutPostRedisplay()

    #Initialise an OpenGL windows with the origin at x, y and size of height, width.
    def init_GL(self, pyWorld, x, y, height, width):
        # initialize GLUT
        glutInit([])

        glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH)
        glutInitWindowSize(height, width)
        glutInitWindowPosition(x, y)
        glutCreateWindow("The Curious Cube")
        glClearDepth(1.0)
        glEnable(GL_DEPTH_TEST)
        glClearColor(0.0, 0.0, 0.0, 0.0)
        glShadeModel(GL_SMOOTH)
        glMatrixMode(GL_MODELVIEW)
        # initialize lighting */
        glLightfv(GL_LIGHT0, GL_DIFFUSE, [1, 1, 1, 1.0])
        glLightModelfv(GL_LIGHT_MODEL_AMBIENT, [1.0, 1.0, 1.0, 1.0])
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        #
        glColorMaterial(GL_FRONT, GL_DIFFUSE)
        glEnable(GL_COLOR_MATERIAL)
        # Automatic vector normalise
        glEnable(GL_NORMALIZE)

        ### Instantiate the virtual world ###
        glutDisplayFunc(pyWorld.drawScene)
        glutMotionFunc(pyWorld.activeMouse)
        glutMouseFunc(pyWorld.completeMouse)
        glutReshapeFunc(pyWorld.resizeScene)
        glutIdleFunc(pyWorld.drawIdleScene)

if __name__ == '__main__':
    s = sys.argv[1:]
    r = FlexCubeRenderer(*s)
    r._render()

