#@PydevCodeAnalysisIgnore
from OpenGL.GLUT import *
from OpenGL.GL import *
from OpenGL.GLE import *
from OpenGL.GLU import *
import objects3D
from time import *
from scipy import ones, zeros, array, clip, arange, sqrt
from socket import *
import string


class FlexCubeRenderer(object): 
  def __init__(self, servIP, ownIP):
      self.oldScreenValues = None
      self.view=0
      self.worldRadius = 400
                
      # Start of mousepointer 
      self.lastx = 0
      self.lasty = 15
      self.lastz= 300
      self.zDis=1
   
      # Start of cube 
      self.cube = [0.0, 0.0, 0.0]
      self.bmpCount=0
      self.actCount=0
      self.calcPhysics=0
      self.newPic=1
      self.picCount=0
      self.target=[80.0,0.0,0.0]
      
      self.centerOfGrav=array([0.0,-2.0,0.0])
      self.points=ones((8,3),float)
      self.savePics=False
      self.drawCounter=0
      self.fps=25
      self.dt=1.0/float(self.fps)

      #UDP Sttings
      self.host = servIP
      self.inPort = 21560
      self.outPort = 21561
      self.inAddr = (ownIP,self.inPort)
      self.outAddr = (self.host,self.outPort)
      self.ownIP=ownIP

      # Create sockets
      self.UDPOutSock = socket(AF_INET,SOCK_DGRAM)
      self.UDPOutSock.sendto(ownIP,self.outAddr)
      self.UDPInSock = socket(AF_INET,SOCK_DGRAM)
      self.UDPInSock.bind(self.inAddr)


  def listen(self):
      # Receive messages
      self.UDPOutSock.sendto(self.ownIP,self.outAddr)
      buf=1024
      self.UDPInSock.settimeout(10)
      try:
        data = self.UDPInSock.recv(buf)
        try:
          data = string.split(str(data), " ")
          self.parse(data)
        except:
          print "Unsupportet data format recived from", self.outAddr, "!"
      except:
        print "Server has quit!"
        # Try to recreate sockets
        self.UDPOutSock = socket(AF_INET,SOCK_DGRAM)
        self.UDPOutSock.sendto(self.ownIP,self.outAddr)
        self.UDPInSock = socket(AF_INET,SOCK_DGRAM)
        self.UDPInSock.bind(self.inAddr)

  def parse(self, data):
      i=0
      j=0
      c=0
      p=False
      while not p:
          self.points[i][j]=eval(data[c])
          c+=1
          j+=1
          if j==3:
              j=0
              i+=1
          if i==8: 
              i=0
              p=True

      p=False
      while not p:
          self.centerOfGrav[i]=eval(data[c])
          c+=1
          i+=1
          if i==3: 
              i=0
              p=True
        
  def saveTo( self, filename, format="JPEG" ):
    """Save current buffer to filename in format"""
    import Image # get PIL's functionality...
    width, height = 800,600
    glPixelStorei(GL_PACK_ALIGNMENT, 1)
    data = glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE)
    image = Image.fromstring( "RGB", (width, height), data )
    image = image.transpose( Image.FLIP_TOP_BOTTOM)
    image.save( filename, format )
    print 'Saved image to ', filename
    return image

  def setTarget(self, target):
    self.target=target[:]
    
  def updateData(self, pos, sensors):
    self.points=pos.copy()
    self.centerOfGrav=sensors.copy()
    
  def _render(self):
    # Call init: Parameter(Window Position -> x, y, height, width)
    self.init_GL(self,300,300,800,600)
    
    self.object = objects3D.Objects3D()
    self.quad = gluNewQuadric()
    glutMainLoop()
    
  def drawScene(self):
    ''' This methode describes the complete scene.'''
    # clear the buffer
    if self.zDis<10: self.zDis+=0.25
    if self.lastz>200: self.lastz-= self.zDis
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()
    
    # Point of view
    glTranslatef(-self.centerOfGrav[0], -self.centerOfGrav[1], -self.centerOfGrav[2]-self.lastz)
    glRotatef(self.lastx, 0.0, 1.0, 0.0)
    glRotatef(self.lasty, 1.0, 0.0, 0.0)      
    
    # Calculates an rotation angle depending an a time factor
    self.rotation=time() % 359

    #Objects
    #Target Ball
    glColor3f(1,0.25,0.25)
    glPushMatrix()
    glTranslate(self.target[0], 0.0, self.target[2])
    glutSolidSphere(1.5, 8,8)
    glPopMatrix()
    
    #Massstab
    for lk in range(41):
      if float(lk-20)/10.0 == (lk-20)/10: 
        glColor3f(0.75,0.75,0.75)
        glPushMatrix()
        glRotatef(90,1,0,0)
        glTranslate(self.worldRadius/40.0*float(lk)-self.worldRadius/2.0, -40.0, -30)
        quad=gluNewQuadric()
        gluCylinder(quad,2, 2, 60, 4, 1);
        glPopMatrix()
      else: 
        if float(lk-20)/5.0 == (lk-20)/5:       
          glColor3f(0.75,0.75,0.75)
          glPushMatrix()
          glRotatef(90,1,0,0)
          glTranslate(self.worldRadius/40.0*float(lk)-self.worldRadius/2.0, -40.0, -15.0)
          quad=gluNewQuadric()
          gluCylinder(quad,1, 1, 30, 4, 1);
          glPopMatrix()
        else:
          glColor3f(0.75,0.75,0.75)
          glPushMatrix()
          glRotatef(90,1,0,0)
          glTranslate(self.worldRadius/40.0*float(lk)-self.worldRadius/2.0, -40.0, -7.5)
          quad=gluNewQuadric()
          gluCylinder(quad,0.5, 0.5, 15, 4, 1);
          glPopMatrix()

    #Mirror Center Ball
    glColor3f(1,1,1)
    glPushMatrix()
    glTranslate(self.centerOfGrav[0], -self.centerOfGrav[1], self.centerOfGrav[2])
    glutSolidSphere(1.5, 8,8)
    glPopMatrix()
    
    #Mirror Cube
    glEnable (GL_BLEND)
    glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    
    glColor4f(0.5,0.75,0.5,0.75)
    glPushMatrix()
    glTranslatef(0, -0.05, 0)
    self.object.drawMirCreat(self.points, self.centerOfGrav)
    glPopMatrix()
    
    # Floor
    tile=self.worldRadius/40.0
    glEnable (GL_BLEND)
    glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    
    for xF in range(40):
        for yF in range(40):
            if float(xF+yF)/2.0 == (xF+yF)/2: glColor3f(0.8, 0.8, 0.7)
            else: glColor4f(0.8, 0.8, 0.8,0.9)
            glPushMatrix()
            glTranslatef(0.0, -0.03, 0.0)
            glBegin(GL_QUADS)
            glNormal(0.0,1.0,0.0)      
            for i in range(2):
              for k in range(2):
                glVertex3f((i+xF-20)*tile,0.0,((k^i)+yF-20)*tile);
            glEnd()
            glPopMatrix()
 
    #Center Ball
    glColor3f(1,1,1)
    glPushMatrix()
    glTranslate(self.centerOfGrav[0], self.centerOfGrav[1], self.centerOfGrav[2])
    glutSolidSphere(1.5, 8,8)
    glPopMatrix()
              	
    # Frank   
    glEnable (GL_BLEND)
    glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    
    glColor4f(0.5,0.75,0.5,0.75)
    glPushMatrix()
    self.object.drawCreature(self.points, self.centerOfGrav)
    glPopMatrix()
    
    glEnable (GL_BLEND)
    glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    
    glColor4f(0,0,0,0.5)
    glPushMatrix()
    self.object.drawShadow(self.points, self.centerOfGrav)
    glPopMatrix()
    
    # swap the buffer
    glutSwapBuffers()    
    
  def drawIdleScene(self):
      self.listen()
      self.drawScene()
      if self.savePics:
          self.saveTo("./screenshots2/image"+repr(10000+self.picCount)+".jpg")
          self.picCount+=1
      sleep(self.dt)
      
  def resizeScene(self, width, height):
    '''Needed if window size changes.'''
    if height == 0:						# Prevent A Divide By Zero If The Window Is Too Small 
      height = 1

    glViewport(0, 0, width, height)		# Reset The Current Viewport And Perspective Transformation
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45.0, float(width)/float(height), 0.1, 700.0)
    glMatrixMode(GL_MODELVIEW)

  def activeMouse(self, x, y):
    '''Returns mouse coordinates while any mouse button is pressed.'''
    # store the mouse coordinate
    if self.mouseButton == GLUT_LEFT_BUTTON:
      self.lastx = x - self.xOffset
      self.lasty = y -self.yOffset
    if self.mouseButton == GLUT_RIGHT_BUTTON:
      self.lastz = y - self.zOffset 
    # redisplay
    glutPostRedisplay()
  
  def passiveMouse(self, x, y):
    '''Returns mouse coordinates while no mouse button is pressed.'''
    pass
    
  def completeMouse(self, button, state, x, y):
    '''Returns mouse coordinates and which button was pressed resp. released.'''
    self.mouseButton = button
    if state == GLUT_DOWN:
      self.xOffset = x - self.lastx
      self.yOffset = y - self.lasty
      self.zOffset = y - self.lastz
    # redisplay
    glutPostRedisplay()
    
  '''Initialise an OpenGL windows with the origin at x, y and size of height, width.'''
  def init_GL(self, pyWorld, x, y, height, width):
    # initialize GLUT 
    glutInit([])
      
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH)
    glutInitWindowSize(height, width)
    glutInitWindowPosition(x,y)
    glutCreateWindow("The Curious Cube")
    glClearDepth(1.0)
    glEnable(GL_DEPTH_TEST)
    glClearColor(0.0, 0.0, 0.0, 0.0)
    glShadeModel(GL_SMOOTH)
    glMatrixMode(GL_MODELVIEW)
    # initialize lighting */
    glLightfv(GL_LIGHT0, GL_DIFFUSE, [1,1,1, 1.0])
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
    r=FlexCubeRenderer(sys.argv[1], sys.argv[2])
    r._render()
