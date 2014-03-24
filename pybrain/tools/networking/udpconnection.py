__author__ = 'Frank Sehnke, sehnke@in.tum.de'

#############################################################################################################
# UDP Connection classes                                                                                    #
#                                                                                                           #
# UDPServer waits till at least one client is connected.                                                    #
# It then sends a list to the connected clients (can also be a list of scipy arrays!)                       #
# There can connect several clients to the server but the same data is sent to all clients.                 #
# Options for the constructor are the server IP and the starting port (2 adjacent ports will be used)       #
#                                                                                                           #
# UDPClient trys to connect to a UDPServer till the connection is established.                              #
# The client then recives data from the server and parses it into an list of the original shape.            #
# Options for the cunstructor are server-, client IP and the starting port (2 adjacent ports will be used)  #
#                                                                                                           #
# Requirements: sockets and scipy.                                                                          #
# Example: FlexCubeEnvironment and FlexCubeRenderer (env sends data to renderer for OpenGL output)          #
#                                                                                                           #
#############################################################################################################

import socket

# The server class
class UDPServer(object):
    def __init__(self, ip="127.0.0.1", port="21560", buf="1024"):
        #Socket settings
        self.host = ip
        self.inPort = eval(port) + 1
        self.outPort = eval(port)
        self.buf = eval(buf) #16384
        self.addr = (self.host, self.inPort)

        #Create socket and bind to address
        self.UDPInSock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.UDPInSock.bind(self.addr)

        #Client lists
        self.clients = 0
        self.cIP = []
        self.addrList = []
        self.UDPOutSockList = []
        print("listening on port", self.inPort)

    # Adding a client to the list
    def addClient(self, cIP):
        self.cIP.append(cIP)
        self.addrList.append((cIP, self.outPort))
        self.UDPOutSockList.append(socket.socket(socket.AF_INET, socket.SOCK_DGRAM))
        print("client", cIP, "connected")
        self.clients += 1

    # Listen for clients
    def listen(self):
        if self.clients < 1:
            self.UDPInSock.settimeout(10)
            try:
                cIP = self.UDPInSock.recv(self.buf)
                self.addClient(cIP)
            except:
                pass
        else:
            # At least one client has to send a sign of life during 2 seconds
            self.UDPInSock.settimeout(2)
            try:
                cIP = self.UDPInSock.recv(self.buf)
                newClient = True
                for i in self.cIP:
                    if cIP == i:
                        newClient = False
                        break
                #Adding new client
                if newClient:
                    self.addClient(cIP)
            except:
                print("All clients disconnected")
                self.clients = 0
                self.cIP = []
                self.addrList = []
                self.UDPOutSockList = []
                print("listening on port", self.inPort)


    # Sending the actual data too all clients
    def send(self, arrayList):
        sendString = repr(arrayList)
        count = 0
        for i in self.UDPOutSockList:
            i.sendto(sendString, self.addrList[count])
            count += 1

# The client class
class UDPClient(object):
    def __init__(self, servIP="127.0.0.1", ownIP="127.0.0.1", port="21560", buf="1024"):
        #UDP Sttings
        self.host = servIP
        self.inPort = eval(port)
        self.outPort = eval(port) + 1
        self.inAddr = (ownIP, self.inPort)
        self.outAddr = (self.host, self.outPort)
        self.ownIP = ownIP
        self.buf = eval(buf) #16384

        # Create sockets
        self.createSockets()

    # Listen for data from server
    def listen(self, arrayList=None):
        # Send alive signal (own IP adress)
        self.UDPOutSock.sendto(self.ownIP, self.outAddr)
        # if there is no data from Server for 10 seconds server is propably down
        self.UDPInSock.settimeout(10)
        try:
            data = self.UDPInSock.recv(self.buf)

            try:
                arrayList = eval(data)
                return arrayList
            except:
                print("Unsupported data format received from", self.outAddr, "!")
                return None

        except:
            print("Server has quit!")
            return None
            # Try to recreate sockets
            #self.createSockets()

    # Creating the sockets
    def createSockets(self):
        self.UDPOutSock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.UDPOutSock.sendto(self.ownIP, self.outAddr)
        self.UDPInSock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.UDPInSock.bind(self.inAddr)

