// Part of Arac Neural Network Composition Library.
// (c) 2008 by Justin S Bayer, <bayer.justin@googlemail.com>

//
// Header file for common functionality of connections..
//


#ifndef Arac_CONNECTIONS_COMMON_INCLUDED
#define Arac_CONNECTIONS_COMMON_INCLUDED


#define IDENTITY_CONNECTION 0
#define FULL_CONNECTION 1


// Forward declarations
struct Layer;
struct IdentityConnection;
struct FullConnection;


//
// Union for accessing all different types of connections.
//

union AnyConnection {
    IdentityConnection* identity_connection_p;
    FullConnection* full_connection_p;
};


//
// Struct to connect two Layers. There is an incoming layer whose output is 
// passed on to the input of the outgoing layer.
// The way the data is passed is specific to the layer type and the forward and
// backward implementations.
// 

struct Connection {
    Layer* inlayer_p;
    Layer* outlayer_p;

    int recurrent;
    
    int inlayerstart;
    int inlayerstop;
    int outlayerstart;
    int outlayerstop;

    int type;
    AnyConnection internal;
};


//
// Append an copy of a Connection to a collection array. 
//
// Allocates space for power of two elements automatically.
//
void append_to_array(Connection*& begin, int& counter, Connection con);


#endif

