// Part of Arac Neural Network Composition Library.
// (c) 2008 by Justin S Bayer, <bayer.justin@googlemail.com>

//
// Header file for IdentityConnection.
//


#ifndef Arac_CONNECTIONS_IDENTITY_INCLUDED
#define Arac_CONNECTIONS_IDENTITY_INCLUDED

// Forward Declaration
struct Connection;          
struct Layer;         

// 
// Struct to represent linear connections.
//  

struct IdentityConnection {
};


// 
// Connect two modules with linear connection,
//
// A linear connection just passes its values on. Thus, if module A passes 
// data to module B, the outgoing dimension of A must be the incoming dimension
// of B.
//

void identity_connect(Layer* inlayer_p, Layer* outlayer_p);

//
// TODO: document
//

void identity_connect(Layer* inlayer_p, Layer* outlayer_p,
                      int inlayerstart, int inlayerstop, 
                      int outlayerstart, int outlayerstop);



//
// Sepcific forward for IdentityConnection.
//
//
// The output of one layer is just passed to the input of another layer of the 
// same size.
//

void connection_forward(Connection* con_p, IdentityConnection* ic_p);

void connection_backward(Connection* con_p, IdentityConnection* ic_p);


#endif

