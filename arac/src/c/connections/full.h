// Part of Arac Neural Network Composition Library.
// (c) 2008 by Justin S Bayer, <bayer.justin@googlemail.com>

//
// Header file for the FullConnection.
//


#ifndef Arac_CONNECTIONS_FULL_INCLUDED
#define Arac_CONNECTIONS_FULL_INCLUDED

#include "common.h"

// 
// Struct to represent full connections.
//

struct FullConnection {
    ParameterContainer weights;
};


// 
// Connect two layers with a full connections.
//
// This is basically a parametrized connection that takes a matrix and uses 
// matrix multiplication as an intermediate step.
//

void 
full_connect(Layer* inlayer_p, Layer* outlayer_p, double* params);

//
// TODO: document
//

void
full_connect(Layer* inlayer_p, Layer* outlayer_p, 
             double* params, 
             int inlayerstart, int inlayerstop, 
             int outlayerstart, int outlayerstop);


//
// Sepcific forward for FullConnection. Values are passed on in a matrix 
// multiplicative manner
//

void connection_forward(Connection* con_p, FullConnection* fc_p);

void connection_backward(Connection* con_p, FullConnection* fc_p);





#endif

