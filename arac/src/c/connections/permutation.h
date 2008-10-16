// Part of Arac Neural Network Composition Library.
// (c) 2008 by Justin S Bayer, <bayer.justin@googlemail.com>

//
// Header file for the PermutationConnection.
//


#ifndef Arac_CONNECTIONS_PERMUTATION_INCLUDED
#define Arac_CONNECTIONS_PERMUTATION_INCLUDED

#include "common.h"

// 
// Struct to represent full connections.
//

struct PermutationConnection {
    int* permutation_p;
    int* invpermutation_p;
    int blocksize;
};


// 
// Connect two layers with a permutation connection.
//

void 
permutation_connect(Layer* inlayer_p, Layer* outlayer_p, 
                    int* permutation_p, int blocksize);

//
// TODO: document
//

void
permutation_connect(Layer* inlayer_p, Layer* outlayer_p, 
                    int* permutation_p, int blocksize,
                    int inlayerstart, int inlayerstop, 
                    int outlayerstart, int outlayerstop);


//
// Sepcific forward for PermuteConnection. Values are passed on in linear
// manner but permutet according to the given permutation.
//

void connection_forward(Connection* con_p, PermutationConnection* c_p);

void connection_backward(Connection* con_p, PermutationConnection* c_p);


#endif

