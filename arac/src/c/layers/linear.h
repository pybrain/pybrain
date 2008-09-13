// Part of Arac Neural Network Composition Library.
// (c) 2008 by Justin S Bayer, <bayer.justin@googlemail.com>

//
// Header file for the LinearLayer.
//


#ifndef Arac_LAYER_LINEAR_INCLUDED
#define Arac_LAYER_LINEAR_INCLUDED


#include "common.h"


struct LinearLayer {
};


//
// Create an identity layer. 
//

Layer* make_linear_layer(int dim);

void make_linear_layer(Layer* layer_p, int dim);


//
// Process the input buffer of a module to the output buffer of module without 
// changes.
//

void layer_forward(Layer* layer_p, LinearLayer* il_p);

void layer_backward(Layer* layer_p, LinearLayer* il_p);


double identity(double x);


double identity_prime(double x);

#endif

