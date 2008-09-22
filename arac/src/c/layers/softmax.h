// Part of Arac Neural Network Composition Library.
// (c) 2008 by Justin S Bayer, <bayer.justin@googlemail.com>

//
// Header file for the SoftmaxLayer.
//


#ifndef Arac_LAYER_SOFTMAX_INCLUDED
#define Arac_LAYER_SOFTMAX_INCLUDED


#include "common.h"


struct SoftmaxLayer {
};


//
// Create an softmax layer.
//

Layer* make_softmax_layer(int dim);

void make_softmax_layer(Layer* layer_p, int dim);


//
// Process the input buffer to the output buffer by making their sum equal to 1
// and apply the exponential function to every cell before.
//

void layer_forward(Layer* layer_p, SoftmaxLayer* sm_p);

void layer_backward(Layer* layer_p, SoftmaxLayer* sm_p);


#endif

