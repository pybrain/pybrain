// Part of Arac Neural Network Composition Library.
// (c) 2008 by Justin S Bayer, <bayer.justin@googlemail.com>

//
// Header file for the SigmoidLayer.
//


#ifndef Arac_LAYER_SIGMOID_INCLUDED
#define Arac_LAYER_SIGMOID_INCLUDED


#include <cmath>
#include "../functions.h"


//
// Create a sigmoid layer. 
//

Layer* make_sigmoid_layer(int dim);

void make_sigmoid_layer(Layer* layer_p, int dim);


//
// Process the input buffer of a layer to the output buffer of that layer
// and apply the sigmoid squashing function to every cell.
//

void layer_forward(Layer* layer_p, SigmoidLayer* sl_p);

void layer_backward(Layer* layer_p, SigmoidLayer* sl_p);


#endif

