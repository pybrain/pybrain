// Part of Arac Neural Network Composition Library.
// (c) 2008 by Justin S Bayer, <bayer.justin@googlemail.com>

//
// Header file for the TanhLayer.
//


#ifndef Arac_LAYER_TANH_INCLUDED
#define Arac_LAYER_TANH_INCLUDED


#include <cmath>
#include "../functions.h"


//
// Create a tanh layer.
//

Layer* make_tanh_layer(int dim);

void make_tanh_layer(Layer* layer_p, int dim);


//
// Process the input buffer of a layer to the output buffer of that layer
// and apply the tanh squashing function to every cell.
//

void layer_forward(Layer* layer_p, TanhLayer* tl_p);

void layer_backward(Layer* layer_p, TanhLayer* tl_p);


#endif
