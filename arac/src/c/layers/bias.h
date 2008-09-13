// Part of Arac Neural Network Composition Library.
// (c) 2008 by Justin S Bayer, <bayer.justin@googlemail.com>

//
// Header file for the BiasLayer.
//


#ifndef Arac_LAYER_BIAS_INCLUDED
#define Arac_LAYER_BIAS_INCLUDED

#include "common.h"

//
// Create a bias layer. 
//
Layer* make_bias_layer();

void make_bias_layer(Layer* layer_p);

void layer_forward(Layer* layer_p, BiasLayer* bl_p);

void layer_backward(Layer* layer_p, BiasLayer* bl_p);


#endif

