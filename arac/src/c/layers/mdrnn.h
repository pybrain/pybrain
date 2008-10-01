//
// MDRNN Layer, as described in "Multi-Dimensional Recurrent Neural Networks"
// by Alex Graves, Santiago Fernández, Jürgen Schmidhuber
//
// This Layer does not support a backward pass.
//

//
// Author: Justin Bayer, bayer.justin@googlemail.com;
//


#ifndef Arac_MDRNN_LINEAR_INCLUDED
#define Arac_MDRNN_LINEAR_INCLUDED

#include "common.h"


struct MdrnnLayer {
    // Dimensions of the sequence.
    int timedim;
    // Shape of the sequence.
    int* shape_p;
    // Shape of an input vector.
    int* blockshape_p;

    // Length of the whole sequence.
    // (Redundant: equals product of the shape.)
    int sequence_length;

    // Size of a single input vector.
    // (Redundant: equals the product of the blockshape.)
    int indim;

    // Amout of hidden cells per dimension.
    int hiddendim;

    // Size of a single output vector
    int outdim;

    // States of the MDLSTM-cells
    double* cell_states_p;

    // In order to use the functionality of arac, an MDRNN-Layer builds up a
    // certain internal network. The network consists of several layers itself.

    // A single hiddenlayer object is saved here in order to move it over the
    // sequence.
    Layer* hiddenlayer_p;

    // Output layer that is used as an output for the swiping hidden layer.
    Layer* outlayer_p;

    // The hiddenlayer gets inputs from several other layers:
    // A layer that manages the inputs ...
    Layer* swipe_inlayer_p;
    // ... and timedim layers that manage the states and outputs of the
    // predecessors ...
    Layer* swipe_predlayers_p;
    // ... and then there's a bias.
    Layer* bias_p;
};


Layer*
make_mdrnn_layer(int timedim, int* shape_p, int* blockshape_p,
                 int hiddendim, int outdim,
                 double** in_to_hidden_weights_p,
                 double** hidden_to_out_weights_p);


void
make_mdrnn_layer(Layer* layer_p,
                 int timedim, int* shape_p, int* blockshape_p,
                 int hiddendim, int outdim,
                 double** in_to_hidden_weights_p,
                 double** hidden_to_out_weights_p);


void layer_forward(Layer* layer_p, MdrnnLayer* mdrnn_p);


void layer_backward(Layer* layer_p, MdrnnLayer* mdrnn_p);


#endif