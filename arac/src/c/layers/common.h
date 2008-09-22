// Part of Arac Neural Network Composition Library.
// (c) 2008 by Justin S Bayer, <bayer.justin@googlemail.com>

//
// Header file for common layer functionality.
//


#ifndef Arac_LAYERS_COMMON_INCLUDED
#define Arac_LAYERS_COMMON_INCLUDED

#include "../common.h"
#include "../connections/connections.h"

#define BIAS_LAYER 0
#define LINEAR_LAYER 1
#define SIGMOID_LAYER 2
#define TANH_LAYER 3
#define MDLSTM_LAYER 4
#define LSTM_LAYER 5
#define SOFTMAX_LAYER 6


// Forward declarations
struct BiasLayer;
struct LinearLayer;
struct LstmLayer;
struct MdLstmLayer;
struct SigmoidLayer;
struct SoftmaxLayer;
struct TanhLayer;


union AnyLayer {
    BiasLayer* bias_layer_p;
    LinearLayer* linear_layer_p;
    LstmLayer* lstm_layer_p;
    MdLstmLayer* mdlstm_layer_p;
    SigmoidLayer* sigmoid_layer_p;
    SoftmaxLayer* softmax_layer_p;
    TanhLayer* tanh_layer_p;
};


// 
// Struct to represent nodes of a network graph. Incoming input is transformed
// with the forward method, the error is given back with the backward method. 
// May have multiple Layers connected to its input and multiple Layers may 
// receive the output. Layers are connected via Connections.
//

struct Layer {
    ParameterContainer inputs;
    ParameterContainer outputs;
    
    int incoming_n;
    int outgoing_n;
    Connection* incoming_p;
    Connection* outgoing_p;
    
    int type;
    AnyLayer internal;
    
    int* timestep_p;
    int* seqlen_p;
};


//
// Prototype functions
// 

//
// Create a Layer by allocating space for it and prepolutating some fields.
//
Layer* make_layer(int input_dim, int output_dim);

void make_layer(Layer* layer_p, int input_dim, int output_dim);


// 
// Helper to apply a function to every input of the layer, writing the result 
// into the output buffer.
//

void layer_map_forward(Layer* layer_p, double (*mapper) (double));


//
// Helper to apply a function to every error of the layer, multiply it with the
// corresponding input and write the result into the input error.
//

void layer_map_backward(Layer* layer_p, double (*mapper) (double));


#endif

