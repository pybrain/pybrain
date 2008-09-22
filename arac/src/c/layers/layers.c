// Part of Arac Neural Network Composition Library.
// (c) 2008 by Justin S Bayer, <bayer.justin@googlemail.com>

#include <iostream>
#include "layers.h"


void forward(Layer* layer_p) 
{
    // The layer_forward function is overloaded for different types of 
    // layers. Due to the union structure, for the specialized functions the 
    // layers pointer is being passed in again, though not necessarily needed.

    switch (layer_p->type) 
    {
        case BIAS_LAYER:
            layer_forward(layer_p, layer_p->internal.bias_layer_p);
            break;
        case LINEAR_LAYER:
            layer_forward(layer_p, layer_p->internal.linear_layer_p);
            break;
        case SIGMOID_LAYER:
            layer_forward(layer_p, layer_p->internal.sigmoid_layer_p);
            break;
        case TANH_LAYER:
            layer_forward(layer_p, layer_p->internal.tanh_layer_p);
            break;
        case MDLSTM_LAYER:
            layer_forward(layer_p, layer_p->internal.mdlstm_layer_p);
            break;
        case LSTM_LAYER:
            layer_forward(layer_p, layer_p->internal.lstm_layer_p);
            break;
        case SOFTMAX_LAYER:
            layer_forward(layer_p, layer_p->internal.softmax_layer_p);
            break;

        default:
            std::cout << "Unknown Layertype for forward: " << layer_p->type 
                      << ". Dying." << std::endl;
            exit(1);
    }
}


void backward(Layer* layer_p) {
    // The layer_backward function is overloaded for different types of 
    // layers. Due to the union structure, for the specialized functions the 
    // layers pointer is being passed in again, though not necessarily needed.

    switch (layer_p->type)
    {
        case BIAS_LAYER:
            layer_backward(layer_p, layer_p->internal.bias_layer_p);
            break;
        case LINEAR_LAYER:
            layer_backward(layer_p, layer_p->internal.linear_layer_p);
            break;
        case SIGMOID_LAYER:
            layer_backward(layer_p, layer_p->internal.sigmoid_layer_p);
            break;
        case TANH_LAYER:
            layer_backward(layer_p, layer_p->internal.tanh_layer_p);
            break;
        case MDLSTM_LAYER:
            layer_backward(layer_p, layer_p->internal.mdlstm_layer_p);
            break;
        case LSTM_LAYER:
            layer_backward(layer_p, layer_p->internal.lstm_layer_p);
            break;
        case SOFTMAX_LAYER:
            layer_backward(layer_p, layer_p->internal.softmax_layer_p);
            break;

        default:
            std::cout << "Unknown Layertype for backward: " << layer_p->type 
                      << ". Dying." << std::endl;
            exit(1);
    }
}

