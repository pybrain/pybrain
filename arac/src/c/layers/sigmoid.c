#include "common.h"
#include "sigmoid.h"


void
make_sigmoid_layer(Layer* layer_p, int dim)
{
    make_layer(layer_p, dim, dim);
    layer_p->type = SIGMOID_LAYER;
}


Layer* 
make_sigmoid_layer(int dim)
{
    Layer* layer_p = make_layer(dim, dim);
    layer_p->type = SIGMOID_LAYER;
    return layer_p;
}


void layer_forward(Layer* layer_p, SigmoidLayer* sl_p)
{
    layer_map_forward(layer_p, sigmoid);
}


void layer_backward(Layer* layer_p, SigmoidLayer* sl_p)
{
    layer_map_backward(layer_p, sigmoid_prime);
}


