#include "common.h"
#include "tanh.h"


void
make_tanh_layer(Layer* layer_p, int dim)
{
    make_layer(layer_p, dim, dim);
    layer_p->type = TANH_LAYER;
}


Layer* 
make_tanh_layer(int dim)
{
    Layer* layer_p = make_layer(dim, dim);
    layer_p->type = TANH_LAYER;
    return layer_p;
}


void layer_forward(Layer* layer_p, TanhLayer* tl_p)
{
    layer_map_forward(layer_p, tanh_);
}


void layer_backward(Layer* layer_p, TanhLayer* tl_p)
{
    layer_map_backward(layer_p, tanh_prime);
}


