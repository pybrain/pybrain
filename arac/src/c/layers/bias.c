#include <cassert>
#include "bias.h"

void
make_bias_layer(Layer* layer_p)
{
    make_layer(layer_p, 1, 1);
    layer_p->type = BIAS_LAYER;
}


Layer* 
make_bias_layer()
{
    Layer* layer_p = make_layer(1, 1);
    layer_p->type = BIAS_LAYER;
    return layer_p;
}


void layer_forward(Layer* layer_p, BiasLayer* bl_p)
{
    layer_p->outputs.contents_p[0] = 1.0;
}


void layer_backward(Layer* layer_p, BiasLayer* bl_p)
{
    ;
}

