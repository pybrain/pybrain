#include "softmax.h"
#include <cmath>
#include <cstring>
#include <iostream>

void
make_softmax_layer(Layer* layer_p, int dim)
{
    make_layer(layer_p, dim, dim);
    layer_p->type = SOFTMAX_LAYER;
}


Layer*
make_softmax_layer(int dim)
{
    Layer* layer_p = make_layer(dim, dim);
    layer_p->type = SOFTMAX_LAYER;
    return layer_p;
}


void layer_forward(Layer* layer_p, SoftmaxLayer* sm_p)
{
    double sum = 0;
    for(int i = 0; i < layer_p->inputs.size; i++)
    {
        double item = exp(layer_p->inputs.contents_p[i]);
        item = item < -500 ? -500 : item;
        item = item > 500 ? 500 : item;
        sum += item;
        layer_p->outputs.contents_p[i] = item;
    }
    for(int i = 0; i < layer_p->outputs.size; i++)
    {
        layer_p->outputs.contents_p[i] /= sum;
    }
}


void layer_backward(Layer* layer_p, SoftmaxLayer* sm_p)
{
    memcpy((void*) layer_p->inputs.error_p,
           (void*) layer_p->outputs.error_p,
           layer_p->inputs.size * sizeof(double));
}