#include "linear.h"

void 
make_linear_layer(Layer* layer_p, int dim)
{
    make_layer(layer_p, dim, dim);
    layer_p->type = LINEAR_LAYER;
}


Layer* 
make_linear_layer(int dim)
{
    Layer* layer_p = make_layer(dim, dim);
    layer_p->type = LINEAR_LAYER;
    return layer_p;
}


void layer_forward(Layer* layer_p, LinearLayer* id_p) 
{
    layer_map_forward(layer_p, identity);
}


void layer_backward(Layer* layer_p, LinearLayer* id_p) 
{
    layer_map_backward(layer_p, identity_prime);
}


double identity(double x)
{
    return x;
}

inline
double identity_prime(double x)
{
    return 1;
}







