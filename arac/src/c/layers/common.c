#include <iostream>
#include <cassert>
#include "common.h"


void make_layer(Layer* layer_p, int input_dim, int output_dim)
{
    layer_p->inputs.size = input_dim;
    layer_p->outputs.size = output_dim;
    layer_p->outgoing_n = 0;
    layer_p->incoming_n = 0;
    layer_p->inputs.contents_p = (double*) malloc(sizeof(double) * input_dim);
    layer_p->outputs.contents_p = (double*) malloc(sizeof(double) * output_dim);
    layer_p->inputs.error_p = (double*) malloc(sizeof(double) * input_dim);
    layer_p->outputs.error_p = (double*) malloc(sizeof(double) * output_dim);
    layer_p->timestep_p = (int*) malloc(sizeof(int));
    layer_p->seqlen_p = (int*) malloc(sizeof(int));

    // Assure that all parameters are zeros
    for(int i = 0; i < input_dim; i++) 
    {
        layer_p->inputs.contents_p[i] = 0.0;
        layer_p->inputs.error_p[i] = 0.0;
    }
    
    for(int i = 0; i < output_dim; i++) 
    {
        layer_p->outputs.contents_p[i] = 0.0;
        layer_p->outputs.error_p[i] = 0.0;
    }

    layer_p->timestep_p[0] = 0;
    layer_p->seqlen_p[0] = 0;
}


Layer* 
make_layer(int input_dim, int output_dim)
{
    Layer* layer_p = (Layer*) malloc(sizeof(Layer));
    make_layer(layer_p, input_dim, output_dim);
    return layer_p;
}


void layer_map_forward(Layer* layer_p, double (*mapper) (double))
{
    assert(layer_p->inputs.size == layer_p->outputs.size);
    assert(layer_p->inputs.contents_p != 0);
    assert(layer_p->outputs.contents_p != 0);

    assert(layer_p->timestep_p != 0);

    // Bufferincrementer depending on the current timestep.
    int bi = (*layer_p->timestep_p) * layer_p->inputs.size;
    // Shortcuts to the doublearrays
    double* from_p = layer_p->inputs.contents_p + bi;
    double* to_p = layer_p->outputs.contents_p + bi;
    
    for (int i = 0; i < layer_p->inputs.size; i++)
    {
        to_p[i] = mapper(from_p[i]);
    }
}


void layer_map_backward(Layer* layer_p, double (*mapper) (double))
{
    assert(layer_p->inputs.size == layer_p->outputs.size);
    assert(layer_p->inputs.contents_p != 0);
    assert(layer_p->outputs.contents_p != 0);

    // Bufferincrementer depending on the current timestep.
    // We have to subtract 1 since the timestep will already be incremented 
    // after activation. 
    int bi = ((*layer_p->timestep_p) - 1) * layer_p->inputs.size;
    // Shortcuts to the doublearrays
    double* to_error_p = layer_p->inputs.error_p + bi;
    double* from_error_p = layer_p->outputs.error_p + bi;
    double* to_inputs_p = layer_p->inputs.contents_p + bi;

    for (int i = 0; i < layer_p->inputs.size; i++)
    {
        to_error_p[i] = mapper(to_inputs_p[i]) * from_error_p[i];
    }
}
