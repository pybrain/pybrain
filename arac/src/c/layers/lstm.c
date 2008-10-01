#include "lstm.h"
#include <cstring>
#include <iostream>
#include <cstdlib>

Layer* make_lstm_layer(int dim)
{
    Layer* layer_p = make_layer(4 * dim, 2 * dim);
    make_lstm_layer(layer_p, dim, 0);
    return layer_p;
}


void make_lstm_layer(Layer* layer_p, int dim)
{
    make_lstm_layer(layer_p, dim, 0);
}


void
make_lstm_layer(Layer* layer_p, int dim, bool use_peepholes)
{
    layer_p->type = LSTM_LAYER;
    layer_p->internal.lstm_layer_p = (LstmLayer*) malloc(sizeof(LstmLayer));
    layer_p->internal.lstm_layer_p->mdlstm_p = \
        make_mdlstm_layer(dim, 1, use_peepholes);
    layer_p->internal.lstm_layer_p->states.contents_p = \
        (double*) malloc(sizeof(double) * dim);
    layer_p->internal.lstm_layer_p->states.error_p = \
        (double*) malloc(sizeof(double) * dim);
    
}


void layer_forward(Layer* layer_p, LstmLayer* ll_p)
{
    // The forward method of the standard lstm basically uses the wrapped mdlstm
    // and copies the inputbuffer and states from the preceeding timesteps to 
    // the right places. Afterwards, it copies the results back to the right 
    // places.
    Layer* mdlstm_p = ll_p->mdlstm_p;

    // Bufferoffset to be used at various places to make up for time/offsets
    int offset;     

    // The actual size of inputs of this layer, without architectural additions
    int size = layer_p->inputs.size / 4;

    // Copy the input buffer to the right place
    offset = (*layer_p->timestep_p) * 4 * size;

    memcpy((void*) mdlstm_p->inputs.contents_p,
           (void*) (layer_p->inputs.contents_p + offset), 
           sizeof(double) * size * 4);

    // Check if we are not in the first timestep, because otherwise there will
    // not be a previous state.
    if ((*layer_p->timestep_p) > 0)
    {
        // Copy the last state to the right place
        offset = ((*layer_p->timestep_p) - 1) * size;
        memcpy((void*) (mdlstm_p->inputs.contents_p + 4 * size),
               (void*) (ll_p->states.contents_p + offset), 
               sizeof(double) * size);
    }

    layer_forward(mdlstm_p, mdlstm_p->internal.mdlstm_layer_p);

    offset = (*layer_p->timestep_p) * size;

    // Copy back the results
    memcpy((void*) (layer_p->outputs.contents_p + offset), 
           (void*) mdlstm_p->outputs.contents_p, 
           sizeof(double) * size);

    // Copy back the state
    memcpy((void*) (ll_p->states.contents_p + offset), 
           (void*) (mdlstm_p->outputs.contents_p + size),
           sizeof(double) * size);
}


void layer_backward(Layer* layer_p, LstmLayer* ll_p)
{
    // The forward method of the standard lstm basically uses the wrapped mdlstm
    // and copies the inputbuffer and states from the preceeding timesteps to 
    // the right places. Afterwards, it copies the results back to the right 
    // places.
    Layer* mdlstm_p = ll_p->mdlstm_p;
    // Bufferoffset to be used at various places to make up for time/offsets.
    int offset;     
    // The actual size of inputs of this layer, without architectural additions.
    int size = layer_p->inputs.size / 4;

    // Copy the output error to the right place.
    offset = ((*layer_p->timestep_p) - 1) * size;
    memcpy((void*) (mdlstm_p->outputs.error_p),
           (void*) (layer_p->outputs.error_p + offset),
           sizeof(double) * size);

    if (*layer_p->timestep_p < *layer_p->seqlen_p) {
        offset = *layer_p->timestep_p * size;
        memcpy((void*) (mdlstm_p->outputs.error_p + size),
               (void*) (ll_p->states.error_p + offset),
               sizeof(double) * size);
    }

    layer_backward(mdlstm_p, mdlstm_p->internal.mdlstm_layer_p);

    // Copy back the error to the right places after calculation.
    offset = ((*layer_p->timestep_p) - 1) * 4 * size;
    memcpy((void*) (layer_p->inputs.error_p  + offset),
           (void*) (mdlstm_p->inputs.error_p),
           sizeof(double) * 4 * size);

    offset = ((*layer_p->timestep_p) - 1) * size;
    memcpy((void*) (ll_p->states.error_p  + offset),
           (void*) (mdlstm_p->inputs.error_p + 4 * size),
           sizeof(double) * size);
}



