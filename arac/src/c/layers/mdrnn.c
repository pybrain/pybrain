#include <cstring>
#include <iostream>

#include "mdrnn.h"
#include "mdlstm.h"
#include "sigmoid.h"

Layer*
make_mdrnn_layer(int timedim, int* shape_p, int* blockshape_p,
                 int hiddendim, int outdim,
                 double** in_to_hidden_weights_p,
                 double** hidden_to_out_weights_p)
{
    // Calculate the total size of inputs
    int input_size = 1;
    for (int i = 0; i < timedim; i++)
    {
        input_size *= shape_p[i];
    }
    // Calculate the total size of outputs.
    int output_size = outdim;
    for (int i = 0; i < timedim; i++)
    {
        output_size *= blockshape_p[i];
    }

    Layer* layer_p = make_layer(input_size, output_size);
    make_mdrnn_layer(layer_p,
                     timedim, shape_p, blockshape_p,
                     hiddendim, outdim,
                     in_to_hidden_weights_p,
                     hidden_to_out_weights_p);
    return layer_p;
}


//
// Initialize some parameters of the layer which are saved.
//
void
initialize_params(Layer* layer_p,
                  int timedim, int* shape_p, int* blockshape_p,
                  int hiddendim, int outdim)
{
    MdrnnLayer* mdrnn_p = layer_p->internal.mdrnn_layer_p;
    mdrnn_p->hiddendim = hiddendim;
    // Calculate the total input size and check if the dimensionality of the
    // input layer is correct.
    layer_p->inputs.size = 1;
    for(int i = 0; i < timedim; i++)
    {
        layer_p->inputs.size *= shape_p[i];
    }

    // Calculate the size of a single block, e.G. the size of one input.
    mdrnn_p->blockshape_p = blockshape_p;
    mdrnn_p->indim = 1;
    for (int i = 0; i < timedim; i++)
    {
        mdrnn_p->indim *= blockshape_p[i];
    }

    // Calculate the length of a single sequence.
    mdrnn_p->sequence_length = layer_p->inputs.size / mdrnn_p->indim;

    // Calculate the total size of the output.
    mdrnn_p->outdim = outdim;
    layer_p->outputs.size = mdrnn_p->sequence_length * outdim;
}


//
// Allocate memory for different parts of the layer.
//

void
initialize_buffers(Layer* layer_p)
{
    MdrnnLayer* mdrnn_p = layer_p->internal.mdrnn_layer_p;
    // Allocate memory for the cell states.
    mdrnn_p->cell_states_p = \
        (double*) malloc(sizeof(double) * mdrnn_p->sequence_length * mdrnn_p->hiddendim);

    // Allocate memory for the output buffer.
    layer_p->outputs.contents_p = \
        (double*) malloc(sizeof(double) * layer_p->outputs.size);
}


//
//  Initialize and create some internal layers.
//

void initialize_layers(Layer* layer_p, int timedim, int hiddendim, int outdim)
{
    MdrnnLayer* mdrnn_p = layer_p->internal.mdrnn_layer_p;
    // Create the hidden layer. (Never use peepholes.)
    mdrnn_p->hiddenlayer_p = make_mdlstm_layer(hiddendim, timedim, false);

    // Create layers that act as the predecessors.
    mdrnn_p->swipe_predlayers_p = (Layer*) malloc(sizeof(Layer) * timedim);
    for(int i = 0; i < timedim; i++)
    {
        mdrnn_p->swipe_predlayers_p[i].outputs.size = 2 * hiddendim;
        mdrnn_p->swipe_predlayers_p[i].outputs.contents_p = \
            (double*) malloc(sizeof(double) * 2 * hiddendim);
    }

    // Create a layer that acts as the input layer.
    mdrnn_p->swipe_inlayer_p = (Layer*) malloc(sizeof(Layer));
    mdrnn_p->swipe_inlayer_p->outputs.size = mdrnn_p->indim;
    mdrnn_p->swipe_inlayer_p->outputs.contents_p = \
        (double*) malloc(sizeof(double) * mdrnn_p->indim);

    // Create a layer that acts as the output layer.
    mdrnn_p->outlayer_p = (Layer*) malloc(sizeof(Layer));
    mdrnn_p->outlayer_p->inputs.size = outdim;
    mdrnn_p->outlayer_p->inputs.contents_p = layer_p->outputs.contents_p;
}


//
// Initialize the connections between the internal layers.
//
void initialize_connections(
                 Layer* layer_p,
                 int timedim, int hiddendim, int outdim,
                 double** in_to_hidden_weights_p,
                 double** hidden_to_out_weights_p)
{
    MdrnnLayer* mdrnn_p = layer_p->internal.mdrnn_layer_p;
    Layer* hiddenlayer_p = mdrnn_p->hiddenlayer_p;

    // Connect the input layer with the hidden layer.
    full_connect(mdrnn_p->swipe_inlayer_p, hiddenlayer_p,
                 in_to_hidden_weights_p[0],
                 0, mdrnn_p->indim,
                 0, (3 + timedim) * hiddendim);

    // Connect the layers under each other. There is two connections for
    // every time dimension - once in every direction.
    // The n'th and (n + timedim)'th connection will be between the same layers,
    // but in different directions.
    for(int i = 0; i < 2 * timedim; i++)
    {
        // Connect the predecessing layer with the hiddenlayer.
        full_connect(mdrnn_p->swipe_predlayers_p + (i % timedim),
                     mdrnn_p->hiddenlayer_p,
                     in_to_hidden_weights_p[i + 1]);

        // Connect the hiddenlayer with the correct part of the outputlayer.
        full_connect(mdrnn_p->hiddenlayer_p,
                     mdrnn_p->outlayer_p,
                     hidden_to_out_weights_p[i],
                     0, mdrnn_p->hiddenlayer_p->outputs.size / 2 - 1,
                     0, outdim - 1);
    }
}


void
make_mdrnn_layer(Layer* layer_p,
                 int timedim, int* shape_p, int* blockshape_p,
                 int hiddendim, int outdim,
                 double** in_to_hidden_weights_p,
                 double** hidden_to_out_weights_p)
{
    // Allocate memory for the layer.
    MdrnnLayer* mdrnn_p = (MdrnnLayer*) malloc(sizeof(MdrnnLayer));
    layer_p->internal.mdrnn_layer_p = mdrnn_p;

    initialize_params(layer_p,
                      timedim, shape_p, blockshape_p,
                      hiddendim, outdim);
    initialize_buffers(layer_p);
    initialize_layers(layer_p, timedim, hiddendim, outdim);
    initialize_connections(layer_p, timedim, hiddendim, outdim,
                           in_to_hidden_weights_p, hidden_to_out_weights_p);
}


void clear_content(ParameterContainer* container_p)
{
    memset(container_p->contents_p, 0, container_p->size * sizeof(double));
}

void clear_buffers(Layer* layer_p)
{
    MdrnnLayer* mdrnn_p = layer_p->internal.mdrnn_layer_p;
    clear_content(&mdrnn_p->hiddenlayer_p->inputs);
    clear_content(&mdrnn_p->hiddenlayer_p->outputs);
    // TODO: possible not necessary
    for(int i = 0; i < mdrnn_p->timedim; i++)
    {
        clear_content(&mdrnn_p->swipe_predlayers_p[i].outputs);
    }

    // clear_content(&mdrnn_p->outlayer_p->inputs);
}


void adjust_bogus_layers(Layer* layer_p, int block)
{
    // Shortcut
    MdrnnLayer* mdrnn_p = layer_p->internal.mdrnn_layer_p;

    // Length of a single output date in bytes.
    int outsize_bytes = sizeof(double) * mdrnn_p->outdim;

    // Length of a single hiddenstate (of the whole layer) in bytes.
    int hiddenstatesize_bytes = sizeof(double) * mdrnn_p->hiddendim;

    // Index at which the states of the input of an mdlstm start.
    int state_index = (3 + mdrnn_p->timedim) * mdrnn_p->hiddendim;

    // Make the input connection refer to the right parts of the input.
    Connection* in_con_p = mdrnn_p->swipe_inlayer_p->outgoing_p;
    in_con_p->inlayerstart = mdrnn_p->indim * block;
    in_con_p->inlayerstop = mdrnn_p->indim * (block + 1);

    // Integer to calculate how many blocks one has to go back to find a
    // predecessing block. This will be multiplied while looping over the
    // dimensions.
    int decrement_blocks = 1;

    for(int i = 0; i < mdrnn_p->timedim; i++)
    {
        // If we would look into a region that does not exist, break the loop.
        if ((block - decrement_blocks) < 0) {
            break;
        }

        // Amount of blocks from the beginning till the output of the
        // predecessing block.
        int pred_block_index = block - decrement_blocks;

        // Amount of bytes from the beginning till the output of the predecessing
        // block.
        int pred_block_offset = mdrnn_p->hiddendim * pred_block_index;
        // Pointer to that data.
        double* pred_output = layer_p->outputs.contents_p + pred_block_offset;
        // First copy the outputs of predecessing layers.
        memcpy((void*) mdrnn_p->swipe_predlayers_p[i].outputs.contents_p,
               (void*) pred_output,
               outsize_bytes);

        // Then copy in the state of the previous cells.
        pred_block_offset = pred_block_index * mdrnn_p->hiddendim;
        pred_block_offset += mdrnn_p->hiddendim * i;
        double* pred_block_state_p = mdrnn_p->cell_states_p + pred_block_offset;

        int this_state_index = mdrnn_p->hiddendim * i + state_index;
        double* this_state_p = \
            mdrnn_p->hiddenlayer_p->inputs.contents_p + this_state_index;

        memcpy((void*) this_state_p,
               (void*) pred_block_state_p,
               hiddenstatesize_bytes);

        // Move on to the next predecessor by multiplying the current
        // decrementer with the next dimension.
        decrement_blocks *= mdrnn_p->shape_p[i];
    }
    // Make the hidden to output connection refer to the right parts of the
    // output.
    Connection* out_con_p = mdrnn_p->hiddenlayer_p->outgoing_p;
    out_con_p->outlayerstart = mdrnn_p->outdim * block;
    out_con_p->outlayerstop = mdrnn_p->outdim * (block + 1);

    // Make the bias to output connection refer to the right parts of the
    // output.
    Connection* bias_out_con_p = mdrnn_p->bias_p->outgoing_p + 1;
    bias_out_con_p->outlayerstart = mdrnn_p->outdim * block;
    bias_out_con_p->outlayerstop = mdrnn_p->outdim * (block + 1);
}


// TODO: find a more reasonable name
void forward_connections(Layer* layer_p, int swipe)
{
    MdrnnLayer* mdrnn_p = layer_p->internal.mdrnn_layer_p; // Shortcut

    forward(&mdrnn_p->swipe_inlayer_p->outgoing_p[0]);
    // forward(&mdrnn_p->bias_p->outgoing_p[0]);
    // forward(&mdrnn_p->bias_p->outgoing_p[1]);

    // std::cout << "Swipe in inputs: ";
    // for(size_t i = 0; i < mdrnn_p->swipe_inlayer_p->inputs.size; ++i)
    // {
    //     std::cout << mdrnn_p->swipe_inlayer_p->inputs.contents_p[i] << " ";
    // }
    // std::cout << std::endl;

    // std::cout << "Swipe in outputs: ";
    // for(size_t i = 0; i < mdrnn_p->swipe_inlayer_p->outputs.size; ++i)
    // {
    //     std::cout << mdrnn_p->swipe_inlayer_p->outputs.contents_p[i] << " ";
    // }
    // std::cout << std::endl;

    int timedim = mdrnn_p->timedim;
    for(int i = 0; i < timedim; i++)
    {
        Layer* cur_pred_p = mdrnn_p->swipe_predlayers_p + i;
        int idx = (i >> timedim) % 2;

        forward(&cur_pred_p->outgoing_p[idx]);
    }

    // std::cout << "Hidden inputs: ";
    // for(size_t i = 0; i < mdrnn_p->hiddenlayer_p->inputs.size; ++i)
    // {
    //     std::cout << mdrnn_p->hiddenlayer_p->inputs.contents_p[i] << " ";
    // }
    // std::cout << std::endl;

    layer_forward(mdrnn_p->hiddenlayer_p,
                  mdrnn_p->hiddenlayer_p->internal.mdlstm_layer_p);

    // std::cout << "Hidden outputs: ";
    // for(size_t i = 0; i < mdrnn_p->hiddenlayer_p->outputs.size; ++i)
    // {
    //     std::cout << mdrnn_p->hiddenlayer_p->outputs.contents_p[i] << " ";
    // }
    // std::cout << std::endl;
    // std::cout << std::endl;

    forward(mdrnn_p->hiddenlayer_p->outgoing_p);

    // std::cout << "Outlayer inputs: ";
    // for(size_t i = 0; i < mdrnn_p->outlayer_p->inputs.size; ++i)
    // {
    //     std::cout << mdrnn_p->outlayer_p->inputs.contents_p[i] << " ";
    // }
    // std::cout << std::endl;
    // std::cout << std::endl;

    // std::cout << "Outlayer outputs: ";
    // for(size_t i = 0; i < mdrnn_p->outlayer_p->outputs.size; ++i)
    // {
    //     std::cout << mdrnn_p->outlayer_p->outputs.contents_p[i] << " ";
    // }
    // std::cout << std::endl;
    // std::cout << std::endl;
}

void backcopy_states(Layer* layer_p, int block)
{
    MdrnnLayer* mdrnn_p = layer_p->internal.mdrnn_layer_p;
    // Index at which the states of the input of an mdlstm start.
    int state_index = mdrnn_p->hiddendim;
    int my_state = block * mdrnn_p->hiddendim;
    memcpy((void*) (mdrnn_p->cell_states_p + my_state),
           (void*) (mdrnn_p->hiddenlayer_p->outputs.contents_p + state_index),
           mdrnn_p->hiddendim * sizeof(double));
}


void layer_forward(Layer* layer_p, MdrnnLayer* mdrnn_p)
{
    int swipes = 2 << (mdrnn_p->timedim - 1);
    for(int i = 0; i < swipes; i++)
    {
        clear_buffers(layer_p);
        for(int j = 0; j < mdrnn_p->sequence_length; j++)
        {
            // TODO: maybe split these into differently named and more functions
            adjust_bogus_layers(layer_p, j);
            forward_connections(layer_p, i);
            backcopy_states(layer_p, j);
        }
    }
}


void layer_backward(Layer* layer_p, MdrnnLayer* mdrnn_p)
{
    ;
}


