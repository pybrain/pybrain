#include <iostream>

#include "mdlstm.h"
#include "../functions.h"


Layer* make_mdlstm_layer(int dim, int timedim)
{
    Layer* layer_p = (Layer*) malloc(sizeof(Layer));
    make_mdlstm_layer(layer_p, dim, timedim);
    return layer_p;
}


Layer* make_mdlstm_layer(int dim, int timedim, bool use_peepholes)
{
    Layer* layer_p = (Layer*) malloc(sizeof(Layer));
    make_mdlstm_layer(layer_p, dim, timedim, use_peepholes);
    return layer_p;
}


void make_mdlstm_layer(Layer* layer_p, int dim, int timedim)
{
    make_mdlstm_layer(layer_p, dim, timedim, 0);
}

    
void make_mdlstm_layer(Layer* layer_p, int dim, int timedim, bool use_peepholes)
{
    make_layer(layer_p, (3 + 2 * timedim) * dim, dim * 2);
    layer_p->type = MDLSTM_LAYER;
    layer_p->internal.mdlstm_layer_p = \
        (MdLstmLayer*) malloc(sizeof(MdLstmLayer));
        
    MdLstmLayer* lstm_p = layer_p->internal.mdlstm_layer_p;
    lstm_p->timedim = timedim;
        
    int memory_needed = sizeof(double) * dim;
    int memory_needed_with_dim = memory_needed * timedim;
    lstm_p->input_squashed_p = (double*) malloc(memory_needed);
    lstm_p->input_gate_squashed_p = (double*) malloc(memory_needed_with_dim);
    lstm_p->input_gate_unsquashed_p = (double*) malloc(memory_needed_with_dim);
    lstm_p->output_gate_squashed_p = (double*) malloc(memory_needed_with_dim);
    lstm_p->output_gate_unsquashed_p = (double*) malloc(memory_needed_with_dim);
    lstm_p->forget_gate_squashed_p = (double*) malloc(memory_needed_with_dim);
    lstm_p->forget_gate_unsquashed_p = (double*) malloc(memory_needed_with_dim);

    if (use_peepholes)
    {
        lstm_p->peephole_input_weights.size = memory_needed;
        lstm_p->peephole_input_weights.contents_p = \
            (double*) malloc(memory_needed);
        lstm_p->peephole_input_weights.error_p = \
            (double*) malloc(memory_needed);

        lstm_p->peephole_output_weights.size = memory_needed;
        lstm_p->peephole_output_weights.contents_p = \
            (double*) malloc(memory_needed);
        lstm_p->peephole_output_weights.error_p = \
            (double*) malloc(memory_needed);
    
        lstm_p->peephole_forget_weights.size = memory_needed_with_dim;
        lstm_p->peephole_forget_weights.contents_p = \
            (double*) malloc(memory_needed_with_dim);
        lstm_p->peephole_forget_weights.error_p = \
            (double*) malloc(memory_needed_with_dim);
    }
    else {
        lstm_p->peephole_input_weights.contents_p = 0;
        lstm_p->peephole_input_weights.error_p = 0;
        lstm_p->peephole_output_weights.contents_p = 0;
        lstm_p->peephole_output_weights.error_p = 0;
        lstm_p->peephole_forget_weights.contents_p = 0;
        lstm_p->peephole_forget_weights.error_p = 0;
    }
    
    lstm_p->gate_squasher = sigmoid;
    lstm_p->gate_squasher_prime = sigmoid_prime;
    lstm_p->cell_squasher = tanh_;
    lstm_p->cell_squasher_prime = tanh_prime;
    lstm_p->output_squasher = tanh_;
    lstm_p->output_squasher_prime = tanh_prime;
}


void layer_forward(Layer* layer_p, MdLstmLayer* mdlstml_p)
{
    int dim = mdlstml_p->timedim;
    int size = layer_p->outputs.size / 2;
    
    double input_p[size];
    double inputstate_p[size];
    double outputstate_p[size];
    
    double (*gate_squasher) (double) = mdlstml_p->gate_squasher;
    double (*cell_squasher) (double) = mdlstml_p->cell_squasher;
    double (*output_squasher) (double) = mdlstml_p->output_squasher;
    
    // Split the whole input into the right chunks
    double* inputbuffer_p = layer_p->inputs.contents_p;
    int i = 0;
    for (int j = 0; j < size; j++, i++)
    {
        mdlstml_p->input_gate_unsquashed_p[j] = inputbuffer_p[i];
    }

    for (int j = 0; j < size * dim; j++, i++)
    {
        mdlstml_p->forget_gate_unsquashed_p[j] = inputbuffer_p[i];
    }
    
    for (int j = 0; j < size; j++, i++)
    {
        mdlstml_p->input_squashed_p[j] = cell_squasher(inputbuffer_p[i]);
    }

    for (int j = 0; j < size; j++, i++)
    {
        mdlstml_p->output_gate_unsquashed_p[j] = inputbuffer_p[i];
    }
    
    for (int j = 0; j < size * dim ; j++, i++)
    {
        inputstate_p[j] = inputbuffer_p[i];
    }
    
    // Change the ingate values with respect to peepholes, if we have peephole 
    // weights
    if (mdlstml_p->peephole_input_weights.contents_p)
    {
        for (int i = 0; i < dim; i++)
        {
            for (int j = 0; j < size; j++)
            {
                mdlstml_p->input_gate_unsquashed_p[j] += \
                    mdlstml_p->peephole_input_weights.contents_p[j] \
                    * inputstate_p[i * size + j];
            }
        }
    }
    
    if (mdlstml_p->peephole_forget_weights.contents_p)
    {
        for (int i = 0; i < size * dim; i++)
        {
            mdlstml_p->forget_gate_unsquashed_p[i] += \
                mdlstml_p->peephole_forget_weights.contents_p[i] * inputstate_p[i];
        }
    }
    
    // Squash the input gates and forget gates.
    for (int i = 0; i < size; i++)
    {
        mdlstml_p->input_gate_squashed_p[i] = \
            gate_squasher(mdlstml_p->input_gate_unsquashed_p[i]);
        mdlstml_p->forget_gate_squashed_p[i] = \
            gate_squasher(mdlstml_p->forget_gate_unsquashed_p[i]);
    }
    
    // Calculate the current cell state.
    for (int i = 0; i < size; i++)
    {
        outputstate_p[i] = mdlstml_p->input_gate_squashed_p[i] \
            * mdlstml_p->input_squashed_p[i];
    }

    // Apply the forget gates.
    for (int i = 0; i < dim; i++)
    {
        for (int j = 0; j < size; j++)
        {
            int index = size * i + j;
            outputstate_p[j] += \
                mdlstml_p->forget_gate_squashed_p[index] * inputstate_p[index];
            
        }
    }
    
    // Change the outgate values if peephole weights are set.
    if (mdlstml_p->peephole_output_weights.contents_p)
    {
        for (int i = 0; i < size; i++)
        {
            mdlstml_p->output_gate_unsquashed_p[i] += \
                mdlstml_p->peephole_output_weights.contents_p[i] * outputstate_p[i];
        }
    }
    
    // Squash the output gates.
    for (int i = 0; i < size; i++)
    {
        mdlstml_p->output_gate_squashed_p[i] = \
            gate_squasher(mdlstml_p->output_gate_unsquashed_p[i]);
    }
    // Save the results to the outputbuffer.
    double* outputbuffer_p = layer_p->outputs.contents_p;
    for (int i = 0; i < size; i++)
    {
        outputbuffer_p[i] = output_squasher(
            mdlstml_p->output_gate_squashed_p[i] * outputstate_p[i]);
        outputbuffer_p[i + size] = outputstate_p[i];
    }

}

void layer_backward(Layer* layer_p, MdLstmLayer* mdlstml_p)
{
    int dim = mdlstml_p->timedim;
    
    int size = layer_p->outputs.size / 2;
    
    double input_p[size];
    double input_state_p[size];
    
    double output_error_p[size];
    double output_state_error_p[size];

    double output_gate_error_p[size];
    double forget_gate_error_p[size * dim];
    double input_gate_error_p[size];
    double input_error_p[size];
    double input_state_error_p[size * dim];
    double state_error_p[size];

    double (*gate_squasher_prime) (double) = mdlstml_p->gate_squasher_prime;
    double (*cell_squasher_prime) (double) = mdlstml_p->cell_squasher_prime;
    double (*output_squasher_prime) (double) = mdlstml_p->output_squasher_prime;

    // Split the whole input into the right chunks
    double* inputbuffer_p = layer_p->inputs.contents_p;
    double* outputstate_p = layer_p->outputs.contents_p + size;

    int i, j;
    for (i = size + size * dim, j = 0; j < size; j++, i++)
    {
        input_p[j] = inputbuffer_p[i];
    }

    for (i = 3 * size + size * dim, j = 0; j < size * dim ; j++, i++)
    {
        input_state_p[j] = inputbuffer_p[i];
    }

    // Shortcut
    double* output_error_buffer_p = layer_p->outputs.error_p;
    
    // Splitting the errorbuffer into two parts.
    for (int i = 0; i < size; i++)
    {
        output_error_p[i] = output_error_buffer_p[i];
    }

    for (int i = 0; i < size; i++)
    {
        output_state_error_p[i] = output_error_buffer_p[i + size];
    }
    
    // Calculate the outgate error.
    for (int i = 0; i < size; i++)
    {
        output_gate_error_p[i] = \
            gate_squasher_prime(mdlstml_p->output_gate_unsquashed_p[i]) \
            * output_error_p[i] \
            * cell_squasher_prime(outputstate_p[i]);
    }
    
    // This is an intermediate for calculations.
    for (int i = 0; i < size; i++)
    {
        state_error_p[i] = \
            output_error_p[i] \
            * mdlstml_p->output_gate_squashed_p[i] \
            * output_squasher_prime(outputstate_p[i]);
    }
    
    for (int i = 0; i < size; i++)
    {
        state_error_p[i] += output_state_error_p[i];
    }
    
    if (mdlstml_p->peephole_output_weights.contents_p)
    {
        for (int i = 0; i < size; i++)
        {
            state_error_p[i] += \
                output_gate_error_p[i] \
                * mdlstml_p->peephole_output_weights.contents_p[i];
        }
    }
    
    // Calculate cell errors.
    for (int i = 0; i < size; i++)
    {
        input_error_p[i] = \
            mdlstml_p->input_gate_squashed_p[i] \
            * cell_squasher_prime(input_p[i]) \
            * state_error_p[i];
    }
    
    // Apply forget gate.
    for (int i = 0; i < dim; i++)
    {
        for (int j = 0; j < size; j++)
        {
            forget_gate_error_p[i * size + j] = \
                gate_squasher_prime(mdlstml_p->forget_gate_unsquashed_p[i * size + j]) \
                * state_error_p[j] \
                * input_state_p[i * size + j];
        }
    }
    
    for (int i = 0; i < size; i++)
    {
        input_gate_error_p[i] = \
            gate_squasher_prime(mdlstml_p->input_gate_unsquashed_p[i]) \
            * mdlstml_p->input_squashed_p[i] \
            * state_error_p[i];
    }
    
    if (mdlstml_p->peephole_output_weights.contents_p)
    {
        for (int i = 0; i < size; i++)
        {
            mdlstml_p->peephole_output_weights.error_p[i] += \
                output_gate_error_p[i] \
                * outputstate_p[i];
        }
        for (int i = 0; i < size * dim; i++)
        {
            mdlstml_p->peephole_forget_weights.error_p[i] += \
                forget_gate_error_p[i] \
                * input_state_p[i];
        }
        for (int i = 0; i < dim; i++)
        {
            for (int j = 0; j < size; j++)
            {
                mdlstml_p->peephole_input_weights.error_p[j] += \
                    input_gate_error_p[j] \
                    * input_state_p[i * size + j];
            }
        }
    }
    
    for (int i = 0; i < dim; i++)
    {
        for (int j = 0; j < size; j++)
        {
            input_state_error_p[i * size + j] += \
                state_error_p[j] * forget_gate_error_p[i * size + j];
            if (mdlstml_p->peephole_output_weights.contents_p)
            {
                input_state_error_p[i * size + j] += \
                    input_gate_error_p[j] \
                    * mdlstml_p->peephole_input_weights.contents_p[j];
                input_state_error_p[i * size + j] += \
                    forget_gate_error_p[i * size + j] \
                    * mdlstml_p->peephole_forget_weights.contents_p[i * size + j];
            }
        }
    }
    
    i = 0;
    for (int j = 0; j < size; j++, i++)
    {
        layer_p->inputs.error_p[i] = input_gate_error_p[j];
    }

    for (int j = 0; j < size * dim; j++, i++)
    {
        layer_p->inputs.error_p[i] = forget_gate_error_p[j];
    }
    
    for (int j = 0; j < size; j++, i++)
    {
        layer_p->inputs.error_p[i] = input_error_p[j];
    }

    for (int j = 0; j < size; j++, i++)
    {
        layer_p->inputs.error_p[i] = output_gate_error_p[j];
    }
    
    for (int j = 0; j < size * dim ; j++, i++)
    {
        layer_p->inputs.error_p[i] = input_state_error_p[j];
    }
}










