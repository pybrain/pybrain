#include <assert.h>
#include <iostream>
#include <string.h>

#include "arac.h"
#include "layers/layers.h"


extern "C"
{


void print_layer(Layer* layer_p)
{
    std::cout << "Layer at #" << layer_p << std::endl;
    std::cout << "  Input-Dim: " << layer_p->inputs.size << std::endl
              << "  Output-Dim: " << layer_p->outputs.size << std::endl
              << "  #Incoming: " << layer_p->incoming_n << std::endl
              << "  #Outgoing: " << layer_p->outgoing_n << std::endl;

    std::cout << "  Inputs at #" << (int) layer_p->inputs.contents_p << ": ";
    for (int i = 0; i < layer_p->inputs.size; i++)
    {
        std::cout << "  " << layer_p->inputs.contents_p[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "  Outputs at #" << (int) layer_p->outputs.contents_p << ": ";
    for (int i = 0; i < layer_p->outputs.size; i++)
    {
        std::cout << "  " << layer_p->outputs.contents_p[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "  Inputerrors at #" << (int) layer_p->inputs.error_p << ": ";
    for (int i = 0; i < layer_p->inputs.size; i++)
    {
        std::cout << "  " << (double) layer_p->inputs.error_p[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "  Outputerrors at #" << (int) layer_p->outputs.error_p << ": ";
    for (int i = 0; i < layer_p->outputs.size; i++)
    {
        std::cout << "  " << (double) layer_p->outputs.error_p[i] << " ";
    }
    
    std::cout << std::endl;
    std::cout << std::endl;
    std::cout.flush();
}


void print_connection(Connection* con_p)
{
    
    std::cout << "Connection at #" << (int) con_p << std::endl
              << "  Input-Layer at #" << (int) con_p->inlayer_p << std::endl
              << "  Output-Layer at #" << (int) con_p->outlayer_p << std::endl
              << "  Input-Layer-Dim: " << con_p->inlayer_p->inputs.size << std::endl
              << "  Output-Layer-Dim: " << con_p->outlayer_p->inputs.size << std::endl
              << "  Slices: " << con_p->inlayerstart << " "
                              << con_p->inlayerstop << " "
                              << con_p->outlayerstart << " "
                              << con_p->outlayerstop << " "
              << std::endl;
              // << "  #Incoming: " << layer->incoming_n << std::endl
              // << "  #Outgoing: " << layer->outgoing_n 

    if (con_p->type == FULL_CONNECTION)
    {
        std::cout << "Weights: ";
        for(int i = 0; i < con_p->internal.full_connection_p->weights.size; i++)
        {
            std::cout << con_p->internal.full_connection_p->weights.contents_p[i] << " ";
        }
        std::cout << std::endl;

        std::cout << "Derivs: ";
        for(int i = 0; i < con_p->internal.full_connection_p->weights.size; i++)
        {
            std::cout << con_p->internal.full_connection_p->weights.error_p[i] << " ";
        }
        std::cout << std::endl;
    }

    std::cout << std::endl;
    std::cout.flush();
}



void activate(Layer* layer_p, int n_layers)
{
    // for (int i = 0; i < n_layers; i++)
    // {
    //     Layer* current_layer_p = &layer_p[i];
    //     print_layer(current_layer_p);
    //     for (int i = 0; i < current_layer_p->incoming_n; i++)
    //     {
    //         Connection& cur_con = current_layer_p->incoming_p[i];
    //         print_connection(&cur_con);
    //     }
    // }

    for (int i = 0; i < n_layers; i++)
    {
        Layer* current_layer_p = &layer_p[i];
        forward(current_layer_p);
        for (int j = 0; j < current_layer_p->outgoing_n; j++)
        {
            Connection& cur_con = current_layer_p->outgoing_p[j];
            forward(&cur_con);
        }
    }

    (*(layer_p->timestep_p))++;
}


void resetAll(Layer* layer_p, int n_layers)
{
    for (int i = 0; i < n_layers; i++)
    {
        Layer* current_layer_p = &layer_p[i];
        reset_layer(current_layer_p, current_layer_p->inputs.size, current_layer_p->outputs.size);
    }
}


void calc_derivs(Layer* layer_p, int n_layers) {
    for (int i = n_layers - 1; i >= 0; i--)
    {
        Layer* current_layer_p = &layer_p[i];
        backward(current_layer_p);
        for (int i = 0; i < current_layer_p->incoming_n; i++)
        {
            Connection& cur_con = current_layer_p->incoming_p[i];
            backward(&cur_con);
        }
    }
    (*(layer_p->timestep_p))--;

    // for (int i = 0; i < n_layers; i++)
    // {
    //     Layer* current_layer_p = &layer_p[i];
    //     print_layer(current_layer_p);
    //     for (int i = 0; i < current_layer_p->incoming_n; i++)
    //     {
    //         Connection& cur_con = current_layer_p->incoming_p[i];
    //         print_connection(&cur_con);
    //     }
    // }
}


void setTimestepPointer(Layer* layer_p, int n_layers, int* target)
{
    for (int i = 0; i < n_layers; i++)
    {
        layer_p[i].timestep_p = target;
    }
}

} // Extern
