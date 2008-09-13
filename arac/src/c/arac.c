#include <assert.h>
#include <iostream>
#include <string.h>

#include "arac.h"


extern "C"
{


void print_layer(Layer* layer)
{
    
    std::cout << "Layer at #" << layer << std::endl
              << "  Input-Dim: " << layer->inputs.size << std::endl
              << "  Output-Dim: " << layer->outputs.size << std::endl
              << "  #Incoming: " << layer->incoming_n << std::endl
              << "  #Outgoing: " << layer->outgoing_n << std::endl;

    std::cout << "  Inputs at #" << layer->inputs.contents_p << ": ";
    for (int i = 0; i < layer->inputs.size; i++) 
    {
        std::cout << "  " << layer->inputs.contents_p[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "  Outputs at #" << layer->inputs.contents_p << ": ";
    for (int i = 0; i < layer->outputs.size; i++) 
    {
        std::cout << "  " << layer->outputs.contents_p[i] << " ";
    }
    
    std::cout << std::endl;
    std::cout << std::endl;
    std::cout.flush();
}


void print_connection(Connection* con_p)
{
    
    std::cout << "Connection at #" << con_p << std::endl
              << "  Input-Layer at #" << con_p->inlayer_p << std::endl
              << "  Output-Layer at #" << con_p->outlayer_p << std::endl
              << "  Output-Layer-Dim: " << con_p->outlayer_p->inputs.size << std::endl
              << "  Output-Layer-Dim: " << con_p->outlayer_p->inputs.size << std::endl
              // << "  #Incoming: " << layer->incoming_n << std::endl
              // << "  #Outgoing: " << layer->outgoing_n 
              << std::endl;

    std::cout.flush();
}



void activate(Layer* layer_p, int n_layers)
{
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

}


void setTimestepPointer(Layer* layer_p, int n_layers, int* target)
{
    for (int i = 0; i < n_layers; i++)
    {
        layer_p[i].timestep_p = target;
    }
}


} // Extern
