// Part of Arac Neural Network Composition Library.
// (c) 2008 by Justin S Bayer, <bayer.justin@googlemail.com>

//
// Main header file of the library.
//


#ifndef Arac_INCLUDED
#define Arac_INCLUDED


#include "connections/connections.h"
#include "layers/layers.h"
#include "functions.h"


extern "C" {
    
//
// Prototype functions
// 
    
// Print out information about a module.
void print_layer(Layer* layer);


// Print out information about a connection.
void print_connection(Connection* con_p);


// Make a forward pass through the whole module graph.
void activate(Layer* layer_p, int n_layers);


// Calculate the errors back from the leaf to the root.
void calc_derivs(Layer* layer_p, int n_layers);


// Set the timestep pointers of the layers to the same memory address.
void setTimestepPointer(Layer* layer_p, int n_layers, int* target);

} // Extern


#endif // Arac_INCLUDED


