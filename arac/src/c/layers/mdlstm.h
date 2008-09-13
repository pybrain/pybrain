//
// MDLSTM Layer, as described in "Multi-Dimensional Recurrent Neural Networks"  
// by Alex Graves, Santiago Fernández, Jürgen Schmidhuber
//

//
// Author: Justin Bayer, bayer.justin@googlemail.com;
//         Tom Schaul, schaul@googlemail.com
//      

// 
// Layout of inputs and outputs
// ----------------------------
// Depending on how many "real" inputs the layer is given, there will be 
// 
//     I = (3 + 2 * d) * s
//     
// inputs over all, where s is the size (= the "true" input) and d is the 
// dimensionality of the MDRNN. The input layout is as follows:
// 
//     Name            Size in doubles
//     -------------------------------
//     input gate      s
//     forget gate     s * d
//     input           s
//     output gate     s
//     states          s * d
//
// The output layout corresponds to
//
//      Name            Size in doubles
//      -------------------------------
//      output          s
//      states          s
//
//


#ifndef Arac_MDLSTM_LINEAR_INCLUDED
#define Arac_MDLSTM_LINEAR_INCLUDED


#include "common.h"


struct MdLstmLayer {
    int timedim;
    
    ParameterContainer peephole_input_weights;
    ParameterContainer peephole_forget_weights;
    ParameterContainer peephole_output_weights;
    
    double* input_squashed_p;
    double* input_gate_squashed_p;
    double* input_gate_unsquashed_p;
    double* output_gate_squashed_p;
    double* output_gate_unsquashed_p;
    double* forget_gate_unsquashed_p;
    double* forget_gate_squashed_p;
    
    double (*gate_squasher) (double);
    double (*gate_squasher_prime) (double);
    double (*cell_squasher) (double);
    double (*cell_squasher_prime) (double);
    double (*output_squasher) (double);
    double (*output_squasher_prime) (double);
};


//
// Create an MdLstmLayer. 
//

Layer* make_mdlstm_layer(int dim, int timedim);

Layer* make_mdlstm_layer(int dim, int timedim, bool use_peepholes);

void make_mdlstm_layer(Layer* layer_p, int dim, int timedim);

void
make_mdlstm_layer(Layer* layer_p, int dim, int timedim, bool use_peepholes);



//
// Process the input buffer of a module to the output buffer of module without 
// changes.
//

void layer_forward(Layer* layer_p, MdLstmLayer* ll_p);

void layer_backward(Layer* layer_p, MdLstmLayer* ll_p);



#endif

