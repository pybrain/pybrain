// Part of Arac Neural Network Composition Library.
// (c) 2008 by Justin S Bayer, <bayer.justin@googlemail.com>

// 
// Header file that accumulates all the different layers.
//


#ifndef Arac_LAYERS_INCLUDED
#define Arac_LAYERS_INCLUDED


#include <cassert>
#include <cstring>

#include "bias.h"
#include "linear.h"
#include "sigmoid.h"
#include "mdlstm.h"
#include "lstm.h"
#include "softmax.h"
#include "tanh.h"


// 
// Function that delegates to the specific implementations of layer_forward
// and layer_backward by different types.
//

void forward(Layer* layer_p);

void backward(Layer* layer_p);


#endif  // Arac_LAYERS_INCLUDED

