#ifndef Arac_LSTM_LINEAR_INCLUDED
#define Arac_LSTM_LINEAR_INCLUDED


#include "common.h"
#include "mdlstm.h"


struct LstmLayer {
    Layer* mdlstm_p;
    ParameterContainer states;
};


Layer* make_lstm_layer(int dim);


void make_lstm_layer(Layer* layer_p, int dim);


void
make_lstm_layer(Layer* layer_p, int dim, bool use_peepholes);


void layer_forward(Layer* layer_p, LstmLayer* ll_p);


void layer_backward(Layer* layer_p, LstmLayer* ll_p);


#endif

