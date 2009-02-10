// Part of Arac Neural Network Composition Library.
// (c) 2008 by Justin S Bayer, <bayer.justin@googlemail.com>

#include <cstring>
#include <cmath>

#include "softmax.h"


using arac::structure::modules::SoftmaxLayer;


void
SoftmaxLayer::_forward()
{
    double sum = 0;
    double* input_p = input()[timestep()];
    double* output_p = output()[timestep()];
    for(int i = 0; i < _insize; i++)
    {
        double item = exp(input_p[i]);
        item = item < -500 ? -500 : item;
        item = item > 500 ? 500 : item;
        sum += item;
        output_p[i] = item;
    }
    for(int i = 0; i < _outsize; i++)
    {
        output_p[i] /= sum;
    }
}


void
SoftmaxLayer::_backward()
{
    int size = _outsize * sizeof(double);
    void* sourcebuffer_p = (void*) outerror()[timestep() - 1];
    void* sinkbuffer_p = (void*) inerror()[timestep() - 1];
    memcpy(sinkbuffer_p, sourcebuffer_p, size);
}