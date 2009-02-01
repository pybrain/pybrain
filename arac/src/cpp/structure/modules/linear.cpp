// Part of Arac Neural Network Composition Library.
// (c) 2008 by Justin S Bayer, <bayer.justin@googlemail.com>


#include <cstring>

#include "linear.h"


using arac::structure::modules::LinearLayer;


void
LinearLayer::_forward()
{
    int size = _insize * sizeof(double);
    void* sourcebuffer_p = (void*) input()[timestep()];
    void* sinkbuffer_p = (void*) output()[timestep()];
    memcpy(sinkbuffer_p, sourcebuffer_p, size);
}


void
LinearLayer::_backward()
{
    int size = _outsize * sizeof(double);
    void* sourcebuffer_p = (void*) outerror()[timestep() - 1];
    void* sinkbuffer_p = (void*) inerror()[timestep() - 1];
    memcpy(sinkbuffer_p, sourcebuffer_p, size);
}