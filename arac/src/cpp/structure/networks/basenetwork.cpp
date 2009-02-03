// Part of Arac Neural Network Composition Library.
// (c) 2008 by Justin S Bayer, <bayer.justin@googlemail.com>


#include <iostream>
#include <cstring>

#include "basenetwork.h"


namespace arac {
namespace structure {
namespace networks {


using arac::structure::networks::BaseNetwork;


BaseNetwork::BaseNetwork() : 
    Module(),
    _dirty(true)
{
}


BaseNetwork::~BaseNetwork()
{
}


void
BaseNetwork::forward()
{
    if (_dirty)
    {
        sort();
    }
    Module::forward();
}


const double*
BaseNetwork::activate(double* input_p)
{
    if (_dirty)
    {
        sort();
    }
    else if (!sequential())
    {
        clear();
    }
    // Copy this input into the inputbuffer.
    memcpy((void*) input()[timestep()], 
           (void*) input_p, 
           sizeof(double) * _insize);
    forward();
    assert(timestep() > 0);
    return output()[timestep() - 1];
}


void
BaseNetwork::activate(double* input_p, double* output_p)
{
    const double* result_p = activate(input_p);
    memcpy(output_p, result_p, sizeof(double) * outsize());
}


const double*
BaseNetwork::back_activate(double* error_p)
{
    memcpy((void*) outerror()[timestep() - 1], 
           (void*) error_p, 
           sizeof(double) * _outsize);

    backward();
    return inerror()[timestep()];
}


void
BaseNetwork::back_activate(double* outerror_p, double* inerror_p)
{
    const double* result_p = back_activate(outerror_p);
    memcpy(inerror_p, result_p, sizeof(double) * insize());
}


}
}
}
