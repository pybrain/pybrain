// Part of Arac Neural Network Composition Library.
// (c) 2008 by Justin S Bayer, <bayer.justin@googlemail.com>

#include <cstring>

#include "tanh.h"
#include "../../common/functions.h"



using arac::structure::modules::TanhLayer;
using arac::common::tanh_;
using arac::common::tanhprime;


void
TanhLayer::_forward()
{
    double* input_p = input()[timestep()];
    double* output_p = output()[timestep()];
    for (int i = 0; i < _insize; i++)
    {
        *output_p = tanh_(*input_p);
        output_p++;
        input_p++;
    }
}


void
TanhLayer::_backward()
{
    double* outerror_p = outerror()[timestep() - 1];
    double* output_p =  output()[timestep() - 1];
    double* inerror_p = inerror()[timestep() - 1];
    for (int i = 0; i < _insize; i++)
    {
        inerror_p[i] += (1 - output_p[i] * output_p[i]) * outerror_p[i];
    }

}