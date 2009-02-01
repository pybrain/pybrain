// Part of Arac Neural Network Composition Library.
// (c) 2008 by Justin S Bayer, <bayer.justin@googlemail.com>


#ifndef Arac_COMMON_FUNCTIONS_INCLUDED
#define Arac_COMMON_FUNCTIONS_INCLUDED

#include <cmath>

namespace arac {
namespace common {

inline
double
sigmoid(double x)
{
    return 1.0 / (1 + exp(-x));
}

inline
double
sigmoidprime(double x)
{
    double evald = sigmoid(x);
    return evald * (1.0 - evald);
}

inline
double
tanh_(double x)
{
    return tanh(x);
}

inline
double
tanhprime(double x)
{
    double evald = tanh(x);
    return 1. - (evald * evald);
}


}
}


#endif