// Part of Arac Neural Network Composition Library.
// (c) 2008 by Justin S Bayer, <bayer.justin@googlemail.com>


#include "bias.h"


using arac::structure::modules::Bias;
using arac::structure::Component;


Bias::Bias() : 
    Module(1, 1)
{
    set_mode(Component::ErrorAgnostic);
}


Bias::~Bias()
{
}
