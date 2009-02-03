// Part of Arac Neural Network Composition Library.
// (c) 2008 by Justin S Bayer, <bayer.justin@googlemail.com>


#ifndef Arac_STRUCTURE_MODULES_BIAS_INCLUDED
#define Arac_STRUCTURE_MODULES_BIAS_INCLUDED

#include <iostream>
#include "module.h"


namespace arac {
namespace structure {
namespace modules {


using arac::structure::modules::Module;


// TODO: document.

class Bias : public Module
{

    public:
        Bias();
        virtual ~Bias();

    protected:

        virtual void _forward();
        virtual void _backward();
    
};


inline
void
Bias::_forward()
{
    output()[timestep()][0] = 1;
}


inline
void
Bias::_backward()
{
    ;
}



}
}
}


#endif
