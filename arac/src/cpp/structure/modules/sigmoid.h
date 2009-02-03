// Part of Arac Neural Network Composition Library.
// (c) 2008 by Justin S Bayer, <bayer.justin@googlemail.com>


#ifndef Arac_STRUCTURE_MODULES_SIGMOID_INCLUDED
#define Arac_STRUCTURE_MODULES_SIGMOID_INCLUDED


#include "module.h"


namespace arac {
namespace structure {
namespace modules {

using arac::structure::modules::Module;


// TODO: document.

class SigmoidLayer : public Module
{
    public:

        SigmoidLayer(int size);
        virtual ~SigmoidLayer();

    protected:
        
        virtual void _forward();
        virtual void _backward();
};


inline SigmoidLayer::~SigmoidLayer() {}


inline SigmoidLayer::SigmoidLayer(int size) :
    Module(size, size)
{
}

    
}
}
}


#endif
