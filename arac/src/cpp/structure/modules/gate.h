// Part of Arac Neural Network Composition Library.
// (c) 2008 by Justin S Bayer, <bayer.justin@googlemail.com>


#ifndef Arac_STRUCTURE_MODULES_GATE_INCLUDED
#define Arac_STRUCTURE_MODULES_GATE_INCLUDED


#include "module.h"


namespace arac {
namespace structure {
namespace modules {


using arac::structure::modules::Module;


// TODO: document.

class GateLayer : public Module
{
    public:

        GateLayer(int size);
        virtual ~GateLayer();

    protected:

        virtual void _forward();
        virtual void _backward();
};


inline GateLayer::~GateLayer() {}


inline GateLayer::GateLayer(int size) :
    Module(size * 2, size)
{
}

    
}
}
}


#endif