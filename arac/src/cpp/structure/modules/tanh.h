// Part of Arac Neural Network Composition Library.
// (c) 2008 by Justin S Bayer, <bayer.justin@googlemail.com>


#ifndef Arac_STRUCTURE_MODULES_TANH_INCLUDED
#define Arac_STRUCTURE_MODULES_TANH_INCLUDED


#include "module.h"


namespace arac {
namespace structure {
namespace modules {

using arac::structure::modules::Module;


// TODO: document.

class TanhLayer : public Module
{
    public:

        TanhLayer(int size);
        virtual ~TanhLayer();
        
    protected:
        
        virtual void _forward();
        virtual void _backward();
};


inline TanhLayer::~TanhLayer() {}


inline TanhLayer::TanhLayer(int size) :
    Module(size, size)
{
}

    
}
}
}


#endif
