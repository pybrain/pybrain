// Part of Arac Neural Network Composition Library.
// (c) 2008 by Justin S Bayer, <bayer.justin@googlemail.com>


#ifndef Arac_STRUCTURE_MODULES_LINEAR_INCLUDED
#define Arac_STRUCTURE_MODULES_LINEAR_INCLUDED


#include "module.h"


namespace arac {
namespace structure {
namespace modules {


using arac::structure::modules::Module;


// TODO: document.

class LinearLayer : public Module
{
    public:

        LinearLayer(int size);
        virtual ~LinearLayer();

    protected:

        virtual void _forward();
        virtual void _backward();
};


inline LinearLayer::~LinearLayer() {}


inline LinearLayer::LinearLayer(int size) :
    Module(size, size)
{
}

    
}
}
}


#endif