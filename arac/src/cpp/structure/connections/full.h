// Part of Arac Neural Network Composition Library.
// (c) 2008 by Justin S Bayer, <bayer.justin@googlemail.com>


#ifndef Arac_STRUCTURE_CONNECTIONS_FULL_INCLUDED
#define Arac_STRUCTURE_CONNECTIONS_FULL_INCLUDED


#include "../modules/module.h"
#include "../parametrized.h"
#include "connection.h"


namespace arac {
namespace structure {
namespace connections {

    
using namespace arac::structure::modules;
using arac::structure::Parametrized;


// TODO: document.

class FullConnection : public Connection, public Parametrized
{
    public: 
        
        FullConnection(Module* incoming_p, Module* outgoing_p);
        FullConnection(Module* incoming_p, Module* outgoing_p,
                       int incomingstart, int incomingstop, 
                       int outgoingstart, int outgoingstop);
        FullConnection(Module* incoming_p, Module* outgoing_p,
                       double* parameters_p, double* derivatives_p,
                       int incomingstart, int incomingstop, 
                       int outgoingstart, int outgoingstop);
        virtual ~FullConnection();
        
    protected:
        
        virtual void _forward();
        virtual void _backward();
};    
    
}
}
}


#endif