// Part of Arac Neural Network Composition Library.
// (c) 2008 by Justin S Bayer, <bayer.justin@googlemail.com>


#ifndef Arac_STRUCTURE_CONNECTIONS_IDENTITY_INCLUDED
#define Arac_STRUCTURE_CONNECTIONS_IDENTITY_INCLUDED


#include "../modules/module.h"
#include "connection.h"


namespace arac {
namespace structure {
namespace connections {

    
using namespace arac::structure::modules;
using arac::structure::Component;


// TODO: document.

class IdentityConnection : public Connection 
{
    public:

        IdentityConnection(Module* incoming_p, Module* outgoing_p);
        IdentityConnection(Module* incoming_p, Module* outgoing_p,
                           int incomingstart, int incomingstop, 
                           int outgoingstart, int outgoingstop);
        virtual ~IdentityConnection();
    
    protected:
        
        virtual void _forward();
        virtual void _backward();
};


}
}
}


#endif