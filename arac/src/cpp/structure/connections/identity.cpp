// Part of Arac Neural Network Composition Library.
// (c) 2008 by Justin S Bayer, <bayer.justin@googlemail.com>


#include <cstring>
#include <iostream>

#include "identity.h"


using arac::structure::connections::Connection;
using arac::structure::connections::IdentityConnection;
using arac::structure::modules::Module;


IdentityConnection::IdentityConnection(Module* incoming_p, Module* outgoing_p) :
    Connection(incoming_p, outgoing_p)
{
    
}


IdentityConnection::IdentityConnection(Module* incoming_p, Module* outgoing_p,
                                       int incomingstart, int incomingstop, 
                                       int outgoingstart, int outgoingstop) :
    Connection(incoming_p, outgoing_p, 
               incomingstart, incomingstop, 
               outgoingstart, outgoingstop)
{
    
}


IdentityConnection::~IdentityConnection()
{
    
}


void
IdentityConnection::_forward()
{
    if ((_recurrent) && (timestep() == 0))
    {
        // Don't use recurrent cons in the first timestep.
        return;
    }
    
    double* sourcebuffer_p = _recurrent ? _incoming_p->output()[timestep() - 1] :
                                          _incoming_p->output()[timestep()];
    sourcebuffer_p += _incomingstart;
    
    double* sinkbuffer_p = _outgoing_p->input()[_timestep] + _outgoingstart;
    int size = (_incomingstop - _incomingstart);
    for(int i = 0; i < size; i++)
    {
        sinkbuffer_p[i] += sourcebuffer_p[i];
    }
}


void
IdentityConnection::_backward()
{
    int this_timestep = timestep() - 1;
    if (this_timestep - get_recurrent() < 0)
    {
        return;
    }
    
    double* sinkbuffer_p = _incoming_p->outerror()[this_timestep - get_recurrent()];
    sinkbuffer_p += _incomingstart;

    double* sourcebuffer_p = _outgoing_p->inerror()[this_timestep] + _outgoingstart;
    int size = (_incomingstop - _incomingstart);
    for(int i = 0; i < size; i++)
    {
        sinkbuffer_p[i] += sourcebuffer_p[i];
    }
}
