// Part of Arac Neural Network Composition Library.
// (c) 2008 by Justin S Bayer, <bayer.justin@googlemail.com>

#include "connection.h"

using arac::structure::connections::Connection;

Connection::Connection(Module* incoming_p, Module* outgoing_p,
                       int incomingstart, int incomingstop, 
                       int outgoingstart, int outgoingstop) :
    _incoming_p(incoming_p),
    _outgoing_p(outgoing_p),
    _incomingstart(incomingstart),
    _incomingstop(incomingstop),
    _outgoingstart(outgoingstart),
    _outgoingstop(outgoingstop),
    _recurrent(false)
{
    
}                

Connection::Connection(Module* incoming_p, Module* outgoing_p) :
    _incoming_p(incoming_p),
    _outgoing_p(outgoing_p),
    _incomingstart(0),
    _incomingstop(incoming_p->outsize()),
    _outgoingstart(0),
    _outgoingstop(outgoing_p->insize()),
    _recurrent(false)
{
    
}


Connection::~Connection()
{
    
}
