// Part of Arac Neural Network Composition Library.
// (c) 2008 by Justin S Bayer, <bayer.justin@googlemail.com>


#ifndef Arac_STRUCTURE_CONNECTIONS_CONNECTION_INCLUDED
#define Arac_STRUCTURE_CONNECTIONS_CONNECTION_INCLUDED


#include <cassert>

#include "../component.h"
#include "../modules/module.h"



namespace arac {
namespace structure {
namespace connections {
    
    
using namespace arac::structure::modules;

// TODO: document.

class Connection : public arac::structure::Component
{
    public: 
        
        Connection(Module* incoming, Module* outgoing,
                   int incomingstart, int incomingstop, 
                   int outgoingstart, int outgoingstop);
        Connection(Module* incoming, Module* outgoing);
        virtual ~Connection();
        
        void set_incomingstart(int n);
        void set_incomingstop(int n);
        void set_outgoingstart(int n);
        void set_outgoingstop(int n);
        
        int get_incomingstart();
        int get_incomingstop();
        int get_outgoingstart();
        int get_outgoingstop();
        
        void set_recurrent(int recurrent);
        int get_recurrent();
        
        Module* incoming();
        Module* outgoing();
        
    protected:
        
        Module* _incoming_p;
        Module* _outgoing_p;
        
        int _incomingstart;
        int _incomingstop;
        int _outgoingstart;
        int _outgoingstop;
        
        int _recurrent;
};
    
    
inline
void
Connection::set_incomingstart(int n)
{
    _incomingstart = n;
    
}


inline
void
Connection::set_incomingstop(int n)
{
    _incomingstop = n;
}


inline
void
Connection::set_outgoingstart(int n)
{
    _outgoingstart = n;
}


inline
void
Connection::set_outgoingstop(int n)
{
    _outgoingstop = n;
}


inline
int
Connection::get_incomingstart()
{
    return _incomingstart;
}


inline
int
Connection::get_incomingstop()
{
    return _incomingstop;
}


inline
int
Connection::get_outgoingstart()
{
    return _outgoingstart;
    
}


inline
int
Connection::get_outgoingstop()
{
    return _outgoingstop;
}


inline
int
Connection::get_recurrent()
{
    return _recurrent;
}


inline
void
Connection::set_recurrent(int recurrent)
{
    assert((!recurrent) || (sequential()));
    _recurrent = recurrent;
}


inline
Module* 
Connection::incoming()
{
    return _incoming_p;
}


inline
Module* 
Connection::outgoing()
{
    return _outgoing_p;
}

    
}
}
}


#endif