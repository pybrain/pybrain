// Part of Arac Neural Network Composition Library.
// (c) 2008 by Justin S Bayer, <bayer.justin@googlemail.com>


#include <cassert>
#include <cstring>

#include "lstm.h"


using arac::structure::modules::LstmLayer;
using arac::structure::Component;
using arac::common::Buffer;


LstmLayer::LstmLayer(int size) :
    _mdlstm(size, 1),
    Module(4 * size, size),
    _state_p(new Buffer(size)),
    _state_error_p(new Buffer(size))
{
    set_mode(Component::Sequential);
}


LstmLayer::~LstmLayer()
{
    if (_state_p)
    {
        delete _state_p;
    }
    if (_state_error_p)
    {
        delete _state_error_p;
    }
}


void LstmLayer::set_mode(Mode mode)
{
    if(!(mode & Component::Sequential))
    {
        // FIXME: Error handling
    }
    _mdlstm.set_mode(Component::Sequential);
    Component::set_mode(mode);
}


void LstmLayer::expand()
{
    state().expand();
    state_error().expand();
    Module::expand();
}


void
LstmLayer::fill_internal_input()
{
    // Copy input into internal MdlstmLayer.
    memcpy(_mdlstm.input()[timestep()], 
           input()[timestep()], 
           insize() * sizeof(double));
}


void
LstmLayer::fill_internal_state()
{
    // Copy states into inputbuffer of internal MDLSTM; fill up with zero if
    // we are in the first timestep.
    double* state_p = _mdlstm.input()[timestep()] + insize();
    if (timestep() > 0)
    {
        memcpy(state_p, 
               state()[timestep() - 1], 
               outsize() * sizeof(double));
    }
    else
    {
        memset(state_p, 0, outsize() * sizeof(double));
    }
}


void
LstmLayer::retrieve_internal_output()
{
    // Copy information back.
    memcpy(output()[timestep()], 
           _mdlstm.output()[timestep()], 
           outsize() * sizeof(double));
}


void
LstmLayer::retrieve_internal_state()
{
    memcpy(state()[timestep()], 
           _mdlstm.output()[timestep()] + outsize(),
           outsize() * sizeof(double));
}


void
LstmLayer::fill_internal_outerror()
{
    memcpy(_mdlstm.outerror()[timestep() - 1], 
           outerror()[timestep() - 1],
           outsize() * sizeof(double));
}


void
LstmLayer::fill_internal_state_error()
{
    if (timestep() - 1 >= input().size())
    {
        memset((void*) (_mdlstm.outerror()[timestep() - 1] + outsize()),
               0,
               outsize() * sizeof(double));
    }
    else 
    {
        assert(timestep() < state_error().size());
        memcpy(_mdlstm.outerror()[timestep() - 1] + outsize(),
               state_error()[timestep()],
               outsize() * sizeof(double));
    }
}


void
LstmLayer::retrieve_internal_inerror()
{
    memcpy((void*) inerror()[timestep() - 1],
           (void*) _mdlstm.inerror()[timestep() - 1],
           insize() * sizeof(double));
}


void 
LstmLayer::retrieve_internal_state_error()
{
    memcpy(state_error()[timestep() - 1],
           _mdlstm.inerror()[timestep() - 1] + insize(),
           outsize() * sizeof(double));
}


void
LstmLayer::_forward()
{
    fill_internal_input();
    fill_internal_state();
    _mdlstm.forward();
    retrieve_internal_output();
    retrieve_internal_state();
}


void
LstmLayer::_backward()
{
    fill_internal_outerror();
    fill_internal_state_error();
    _mdlstm.backward();
    retrieve_internal_inerror();
    retrieve_internal_state_error();
}
