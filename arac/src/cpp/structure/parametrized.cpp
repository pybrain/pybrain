// Part of Arac Neural Network Composition Library.
// (c) 2008 by Justin S Bayer, <bayer.justin@googlemail.com>


#include <cstring>

#include "parametrized.h"


using arac::structure::Parametrized;


Parametrized::Parametrized() :
    _parameters_p(0),
    _derivatives_p(0)
{
    
}


Parametrized::Parametrized(int size) 
{
    _size = size;
    _parameters_p = new double[size];
    _derivatives_p = new double[size];

    memset(_parameters_p, 0, size * sizeof(double));
    memset(_derivatives_p, 0, size * sizeof(double));
}


Parametrized::Parametrized(int size, 
                           double* parameters_p, 
                           double* derivatives_p) :
_size(size),
_parameters_p(parameters_p),
_derivatives_p(derivatives_p),
_parameters_owner(false),
_derivatives_owner(false)
{
    
}                         


Parametrized::~Parametrized() {
    if (parameters_owner())
    {
        delete _parameters_p;
    }
    if (derivatives_owner())
    {
        delete _derivatives_p;
    }
}


bool
Parametrized::parameters_owner()
{
    return _parameters_owner;
}


bool
Parametrized::derivatives_owner()
{
    return _derivatives_owner;
}


double*
 Parametrized::get_parameters() const
{
    return _parameters_p;
}


void
Parametrized::set_parameters(double* parameters_p)
{
    if (parameters_owner())
    {
        delete _parameters_p;
    }
    _parameters_p = parameters_p;
    _parameters_owner = false;
}
 
    
double*
Parametrized::get_derivatives() const
{
    return _derivatives_p;
}


void
Parametrized::set_derivatives(double* derivatives_p)
{
    if (derivatives_owner())
    {
        delete _derivatives_p;
    }
    _derivatives_p = derivatives_p;
    _derivatives_owner = false;
}


void
Parametrized::clear_derivatives()
{
    // TODO: test this
    memset(_derivatives_p, 0, _size * sizeof(double));
}


