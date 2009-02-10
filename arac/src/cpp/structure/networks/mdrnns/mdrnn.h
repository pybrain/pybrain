// Part of Arac Neural Network Composition Library.
// (c) 2008 by Justin S Bayer, <bayer.justin@googlemail.com>


#ifndef Arac_STRUCTURE_NETWORKS_MDRNNS_MDRNN_INCLUDED
#define Arac_STRUCTURE_NETWORKS_MDRNNS_MDRNN_INCLUDED


#include <cassert>

#include "basemdrnn.h"
#include "../../connections/connections.h"
#include "../../modules/modules.h"
#include "../../parametrized.h"


namespace arac {
namespace structure {
namespace networks {
namespace mdrnns {


template <class module_type>
class Mdrnn : public BaseMdrnn, public arac::structure::Parametrized
{
    public:
        
        Mdrnn(int timedim, int hiddensize);
        virtual ~Mdrnn();

        void set_sequence_shape(int dim, int val);
        const double* get_sequence_shape();

        int sequencelength();

        void set_block_shape(int dim, int val);
        const double* get_block_shape();

        int blocksize();

        void sort();
        
        virtual void _forward();
        
        virtual void _backward();
        
        
    protected:
        
        void init_multiplied_sizes();
        
        void next_coords(double* coords);
        void coords_by_index(double* coords_p, int index);
        void index_by_coords(int& index, double* coords_p);
        
        int _hiddensize;
        
        int _sequencelength;
        int _blocksize;
        
        double* _sequence_shape_p;
        double* _block_shape_p;
        
        // Size of the previous dimensions in memory; example: if a shape of 
        // (4, 4, 4) is given, each element holds the product of the previous
        // dimensions: (1, 4, 16) with the special case of the first element 
        // being one.
        double* _multiplied_sizes_p;
        
        module_type* _module_p;
        
        std::vector<arac::structure::connections::FullConnection*> _connections;
};


template <class module_type>
inline
void
Mdrnn<module_type>::set_sequence_shape(int dim, int val)
{
    assert(dim < _timedim);
    if (_sequence_shape_p[dim] == val)
    {
        return;
    }
    _dirty = true;
    _sequence_shape_p[dim] = val;
    
    _sequencelength = 1;
    for (int i = 0; i < _timedim; i++)
    {
        _sequencelength *= _sequence_shape_p[i];
    }
    _insize = _sequencelength;
    _outsize = _sequencelength;
}


template <class module_type>
inline
const double*
Mdrnn<module_type>::get_sequence_shape()
{
    return _sequence_shape_p;
}


template <class module_type>
inline
int
Mdrnn<module_type>::blocksize()
{
    return _blocksize;
}


template <class module_type>
inline
void
Mdrnn<module_type>::set_block_shape(int dim, int val)
{
    assert(dim < _timedim);
    if (_block_shape_p[dim] == val)
    {
        return;
    }
    
    _dirty = true;
    _block_shape_p[dim] = val;
    
    _blocksize = 1;
    for (int i = 0; i < _timedim; i++)
    {
        _blocksize *= _block_shape_p[i];
    }
}


template <class module_type>
inline
const double*
Mdrnn<module_type>::get_block_shape()
{
    return _block_shape_p;
}


template <class module_type>
inline
int
Mdrnn<module_type>::sequencelength()
{
    return _sequencelength;
}


template <class module_type>
inline
void
Mdrnn<module_type>::next_coords(double* coords_p)
{
    int i;
    int carry = 0;
    for(i = 0; i < _timedim; i++)
    {
        if (coords_p[i] < _sequence_shape_p[i] - 1)
        {
            coords_p[i] += 1;
            break;
        }
        else 
        {
            coords_p[i] = 0;
        }
    }
}


template <class module_type>
inline
void
Mdrnn<module_type>::coords_by_index(double* coords_p, int index)
{
    int divisor = sequencelength() / blocksize();
    for(int i = _timedim - 1; i <= 0; i--)
    {
        divisor /= _sequence_shape_p[i] / _block_shape_p[i];
        coords_p[i] = index / divisor;
        index = index % divisor;
    }
}

template <class module_type>
inline
void
Mdrnn<module_type>::index_by_coords(int& index, double* coords_p)
{
    index = 0;
    int smallcubesize = 1;
    for(int i = 0; i < _timedim; i++)
    {
        index += coords_p[i] * _sequence_shape_p[i] / _block_shape_p[i];
        smallcubesize *= _sequence_shape_p[i] / _block_shape_p[i];
    }
}


}
}
}
}

// HACK, because this is a template and we want to make a library. See:
//
//      http://yosefk.com/c++fqa/templates.html#fqa-35.13
//
//  for details.

#include "mdrnn.cpp"

#endif