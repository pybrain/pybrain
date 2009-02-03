// Part of Arac Neural Network Composition Library.
// (c) 2008 by Justin S Bayer, <bayer.justin@googlemail.com>


#ifndef ARAC_MDRNN_C
#define ARAC_MDRNN_C


#include <iostream>

#include "mdrnn.h"


using arac::structure::Component;
using arac::structure::connections::FullConnection;
using arac::structure::networks::mdrnns::Mdrnn;


template <class module_type>
Mdrnn<module_type>::Mdrnn(int timedim, int hiddensize) :
    BaseMdrnn(timedim),
    Parametrized(hiddensize * hiddensize * timedim),
    _hiddensize(hiddensize),
    _module_p(0)
{
    _sequence_shape_p = new double[_timedim];
    _block_shape_p = new double[_timedim];
    _multiplied_sizes_p = new double[_timedim];
    for(int i = 0; i < _timedim; i++)
    {
        set_sequence_shape(i, 1);
        set_block_shape(i, 1);
    }
}


template <class module_type>
Mdrnn<module_type>::~Mdrnn()
{
    delete[] _sequence_shape_p;
    delete[] _block_shape_p;
    delete _module_p;
    
    std::vector<FullConnection*>::iterator con_iter;
    for(con_iter = _connections.begin(); 
        con_iter != _connections.end(); 
        con_iter++)
    {
        delete *con_iter;
    }
}


template <class module_type>
void
Mdrnn<module_type>::init_multiplied_sizes()
{
    int size = 1;
    for(int i = 0; i < _timedim; i++)
    {
        _multiplied_sizes_p[i] = size;
        size *= _sequence_shape_p[i];
    }
}


template <class module_type>
void
Mdrnn<module_type>::sort()
{
    // Initialize multilied sizes.
    init_multiplied_sizes();
    
    // Initialize module.
    if (_module_p != 0)
    {
        delete _module_p;
    }
    _module_p = new module_type(_hiddensize * _blocksize);
    _module_p->set_mode(Component::Sequential);
    
    // Initialize recurrent self connections.
    int recurrency = 1;
    for(int i = 0; i < _timedim; i++)
    {
        FullConnection* con_p = new FullConnection(_module_p, _module_p);
        con_p->set_mode(Component::Sequential);
        con_p->set_recurrent(recurrency);
        recurrency *= _sequence_shape_p[i] / _block_shape_p[i];
        con_p->set_parameters(_parameters_p + i * _hiddensize * _hiddensize);
        _connections.push_back(con_p);
    }
    
    // Ininitialize buffers.
    init_buffers();
    
    // Indicate thast the net is ready for use.
    _dirty = false;
}


template <class module_type>
void
Mdrnn<module_type>::_forward()
{
    // We keep the coordinates of the current block in here.
    double* coords_p = new double[_timedim];
    // TODO: save memory by not copying but referencing.
    for(int i = 0; i < sequencelength(); i++)
    {
        std::vector<FullConnection*>::iterator con_iter;
        int j = 0;
        for(con_iter = _connections.begin(); 
            con_iter != _connections.end(); 
            con_iter++)
        {
            // If the current coordinate is zero, we are at a border of the 
            // input in that dimension. In that case, the connections may not be
            // forwarded, since we don't want to look around corners.
            if (coords_p[j] == 0)
            {
                (*con_iter)->dry_forward();
            }
            else
            {
                (*con_iter)->forward();
            }
            j++;
        }
        _module_p->add_to_input(input()[timestep()] + blocksize() * i);
        _module_p->forward();
        next_coords(coords_p);
    }
    // Copy the output to the mdrnns outputbuffer.
    // TODO: save memory by not copying but referencing.
    std::vector<double*>::iterator dblp_iter;
    for(int i = 0; i < sequencelength(); i++)
    {
        memcpy(output()[timestep()] + i * blocksize(), 
               _module_p->output()[i], 
               blocksize() * sizeof(double));
    }
}


template <class module_type>
void
Mdrnn<module_type>::_backward()
{
    // We keep the coordinates of the current block in here.
    double* coords_p = new double[_timedim];
    // TODO: save memory by not copying but referencing.
    for(int i = sequencelength() - 1; i >= 0; i--)
    {
        std::vector<FullConnection*>::iterator con_iter;
        int j = 0;
        for(con_iter = _connections.begin(); 
            con_iter != _connections.end(); 
            con_iter++)
        {
            // If the current coordinate is zero, we are at a border of the 
            // input in that dimension. In that case, the connections may not be
            // forwarded, since we don't want to look around corners.
            if (coords_p[j] == 0)
            {
                (*con_iter)->dry_backward();
            }
            else
            {
                (*con_iter)->backward();
            }
            j++;
        }
        _module_p->add_to_outerror(outerror()[timestep() - 1] + blocksize() * i);
        _module_p->backward();
        next_coords(coords_p);
    }
    // Copy the output to the mdrnns outputbuffer.
    // TODO: save memory by not copying but referencing.
    for(int i = 0; i < sequencelength(); i++)
    {
        memcpy(inerror()[timestep() - 1] + i * blocksize(), 
               _module_p->outerror()[i], 
               blocksize() * sizeof(double));
    }
}



#endif
