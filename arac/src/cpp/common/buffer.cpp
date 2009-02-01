// Part of Arac Neural Network Composition Library.
// (c) 2008 by Justin S Bayer, <bayer.justin@googlemail.com>

#include <iostream>
#include <cstring>

#include "buffer.h"


using arac::common::Buffer;


Buffer::Buffer(int rowsize, bool owner) : 
    _rowsize(rowsize), 
    _owner(owner)
{
    expand();
}


Buffer::~Buffer()
{
    free_memory();
}


void Buffer::add(double* addend_p, int index)
{
    double* current_p = index == -1 ? _content.back() : _content[index];
    for(int i = 0; i < _rowsize; i++)
    {
        current_p[i] += addend_p[i];
    }
}


void Buffer::expand()
{
    if (!owner())
    {
        return;
    }
    double* new_chunk = new double[_rowsize];
    memset((void*) new_chunk, 0, sizeof(double) * _rowsize);
    _content.push_back(new_chunk);
}


void Buffer::clear()
{
    for(int i = 0; i < size(); i++)
    {
        clear_at(i);
    }
}

// TODO: write a test for this
void
Buffer::clear_at(int index)
{
    memset((void*) _content[index], 0, sizeof(double) * _rowsize);
}


void Buffer::free_memory()
{
    if (owner())
    {
        DoublePtrVec::iterator iter;
        for(iter = _content.begin(); iter != _content.end(); iter++)
        {
            delete[] *iter;
        }
    }
    _content.clear();
}



        
