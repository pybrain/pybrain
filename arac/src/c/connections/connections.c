// Part of Arac Neural Network Composition Library.
// (c) 2008 by Justin S Bayer, <bayer.justin@googlemail.com>


#include <iostream>
#include "connections.h"
#include "../layers/layers.h"


void forward(Connection* con_p)
{
    assert(con_p->inlayerstart < con_p->inlayer_p->outputs.size);
    assert(con_p->outlayerstart < con_p->outlayer_p->inputs.size);

    // Since the slices are [start, stop), we need to allow endings that are
    // one bigger than the length.
    assert(con_p->inlayerstop <= con_p->inlayer_p->outputs.size + 1);
    assert(con_p->outlayerstop <= con_p->outlayer_p->inputs.size + 1);
    
    assert(con_p->inlayerstart < con_p->inlayerstop);
    assert(con_p->outlayerstart < con_p->outlayerstop);

    // Don't forward recurrent connections in the first timestep.
    std::cout << "recurrent? " << con_p->recurrent << " \n";
    if (con_p->recurrent)
    {
        std::cout << "not forwarding, recurrent \n";
        if (*(con_p->inlayer_p->timestep_p) == 0)
        {
            return;
        }
    }
    switch (con_p->type) 
    {
        case IDENTITY_CONNECTION:
            connection_forward(con_p, con_p->internal.identity_connection_p);
            break;
        case FULL_CONNECTION:
            connection_forward(con_p, con_p->internal.full_connection_p);
            break;
        case PERMUTATION_CONNECTION:
            connection_forward(con_p, con_p->internal.permutation_connection_p);
            break;

    }
}


void backward(Connection* con_p) 
{
    // Don't backward recurrent connections in the last timestep.
    if (con_p->recurrent)
    {
        if (*(con_p->inlayer_p->timestep_p) == *(con_p->inlayer_p->seqlen_p))
        {
            return;
        }
    }
    switch (con_p->type) 
    {
        case IDENTITY_CONNECTION:
            connection_backward(con_p, con_p->internal.identity_connection_p);
            break;
        case FULL_CONNECTION:
            connection_backward(con_p, con_p->internal.full_connection_p);
            break;
        case PERMUTATION_CONNECTION:
            connection_backward(con_p, con_p->internal.permutation_connection_p);
            break;
    }
}

