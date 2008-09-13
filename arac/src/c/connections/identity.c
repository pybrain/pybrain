#include <iostream>

#include "../layers/layers.h"
#include "identity.h"
#include "common.h"


void 
identity_connect(Layer* inlayer_p, Layer* outlayer_p)
{
    identity_connect(inlayer_p, outlayer_p, 
                     0, inlayer_p->outputs.size,
                     0, outlayer_p->inputs.size);
}


void 
identity_connect(Layer* inlayer_p, Layer* outlayer_p,
                 int inlayerstart, int inlayerstop, 
                 int outlayerstart, int outlayerstop)
{
    // Only works for slices of the same length.
    assert(inlayerstart - inlayerstop == outlayerstart - outlayerstop);
    
    Connection con;
    con.type = IDENTITY_CONNECTION;
    con.recurrent = 0;
    
    con.inlayerstart = inlayerstart;
    con.inlayerstop = inlayerstop;
    con.outlayerstart = outlayerstart;
    con.outlayerstop = outlayerstop;
    
    con.inlayer_p = inlayer_p;
    con.outlayer_p = outlayer_p;
    
    append_to_array(inlayer_p->outgoing_p, inlayer_p->outgoing_n, con);
    append_to_array(outlayer_p->incoming_p, outlayer_p->incoming_n, con);
}


void 
connection_forward(Connection* con_p, IdentityConnection* lc_p)
{
    assert(con_p->inlayerstart - con_p->inlayerstop \
        == con_p->outlayerstart - con_p->outlayerstop);

    Layer* inlayer_p = con_p->inlayer_p;
    Layer* outlayer_p = con_p->outlayer_p;

    int timestep = *(inlayer_p->timestep_p);
    if (con_p->recurrent)
    {
        timestep -= 1;
    }

    // Increment for the bufferpointers that is needed due to the timestep
    int bi = timestep * inlayer_p->outputs.size;
    
    // Amount of units that are being forwarded depending on the connection
    int steps = con_p->inlayerstop - con_p->outlayerstart;
    
    // Shortcuts for the double arrays
    double* from_p = \
        inlayer_p->outputs.contents_p + bi + con_p->inlayerstart;
    double* to_p = \
        outlayer_p->inputs.contents_p + bi + con_p->outlayerstart;
    for (int i = 0; i < steps; i++)
    {
        to_p[i] += from_p[i];
    }

}


void
connection_backward(Connection* con_p, IdentityConnection* lc_p)
{
    Layer* inlayer_p = con_p->inlayer_p;
    Layer* outlayer_p = con_p->outlayer_p;
    assert(inlayer_p->outputs.size == outlayer_p->inputs.size);
    
    int size = inlayer_p->outputs.size;
    
    // We have to subtract 1 since the timestep will already be incremented 
    // after activation - except if this is a recurrent connection, since this
    // one looks into the future anyways.
    int timestep = *(inlayer_p->timestep_p);
    if (!con_p->recurrent)
    {
        timestep--;
    }
    // Increment for the bufferpointers that is needed due to the timestep
    int bi = timestep * size;
    // Amount of units that are being forwarded depending on the connection
    int steps = con_p->inlayerstop - con_p->inlayerstart;
    // Shortcuts for the double arrays
    double* from_p = \
        outlayer_p->inputs.error_p + bi + con_p->outlayerstart;
    double* to_p = \
        inlayer_p->outputs.error_p + bi + con_p->inlayerstart;

    for (int i = 0; i < inlayer_p->outputs.size; i++)
    {
        to_p[i] += from_p[i];
    }
}


