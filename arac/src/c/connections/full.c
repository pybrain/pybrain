#include <iostream>
#include <cstdlib>
#include <cassert>

extern "C"
{
    #include "cblas.h"
}

#include "../layers/common.h"
#include "full.h"




void
full_connect(Layer* inlayer_p, Layer* outlayer_p, double* params)
{
    full_connect(inlayer_p, outlayer_p, 
                 params,
                 0, inlayer_p->outputs.size,
                 0, outlayer_p->inputs.size);
}


void
full_connect(Layer* inlayer_p, Layer* outlayer_p, 
             double* params, 
             int inlayerstart, int inlayerstop, 
             int outlayerstart, int outlayerstop)
{
    Connection con;
    con.type = FULL_CONNECTION;
    con.recurrent = 0;

    con.inlayerstart = inlayerstart;
    con.inlayerstop = inlayerstop;
    con.outlayerstart = outlayerstart;
    con.outlayerstop = outlayerstop;
    
    con.internal.full_connection_p = \
        (FullConnection*) malloc(sizeof(FullConnection));
    
    int size = inlayer_p->outputs.size * outlayer_p->inputs.size;
    con.internal.full_connection_p->weights.size = size;
        
    con.internal.full_connection_p->weights.contents_p = params;
    con.internal.full_connection_p->weights.error_p = \
        (double*) malloc(sizeof(double) * size);
    
    con.inlayer_p = inlayer_p;
    con.outlayer_p = outlayer_p;
    
    append_to_array(outlayer_p->incoming_p, outlayer_p->incoming_n, con);
    append_to_array(inlayer_p->outgoing_p, inlayer_p->outgoing_n, con);
}         

void
connection_forward(Connection* con_p, FullConnection* fc_p)
{
    // Shortcuts
    Layer* from_p = con_p->inlayer_p;
    Layer* to_p = con_p->outlayer_p;

    // This will be zero for non-recurrent networks.
    int timestep = (*from_p->timestep_p);
    
    // Buffer incrementers with respect to time.
    if (con_p->recurrent) 
    {
        // Move on timestep back if the connection is a recurrent one.
        timestep -= 1;
    }
    
    int bi_from = timestep * from_p->outputs.size + con_p->inlayerstart;
    int bi_to = timestep * to_p->inputs.size + con_p->outlayerstart;
    
    // TODO: remove
    std::cout << "From " << bi_from << " to " << bi_to << std::endl;

    int indim = con_p->inlayerstop - con_p->inlayerstart;
    int outdim = con_p->outlayerstop - con_p->outlayerstart;

    cblas_dgemv(CblasRowMajor, 
                // Transpose the matrix since we want to multiply from the right
                CblasNoTrans,
                // Dimensions of the matrix
                outdim,        
                indim,
                // Scalar for the matrix
                1.0,                    
                // Pointer to the matrix
                fc_p->weights.contents_p,    
                // Dimension of the vector
                indim,
                // Pointer to the vector
                from_p->outputs.contents_p + bi_from,   
                // ??? some incrementer
                1,                      
                // Scalar of the target vector
                1.0,                    
                // Pointer to the target vector
                to_p->inputs.contents_p + bi_to,         
                // ??? some incrementer
                1);   
}


void connection_backward(Connection* con_p, FullConnection* fc_p)
{
    // Shortcuts
    Layer* from_p = con_p->inlayer_p;
    Layer* to_p = con_p->outlayer_p;
    
    int indim = from_p->outputs.size;
    int outdim = to_p->inputs.size;
    
    // Buffer incrementers with respect to time
    // We have to subtract 1 since the timestep will already be incremented 
    // after activation. 
    int timestep = *(from_p->timestep_p) - 1; 
    if (con_p->recurrent) {
        timestep += 1;
    }
    int bi_in = indim * timestep;
    int bi_out = outdim * timestep;
    
    double* inerror_p = from_p->outputs.error_p + bi_in;
    double* outerror_p = to_p->inputs.error_p + bi_out;
    double* inbuffer_p = from_p->outputs.contents_p + bi_in;
    double* weights_p = fc_p->weights.contents_p + bi_out;
    
    assert(fc_p->weights.size == indim * outdim);
    
    for (int i = 0; i < outdim; i++)
    {
        for (int j = 0; j < indim; j++)
        {
            inerror_p[j] += *weights_p * outerror_p[i];
            weights_p++;
        }
    }

    weights_p = fc_p->weights.contents_p;
    double* derivs_p = fc_p->weights.error_p;
    
    for (int i = 0; i < outdim; i++)
    {
        for (int j = 0; j < indim; j++)
        {
            *derivs_p += outerror_p[i] * inbuffer_p[j];
            derivs_p++;
        }
    }
}

