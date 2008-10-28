#include <iostream>
#include <cstdlib>
#include <cassert>


#include "../layers/common.h"
#include "permutation.h"


void 
permutation_connect(Layer* inlayer_p, Layer* outlayer_p, 
                    int* permutation_p, int blocksize)
{
    permutation_connect(inlayer_p, outlayer_p, 
                        permutation_p, blocksize,
                        0, inlayer_p->outputs.size,
                        0, outlayer_p->inputs.size);
}


void
permutation_connect(Layer* inlayer_p, Layer* outlayer_p, 
                    int* permutation_p, int blocksize,
                    int inlayerstart, int inlayerstop, 
                    int outlayerstart, int outlayerstop)
{
    Connection con;
    con.type = PERMUTATION_CONNECTION;

    con.recurrent = 0;
    
    con.inlayerstart = inlayerstart;
    con.inlayerstop = inlayerstop;
    con.outlayerstart = outlayerstart;
    con.outlayerstop = outlayerstop;
    
    con.internal.permutation_connection_p = \
        (PermutationConnection*) malloc(sizeof(PermutationConnection));
        
    con.internal.permutation_connection_p->permutation_p = permutation_p;
    con.internal.permutation_connection_p->blocksize = blocksize;
        
    con.inlayer_p = inlayer_p;
    con.outlayer_p= outlayer_p;
    
    append_to_array(outlayer_p->incoming_p, outlayer_p->incoming_n, con);
    append_to_array(inlayer_p->outgoing_p, inlayer_p->outgoing_n, con);
}                    


void connection_forward(Connection* con_p, PermutationConnection* c_p)
{
    for(int i = 0; i < con_p->inlayer_p->outputs.size / c_p->blocksize; i++)
    {
        int source_offset = i * c_p->blocksize;
        int target_offset = c_p->permutation_p[i] * c_p->blocksize;
        for(int j = 0; j < c_p->blocksize; j++)
        {
            con_p->outlayer_p->inputs.contents_p[target_offset + j] = \
                con_p->inlayer_p->outputs.contents_p[source_offset + j];
        }
    }
}


void connection_backward(Connection* con_p, PermutationConnection* c_p)
{
    for(int i = 0; i < con_p->outlayer_p->inputs.size / c_p->blocksize; i++)
    {
        int source_offset = i * c_p->blocksize;
        int target_offset = c_p->permutation_p[i] * c_p->blocksize;
        for(int j = 0; j < c_p->blocksize; j++)
        {
            con_p->inlayer_p->outputs.error_p[target_offset + j] = \
                con_p->outlayer_p->inputs.error_p[source_offset + j];
        }
    }

    
}
