#include <cstdlib>
#include <cstring>
#include <iostream>
#include "common.h"


void
append_to_array(Connection*& begin, int& counter, Connection con)
{
    if (((counter > 0) && ((counter & (counter - 1)) == 0)) || counter == 0) 
    {
        // We need to allocate new space if either the element count is a power 
        // of two or it is zero.
        int newsize;
        if (counter != 0)
        {
            newsize = counter << 1;
        }
        else
        {
            newsize = 1;
        }
        void* newarray = malloc(newsize * sizeof(Connection));
        memcpy(newarray, (void*) begin, counter * sizeof(Connection));
        free(begin);
        begin = (Connection*) newarray;
    }
    
    begin[counter] = con;
    counter++;
}

