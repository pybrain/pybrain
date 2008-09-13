// Part of Arac Neural Network Composition Library.
// (c) 2008 by Justin S Bayer, <bayer.justin@googlemail.com>

//
// Header file that accumulates all the different connections.
//


#ifndef Arac_CONNECTIONS_INCLUDED
#define Arac_CONNECTIONS_INCLUDED


#include <assert.h>

#include "../common.h"
#include "identity.h"
#include "full.h"


//
// Prototype functions
// 
    
// 
// Function that delegates to the specific implementations of connection_forward
// and connection_backwardby different types.
//

void forward(Connection* con_p);

void backward(Connection* con_p);


// The connection_forward function is overloaded for different types of 
// connections. Due to the union structure, for the specialized functions the 
// connection pointer is being passed in again, though not necessarily needed.

#endif  // Arac_CONNECTIONS_INCLUDED

