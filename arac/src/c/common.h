// Part of Arac Neural Network Composition Library.
// (c) 2008 by Justin S Bayer, <bayer.justin@googlemail.com>

//
// Header file for common functionality.
//


#ifndef Arac_COMMON_INCLUDED
#define Arac_COMMON_INCLUDED


//
// Struct to hold parameters. Since these parameters will be adjusted during the
// learning process, there is also a field for errors.
//

struct ParameterContainer {
    int size;
    double* contents_p;
    double* error_p;
};


#endif

