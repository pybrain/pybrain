// Part of Arac Neural Network Composition Library.
// (c) 2008 by Justin S Bayer, <bayer.justin@googlemail.com>


#ifndef Arac_STRUCTURE_NETWORKS_MDRNNS_BASEMDRNN_INCLUDED
#define Arac_STRUCTURE_NETWORKS_MDRNNS_BASEMDRNN_INCLUDED


#include "../basenetwork.h"


namespace arac {
namespace structure {
namespace networks {
namespace mdrnns {


class BaseMdrnn : public BaseNetwork
{
    public:
        
        BaseMdrnn(int timedim);
        virtual ~BaseMdrnn();

    protected:
        
        int _timedim;
};



}
}
}
}


#endif