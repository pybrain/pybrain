// Part of Arac Neural Network Composition Library.
// (c) 2008 by Justin S Bayer, <bayer.justin@googlemail.com>


#ifndef Arac_STRUCTURE_MODULES_MDLSTM_INCLUDED
#define Arac_STRUCTURE_MODULES_MDLSTM_INCLUDED


#include "module.h"


namespace arac {
namespace structure {
namespace modules {


// TODO: document.

class MdlstmLayer : public arac::structure::modules::Module
{
    public:

        MdlstmLayer(int timedim, int size);
        virtual ~MdlstmLayer();

    protected:
        
        virtual void _forward();
        virtual void _backward();
        
        virtual void expand();
        
        int _timedim;
        
        arac::common::Buffer _input_squashed;
        arac::common::Buffer _input_gate_squashed;
        arac::common::Buffer _input_gate_unsquashed;
        arac::common::Buffer _output_gate_squashed;
        arac::common::Buffer _output_gate_unsquashed;
        arac::common::Buffer _forget_gate_unsquashed;
        arac::common::Buffer _forget_gate_squashed;
};


}
}
}


#endif