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
        
        arac::common::Buffer& input_squashed();
        
        arac::common::Buffer& output_gate_squashed();
        arac::common::Buffer& output_gate_unsquashed();
        
        arac::common::Buffer& input_gate_squashed();
        arac::common::Buffer& input_gate_unsquashed();

        arac::common::Buffer& forget_gate_squashed();
        arac::common::Buffer& forget_gate_unsquashed();

    private:
        
        // Set the intermediate buffers to zero.
        // TODO: find better name.
        void clear_intermediates();
        
        virtual void _forward();
        virtual void _backward();
        
        virtual void expand();
        
        int _timedim;
        
        arac::common::Buffer _input_squashed;
        
        arac::common::Buffer _input_gate_squashed;
        arac::common::Buffer _input_gate_unsquashed;
        
        arac::common::Buffer _output_gate_squashed;
        arac::common::Buffer _output_gate_unsquashed;
        
        arac::common::Buffer _forget_gate_squashed;
        arac::common::Buffer _forget_gate_unsquashed;
        
        // Intermediate buffers.
        double* _inter_input_p;
        double* _output_state_p;
        double* _input_state_p;
        double* _output_error_p;
        double* _output_state_error_p;
        double* _output_gate_error_p;
        double* _forget_gate_error_p;
        double* _input_gate_error_p;
        double* _input_error_p;
        double* _input_state_error_p;
        double* _state_error_p;
        
        double* _outputbuffer_p;
};


inline
arac::common::Buffer& 
MdlstmLayer::input_squashed()
{
    return _input_squashed;
}


inline
arac::common::Buffer& 
MdlstmLayer::output_gate_squashed()
{
    return _output_gate_squashed;
    
}


inline
arac::common::Buffer&
MdlstmLayer::output_gate_unsquashed()
{
    return _output_gate_unsquashed;
}


inline
arac::common::Buffer& 
MdlstmLayer::input_gate_squashed()
{
    return _input_gate_squashed;
}


inline
arac::common::Buffer& 
MdlstmLayer::input_gate_unsquashed()
{
    return _input_gate_unsquashed;
}


inline
arac::common::Buffer&
MdlstmLayer::forget_gate_squashed()
{
    return _forget_gate_squashed;
}


inline
arac::common::Buffer& 
MdlstmLayer::forget_gate_unsquashed()
{
    return _forget_gate_unsquashed;
}


} } }  // Namespace.


#endif