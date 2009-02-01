// Part of Arac Neural Network Composition Library.
// (c) 2008 by Justin S Bayer, <bayer.justin@googlemail.com>


#ifndef Arac_STRUCTURE_MODULES_LSTM_INCLUDED
#define Arac_STRUCTURE_MODULES_LSTM_INCLUDED


#include "module.h"
#include "mdlstm.h"
#include "../../common/common.h"


namespace arac {
namespace structure {
namespace modules {


// TODO: document.

class LstmLayer : public Module
{
    public:

        LstmLayer(int size);
        virtual ~LstmLayer();
        
        virtual void set_mode(arac::structure::Component::Mode mode);
        
        arac::common::Buffer& state();
        arac::common::Buffer& state_error();
        

    protected:
        
        virtual void _forward();
        virtual void _backward();
        virtual void expand();
        
        void fill_internal_state();
        void retrieve_internal_state();
        void fill_internal_input();
        void retrieve_internal_output();
        void fill_internal_outerror();
        void retrieve_internal_inerror();
        void fill_internal_state_error();
        void retrieve_internal_state_error();
        
        // wrapped MdlstmLayer
        MdlstmLayer _mdlstm;

        arac::common::Buffer* _state_p;
        arac::common::Buffer* _state_error_p;
        
};

inline
arac::common::Buffer& LstmLayer::state()
{
    return *_state_p;
}


inline
arac::common::Buffer& LstmLayer::state_error()
{
    return *_state_error_p;
}




}
}
}


#endif