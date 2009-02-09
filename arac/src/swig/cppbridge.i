%module cppbridge
%{
#define SWIG_FILE_WITH_INIT
    
#include <iostream>

#include <numpy/arrayobject.h>

#include "../cpp/arac.h"


using namespace arac::common;
using namespace arac::structure;
using namespace arac::structure::connections;
using namespace arac::structure::modules;
using namespace arac::structure::networks;


void init_buffer(Buffer& buffer, double* content_p, int length, int rowsize)
{
    buffer.free_memory();
    buffer.set_rowsize(rowsize);
    for(int i = 0; i < length; i++)
    {
        buffer.append(content_p + i * rowsize);
    }
}


%}
%include "numpy.i"
%init %{
    import_array();
%}


%apply (double* INPLACE_ARRAY1, int DIM1) {(double* input_p, int inlength), 
                                           (double* output_p, int outlength)};
%apply (double* INPLACE_ARRAY1, int DIM1) {(double* outerror_p, int outlength), 
                                           (double* inerror_p, int inlength)};
%apply (double* INPLACE_ARRAY1, int DIM1) {(double* parameters_p, int parameter_size), 
                                           (double* derivatives_p, int derivative_size)};
%apply (double* INPLACE_ARRAY1, int DIM1) {(double* outerror_p, int outlength), 
                                           (double* inerror_p, int inlength)};
%apply (double* INPLACE_ARRAY2, int DIM1, int DIM2) {
    (double* content_p, int length, int rowsize)};





%nodefaultctor Buffer;
class Buffer
{
    // Add the given pointer as a row.
    void append(double* row);
};


%extend Buffer 
{
    void append(double* row_p, int this_size)
    {
        if (this_size != $self->rowsize()) {
            PyErr_Format(PyExc_ValueError, "Row has wrong length: (%d,%d) given",
                         this_size, $self->rowsize());
            return;
        }
        $self->free_memory();
        $self->append(row_p);
    }
};


class Component 
{
    public: 
        enum Mode 
        {
            Simple = 0,
            ErrorAgnostic = 1,
            Sequential = 2,
            
            SequentialErrorAgnostic = 3,
        };
        virtual Component();
        virtual ~Component();
        virtual void forward();
        virtual void backward();
        virtual void dry_forward();
        virtual void dry_backward();
        virtual void set_mode(Mode mode);
        virtual void clear();
        Mode get_mode();
        bool sequential();
        int timestep();
        int sequencelength();
        bool error_agnostic();
    protected:
        virtual void _forward() = 0;
        virtual void _backward() = 0;

};


class Module : public Component
{
    public:
        Module();
        Module(int insize, int outsize);
        virtual ~Module();
        virtual void forward();
        void add_to_input(double* addend_p);
        void add_to_outerror(double* addend_p);
        virtual void clear();
        arac::common::Buffer& input();
        arac::common::Buffer& output();
        arac::common::Buffer& inerror();
        arac::common::Buffer& outerror();
        int insize();
        int outsize();
        bool last_timestep();
};


%extend Module 
{
    void init_input(double* content_p, int length, int rowsize)
    {
        init_buffer($self->input(), content_p, length, rowsize);
    }

    void init_output(double* content_p, int length, int rowsize)
    {
        init_buffer($self->output(), content_p, length, rowsize);
    }
    
    void init_inerror(double* content_p, int length, int rowsize)
    {
        init_buffer($self->inerror(), content_p, length, rowsize);
    }

    void init_outerror(double* content_p, int length, int rowsize)
    {
        init_buffer($self->outerror(), content_p, length, rowsize);
    }
}


class Parametrized 
{
    public: 
        Parametrized();
        Parametrized(int size);
        virtual ~Parametrized();
        double* get_parameters() const;
        void set_parameters(double* parameters_p);
        double* get_derivatives() const;
        void set_derivatives(double* derivatives_p);
        void clear_derivatives();
};


class Connection : public Component
{
    public: 
        Connection(Module* incoming, Module* outgoing,
                   int incomingstart, int incomingstop, 
                   int outgoingstart, int outgoingstop);
        Connection(Module* incoming, Module* outgoing);
        virtual ~Connection();
        
        void set_incomingstart(int n);
        void set_incomingstop(int n);
        void set_outgoingstart(int n);
        void set_outgoingstop(int n);
        
        int get_incomingstart();
        int get_incomingstop();
        int get_outgoingstart();
        int get_outgoingstop();
        
        void set_recurrent(int recurrent);
        int get_recurrent();
        
        Module* incoming();
        Module* outgoing();
};


%feature("notabstract") IdentityConnection;
class IdentityConnection : public Connection 
{
    public:
        IdentityConnection(Module* incoming_p, Module* outgoing_p);
        IdentityConnection(Module* incoming_p, Module* outgoing_p,
                           int incomingstart, int incomingstop, 
                           int outgoingstart, int outgoingstop);
        virtual ~IdentityConnection();
};


%feature("notabstract") Bias;
class Bias : public Module
{
    public:
        Bias();
        virtual ~Bias();
};


%feature("notabstract") GateLayer;
class GateLayer : public Module
{
    public:
        GateLayer(int size);
        virtual ~GateLayer();
};


%feature("notabstract") LinearLayer;
class LinearLayer : public Module
{
    public:
        LinearLayer(int size);
        virtual ~LinearLayer();
};


%feature("notabstract") LstmLayer;
class LstmLayer : public Module
{
    public:
        LstmLayer(int size);
        virtual ~LstmLayer();
};


%extend LstmLayer {

    void init_state(double* content_p, int length, int rowsize)
    {
        init_buffer($self->state(), content_p, length, rowsize);
    }

    void init_state_error(double* content_p, int length, int rowsize)
    {
        init_buffer($self->state_error(), content_p, length, rowsize);
    }
}


%feature("notabstract") MdlstmLayer;
class MdlstmLayer : public Module
{
    public:
        MdlstmLayer(int timedim, int size);
        virtual ~MdlstmLayer();
};


%extend MdlstmLayer {

    void init_input_squashed(double* content_p, int length, int rowsize)
    {
        init_buffer($self->input_squashed(), content_p, length, rowsize);
    }
    
    void init_input_gate_squashed(double* content_p, int length, int rowsize)
    {
        init_buffer($self->input_gate_squashed(), content_p, length, rowsize);
    }
    
    void init_input_gate_unsquashed(double* content_p, int length, int rowsize)
    {
        init_buffer($self->input_gate_unsquashed(), content_p, length, rowsize);
    }
    
    void init_output_gate_squashed(double* content_p, int length, int rowsize)
    {
        init_buffer($self->output_gate_squashed(), content_p, length, rowsize);
    }
    
    void init_output_gate_unsquashed(double* content_p, int length, int rowsize)
    {
        init_buffer($self->output_gate_unsquashed(), content_p, length, rowsize);
    }
        
    void init_forget_gate_unsquashed(double* content_p, int length, int rowsize)
    {
        init_buffer($self->forget_gate_unsquashed(), content_p, length, rowsize);
    }
    
    void init_forget_gate_squashed(double* content_p, int length, int rowsize)
    {
        init_buffer($self->forget_gate_squashed(), content_p, length, rowsize);
    }
}



%feature("notabstract") PartialSoftmaxLayer;
class PartialSoftmaxLayer : public Module
{
    public:
        PartialSoftmaxLayer(int size, int slicelength);
        virtual ~PartialSoftmaxLayer();
};


%feature("notabstract") SigmoidLayer;
class SigmoidLayer : public Module
{
    public:
        SigmoidLayer(int size);
        virtual ~LinearLayer();
};


%feature("notabstract") SoftmaxLayer;
class SoftmaxLayer : public Module
{
    public:
        SoftmaxLayer(int size);
        virtual ~SoftmaxLayer();
};


%feature("notabstract") TanhLayer;
class TanhLayer : public Module
{
    public:
        TanhLayer(int size);
        virtual ~TanhLayer();
};


class BaseNetwork : public Module
{
    
    public:
        BaseNetwork();
        virtual ~BaseNetwork();
    
        virtual void activate(double* input_p, double* output_p);
        virtual void back_activate(double* outerror_p, double* inerror_p);
        virtual void forward();
        
    protected:
        virtual void sort() = 0;
};


%feature("notabstract") FullConnection;
class FullConnection : public Connection, public Parametrized
{
    public: 
        
        FullConnection(Module* incoming_p, Module* outgoing_p);
        FullConnection(Module* incoming_p, Module* outgoing_p,
                       int incomingstart, int incomingstop, 
                       int outgoingstart, int outgoingstop);
        FullConnection(Module* incoming_p, Module* outgoing_p,
                       double* parameters_p, double* derivatives_p,
                       int incomingstart, int incomingstop, 
                       int outgoingstart, int outgoingstop);
        virtual ~FullConnection();
        
        %extend 
        {
            FullConnection(Module* incoming_p, Module* outgoing_p,
                           double* parameters_p, int parameter_size,
                           double* derivatives_p, int derivative_size,
                           int incomingstart, int incomingstop, 
                           int outgoingstart, int outgoingstop)
    
            {
                int required_size = \
                    (incomingstop - incomingstart) * (outgoingstop - outgoingstart);
                if (parameter_size != required_size)
                {
                    PyErr_Format(PyExc_ValueError, 
                         "Parameters have wrong size: should be %d instead of %d.",
                         required_size, parameter_size);
                    return 0;
                }
                if (derivative_size != required_size)
                {
                    PyErr_Format(PyExc_ValueError, 
                         "Derivatives have wrong size: should be %d instead of %d.",
                         required_size, parameter_size);
                    return 0;
                }
        
                FullConnection* con = new FullConnection(incoming_p, outgoing_p, 
                                                         parameters_p, derivatives_p,
                                                         incomingstart, incomingstop,
                                                         outgoingstart, outgoingstop);
                return con;
            }
        }
};    


%feature("notabstract") LinearConnection;
class LinearConnection : public Connection, public Parametrized
{
    public: 
        
        LinearConnection(Module* incoming_p, Module* outgoing_p);
        LinearConnection(Module* incoming_p, Module* outgoing_p,
                       int incomingstart, int incomingstop, 
                       int outgoingstart, int outgoingstop);
        LinearConnection(Module* incoming_p, Module* outgoing_p,
                       double* parameters_p, double* derivatives_p,
                       int incomingstart, int incomingstop, 
                       int outgoingstart, int outgoingstop);
        virtual ~LinearConnection();
        
        %extend
        {
            LinearConnection(Module* incoming_p, Module* outgoing_p,
                           double* parameters_p, int parameter_size,
                           double* derivatives_p, int derivative_size,
                           int incomingstart, int incomingstop, 
                           int outgoingstart, int outgoingstop)
            {
                int required_size = incomingstop - incomingstart;
                if (outgoingstop - outgoingstart != required_size)
                {
                    PyErr_Format(PyExc_ValueError, 
                         "Slice sizes are not equal. (%d, %d).",
                         required_size, outgoingstop - outgoingstart);
                    return 0;
                }
                if (parameter_size != required_size)
                {
                    PyErr_Format(PyExc_ValueError, 
                         "Parameters have wrong size: should be %d instead of %d.",
                         required_size, parameter_size);
                    return 0;
                }
                if (derivative_size != incomingstop - incomingstart)
                {
                    PyErr_Format(PyExc_ValueError, 
                         "Derivatives have wrong size: should be %d instead of %d.",
                         required_size, parameter_size);
                    return 0;
                }
                
                LinearConnection* con = new LinearConnection(
                                            incoming_p, outgoing_p, 
                                            parameters_p, derivatives_p,
                                            incomingstart, incomingstop,
                                            outgoingstart, outgoingstop);
                return con;
            }
        }
};    


%feature("notabstract") Network;
class Network : public BaseNetwork
{
    public: 
        
        enum ModuleType {
            Simple = 0,
            InputModule = 1,
            OutputModule = 2,
            InputOutputModule = 3
        };
        Network();
        virtual ~Network();
        virtual void clear();
        void add_module(Module* module_p, ModuleType type=Simple);
        void add_connection(Connection* con_p);
};        
        
        
%extend Network
{
    void activate(double* input_p, int inlength, 
                  double* output_p, int outlength)
    {
        // TODO: check bounds of in and output
        // if (inlength != $self->insize()) {
        //     PyErr_Format(PyExc_ValueError, 
        //                  "Input has wrong size: %d instead of %d",
        //                  inlength, $self->insize());
        //     return;
        // }
        // if (outlength != $self->outsize()) {
        //     PyErr_Format(PyExc_ValueError, 
        //                  "Output has wrong size: %d instead of %d",
        //                  outlength, $self->outsize());
        //     return;
        // }

        $self->activate(input_p, output_p);
    }

    virtual void back_activate(double* outerror_p, int outlength, 
                               double* inerror_p, int inlength)
    {
        if (inlength != $self->insize() or outlength != $self->outsize()) {
            PyErr_Format(PyExc_ValueError, "Arrays of lengths (%d,%d) given",
                         inlength, outlength);
            return;
        }
        $self->back_activate(outerror_p, inerror_p);
    }
};