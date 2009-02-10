// Part of Arac Neural Network Composition Library.
// (c) 2008 by Justin S Bayer, <bayer.justin@googlemail.com>


#ifndef Arac_STRUCTURE_PARAMETRIZED_INCLUDED
#define Arac_STRUCTURE_PARAMETRIZED_INCLUDED


namespace arac {
namespace structure {
    

// TODO: document.

class Parametrized 
{
    public: 
    
        Parametrized();
        Parametrized(int size);
        Parametrized(int size, double* parameters_p, double* derivatives_p);
        virtual ~Parametrized();
        
        double* get_parameters() const;
        void set_parameters(double* parameters_p);
        
        double* get_derivatives() const;
        void set_derivatives(double* derivatives_p);
        
        void clear_derivatives();
        
        bool parameters_owner();
        bool derivatives_owner();
        
    protected:
        
        double* _parameters_p;
        double* _derivatives_p;
        int _size;
        bool _parameters_owner;
        bool _derivatives_owner;
};
 
 
}
}

#endif