// Part of Arac Neural Network Composition Library.
// (c) 2008 by Justin S Bayer, <bayer.justin@googlemail.com>

#include <cstring>
#include <iostream>

#include "mdlstm.h"
#include "../../common/common.h"


using arac::common::Buffer;
using arac::common::sigmoid;
using arac::common::sigmoidprime;
using arac::common::tanhprime;
using arac::common::tanh_;
using arac::structure::modules::MdlstmLayer;


MdlstmLayer::MdlstmLayer(int timedim, int size) :
    Module((3 + 2 * timedim) * size, 2 * size),
    _timedim(timedim),
    _input_squashed(size),
    _input_gate_squashed(size),
    _input_gate_unsquashed(size),
    _output_gate_squashed(size),
    _output_gate_unsquashed(size),
    _forget_gate_squashed(size * timedim),
    _forget_gate_unsquashed(size * timedim)
{
    _inter_input_p = new double[size];
    _input_state_p = new double[size * _timedim];
    _output_error_p = new double[size];
    _output_state_p = new double[size * _timedim];
    _output_state_error_p = new double[size * _timedim];
    _output_gate_error_p = new double[size];
    _forget_gate_error_p = new double[size * _timedim];
    _input_gate_error_p = new double[size];
    _input_error_p = new double[size];
    _input_state_error_p = new double[size * _timedim];
    _state_error_p = new double[size * _timedim];
}


MdlstmLayer::~MdlstmLayer() 
{
    delete[] _inter_input_p;
    delete[] _input_state_p;
    delete[] _output_error_p;
    delete[] _output_state_p;
    delete[] _output_state_error_p;
    delete[] _output_gate_error_p;
    delete[] _forget_gate_error_p;
    delete[] _input_gate_error_p;
    delete[] _input_error_p;
    delete[] _input_state_error_p;
    delete[] _state_error_p;
}


void
MdlstmLayer::clear_intermediates()
{
    int size = _outsize / 2;
    memset(_inter_input_p, 0, sizeof(double) * size);
    memset(_input_state_p, 0, sizeof(double) * size);
    memset(_output_error_p, 0, sizeof(double) * size);
    memset(_output_state_p, 0, sizeof(double) * size);
    memset(_output_state_error_p, 0, sizeof(double) * size);
    memset(_output_gate_error_p, 0, sizeof(double) * size);
    memset(_forget_gate_error_p, 0, sizeof(double) * size * _timedim);
    memset(_input_gate_error_p, 0, sizeof(double) * size);
    memset(_input_error_p, 0, sizeof(double) * size);
    memset(_input_state_error_p, 0, sizeof(double) * size * _timedim);
    memset(_state_error_p, 0, sizeof(double) * size);
}


void
MdlstmLayer::expand()
{
    _input_squashed.expand();
    _input_gate_squashed.expand();
    _input_gate_unsquashed.expand();
    _output_gate_squashed.expand();
    _output_gate_unsquashed.expand();
    _forget_gate_unsquashed.expand();
    _forget_gate_squashed.expand();
    Module::expand();
}


void
MdlstmLayer::_forward()
{
    clear_intermediates();
    int size = _outsize / 2;
    
    double (*gate_squasher) (double) = sigmoid;
    double (*cell_squasher) (double) = tanh_;
    double (*output_squasher) (double) = tanh_;
    
    // Split the whole input into the right chunks
    double* inputbuffer_p = input()[_timestep];
    int i = 0;
    for (int j = 0; j < size; j++, i++)
    {
        input_gate_unsquashed()[_timestep][j] = inputbuffer_p[i];
    }

    for (int j = 0; j < size * _timedim; j++, i++)
    {
        forget_gate_unsquashed()[_timestep][j] = inputbuffer_p[i];
    }
    
    for (int j = 0; j < size; j++, i++)
    {
        input_squashed()[_timestep][j] = cell_squasher(inputbuffer_p[i]);
    }

    for (int j = 0; j < size; j++, i++)
    {
        output_gate_unsquashed()[_timestep][j] = inputbuffer_p[i];
    }
    
    for (int j = 0; j < size * _timedim ; j++, i++)
    {
        _input_state_p[j] = inputbuffer_p[i];
    }
    
    // TODO: include peepholes
    // Change the ingate values with respect to peepholes, if we have peephole 
    // weights
    // if (mdlstml_p->peephole_input_weights.contents_p)
    // {
    //     for (int i = 0; i < _timedim; i++)
    //     {
    //         for (int j = 0; j < size; j++)
    //         {
    //             _input_gate_unsquashed.current()[j] += \
    //                 mdlstml_p->peephole_input_weights.contents_p[j] \
    //                 * inputstate_p[i * size + j];
    //         }
    //     }
    // }
    // 
    // if (mdlstml_p->peephole_forget_weights.contents_p)
    // {
    //     for (int i = 0; i < size * _timedim; i++)
    //     {
    //         _forget_gate_unsquashed.current()[i] += \
    //             mdlstml_p->peephole_forget_weights.contents_p[i] * inputstate_p[i];
    //     }
    // }
    
    // Squash the input gates and forget gates.
    for (int i = 0; i < size; i++)
    {
        input_gate_squashed()[timestep()][i] = \
            gate_squasher(_input_gate_unsquashed[timestep()][i]);
        forget_gate_squashed()[timestep()][i] = \
            gate_squasher(forget_gate_unsquashed()[timestep()][i]);
    }
    
    // Calculate the current cell state.
    for (int i = 0; i < size; i++)
    {
        _output_state_p[i] = input_gate_squashed()[timestep()][i] \
            * input_squashed()[timestep()][i];
    }

    // Apply the forget gates.
    for (int i = 0; i < _timedim; i++)
    {
        for (int j = 0; j < size; j++)
        {
            int index = size * i + j;
            _output_state_p[j] += \
                forget_gate_squashed()[timestep()][index] * _input_state_p[index];
            
        }
    }

    // TODO: include peepholes
    // // Change the outgate values if peephole weights are set.
    // if (mdlstml_p->peephole_output_weights.contents_p)
    // {
    //     for (int i = 0; i < size; i++)
    //     {
    //         _output_gate_unsquashed.current()[i] += \
    //             mdlstml_p->peephole_output_weights.contents_p[i] * outputstate_p[i];
    //     }
    // }
    
    // Squash the output gates.
    for (int i = 0; i < size; i++)
    {
        output_gate_squashed()[timestep()][i] = \
            gate_squasher(output_gate_unsquashed()[timestep()][i]);
    }
    // Save the results to the outputbuffer.
    _outputbuffer_p = output()[timestep()];
    for (int i = 0; i < size; i++)
    {
        _outputbuffer_p[i] = output_gate_squashed()[timestep()][i] *
            output_squasher(_output_state_p[i]);
        _outputbuffer_p[i + size] = _output_state_p[i];
    }
}


void
MdlstmLayer::_backward()
{
    clear_intermediates();
    int size = _outsize / 2;
    int this_timestep = timestep() - 1;
    
    double (*cell_squasher) (double) = tanh_;
    // double (*output_squasher) (double) = tanh_;
    double (*gate_squasher_prime) (double) = sigmoidprime;
    double (*cell_squasher_prime) (double) = tanhprime;
    // double (*output_squasher_prime) (double) = tanhprime;

    // Split the whole input into the right chunks
    double* inputbuffer_p = input()[this_timestep];
    double* outputstate_p = output()[this_timestep] + size;
    double* nextstateerror_p = outerror()[this_timestep] + size;

    int i, j;
    for (i = size + size * _timedim, j = 0; j < size; j++, i++)
    {
        _inter_input_p[j] = inputbuffer_p[i];
    }

    for (i = 3 * size + size * _timedim, j = 0; j < size * _timedim ; j++, i++)
    {
        _input_state_p[j] = inputbuffer_p[i];
    }

    // Shortcut
    double* output_error_buffer_p = outerror()[this_timestep];
    
    // Splitting the errorbuffer into two parts.
    for (int i = 0; i < size; i++)
    {
        _output_error_p[i] = output_error_buffer_p[i];
    }

    for (int i = 0; i < size; i++)
    {
        _output_state_error_p[i] = output_error_buffer_p[i + size];
    }
    
    // Calculate the outgate error.
    for (int i = 0; i < size; i++)
    {
        _output_gate_error_p[i] = \
            gate_squasher_prime(_output_gate_unsquashed[this_timestep][i]) \
            * _output_error_p[i] \
            * cell_squasher(outputstate_p[i]);
    }

    // This is an intermediate for calculations.
    for (int i = 0; i < size; i++)
    {
        _state_error_p[i] = \
            _output_error_p[i] \
            * _output_gate_squashed[this_timestep][i] \
            * tanhprime(outputstate_p[i]);  // FIXME: use func-pointer here.
        _state_error_p[i] += nextstateerror_p[i];
    }
    
    // TODO: integrate peepholes
    // if (mdlstml_p->peephole_output_weights.contents_p)
    // {
    //     for (int i = 0; i < size; i++)
    //     {
    //         _state_error_p[i] += \
    //             _output_gate_error_p[i] \
    //             * mdlstml_p->peephole_output_weights.contents_p[i];
    //     }
    // }
    
    // Calculate cell errors.
    for (int i = 0; i < size; i++)
    {
        _input_error_p[i] = \
            _input_gate_squashed[this_timestep][i] \
            * cell_squasher_prime(_inter_input_p[i]) \
            * _state_error_p[i];
    }
    
    // Calc forget gate errors.
    for (int i = 0; i < _timedim; i++)
    {
        for (int j = 0; j < size; j++)
        {
            _forget_gate_error_p[i * size + j] = \
                gate_squasher_prime(_forget_gate_unsquashed[this_timestep][i * size + j]) \
                * _state_error_p[j] \
                * _input_state_p[i * size + j];
        }
    }
    
    // FIXME: strangely, tests fail if this loop is removed.
    for (int i = 0; i < size * _timedim; i++)
    {
    }

    for (int i = 0; i < size; i++)
    {
        _input_gate_error_p[i] = \
            gate_squasher_prime(_input_gate_unsquashed[this_timestep][i]) \
            * _input_squashed[this_timestep][i] \
            * _state_error_p[i];
    }
    
    // FIXME: strangely, tests fail if this loop is removed.
    for (int i = 0; i < size; i++)
    {
    }

    // TODO: integrate peepholes
    // if (mdlstml_p->peephole_output_weights.contents_p)
    // {
    //     for (int i = 0; i < size; i++)
    //     {
    //         mdlstml_p->peephole_output_weights.error_p[i] += \
    //             output_gate_error_p[i] \
    //             * outputstate_p[i];
    //     }
    //     for (int i = 0; i < size * _timedim; i++)
    //     {
    //         mdlstml_p->peephole_forget_weights.error_p[i] += \
    //             _forget_gate_error_p[i] \
    //             * _input_state_p[i];
    //     }
    //     for (int i = 0; i < _timedim; i++)
    //     {
    //         for (int j = 0; j < size; j++)
    //         {
    //             mdlstml_p->peephole_input_weights.error_p[j] += \
    //                 _input_gate_error_p[j] \
    //                 * _input_state_p[i * size + j];
    //         }
    //     }
    // }
    
    for (int i = 0; i < _timedim; i++)
    {
        for (int j = 0; j < size; j++)
        {
            _input_state_error_p[i * size + j] += \
                _state_error_p[j] * _forget_gate_squashed[this_timestep][i * size + j];
            // TODO: integrate peepholes
            // if (mdlstml_p->peephole_output_weights.contents_p)
            // {
            //     _input_state_error_p[i * size + j] += \
            //         _input_gate_error_p[j] \
            //         * mdlstml_p->peephole_input_weights.contents_p[j];
            //     _input_state_error_p[i * size + j] += \
            //         forget_gate_error_p[i * size + j] \
            //         * mdlstml_p->peephole_forget_weights.contents_p[i * size + j];
            // }
        }
    }
    
    // FIXME: strangely, tests fail if this loop is removed.
    for (int i = 0; i < size * _timedim; i++)
    {
    }
    
    double* inerror_p = inerror()[this_timestep];
    i = 0;
    for (int j = 0; j < size; j++, i++)
    {
        inerror_p[i] = _input_gate_error_p[j];
    }

    for (int j = 0; j < size * _timedim; j++, i++)
    {
        inerror_p[i] = _forget_gate_error_p[j];
    }
    
    for (int j = 0; j < size; j++, i++)
    {
        inerror_p[i] = _input_error_p[j];
    }

    for (int j = 0; j < size; j++, i++)
    {
        inerror_p[i] = _output_gate_error_p[j];
    }
    
    for (int j = 0; j < size * _timedim ; j++, i++)
    {
        inerror_p[i] = _input_state_error_p[j];
    }
}