// Part of Arac Neural Network Composition Library.
// (c) 2008 by Justin S Bayer, <bayer.justin@googlemail.com>


#include <cstring>
#include <iostream>
#include <list>

#include "network.h"


using namespace std;
using arac::structure::networks::Network;


Network::Network() :
    BaseNetwork()
{
}


Network::~Network() 
{
}


// TODO: write a test for this.
void
Network::clear()
{
    BaseNetwork::clear();
    std::vector<Component*>::iterator comp_iter;
    for(comp_iter = _components_sorted.begin(); 
        comp_iter != _components_sorted.end();
        ++comp_iter)
    {
        (*comp_iter)->clear();
    }
}


void
Network::add_module(Module* module_p, Network::ModuleType type)
{
    _dirty = true;
    _modules.insert(std::make_pair(module_p, type));
    if (type & Network::InputModule)
    {
        _inmodules.push_back(module_p);
    }
    if (type & Network::OutputModule)
    {
        _outmodules.push_back(module_p);
    }
}


void
Network::add_connection(Connection* con_p)
{
    _dirty = true;
    _connections.push_back(con_p);
    _outgoing_connections[con_p->incoming()].push_back(con_p);
}


void
Network::_forward()
{
    // Make sure that all the modules are in the right order.
    if (_dirty)
    {
        sort();
    }
    // Copy the input into the inbuffers of the modules.
    std::vector<Module*>::iterator mod_iter;
    int n_timestep = timestep();
    double* input_p = input()[n_timestep];
    for(mod_iter = _inmodules.begin();
        mod_iter != _inmodules.end();
        mod_iter++)
    {
        Module* module_p = *mod_iter;
        int size = module_p->insize();
        int timestep = module_p->timestep();
        double* sink_p = module_p->input()[timestep];
        memcpy((void*) sink_p, (void*) input_p, size * sizeof(double));
        input_p += size;
    }
    // Forward the components in the right order.
    std::vector<Component*>::iterator comp_iter;
    for(comp_iter = _components_sorted.begin(); 
        comp_iter != _components_sorted.end();
        ++comp_iter)
    {
        (*comp_iter)->forward();
    }
    // Copy outputs into the outputbuffer.
    double* sink_p = output()[timestep()];
    for(mod_iter = _outmodules.begin(); mod_iter != _outmodules.end(); mod_iter++)
    {
        Module* module_p = *mod_iter;
        int size = module_p->outsize();
        // Modules have already been forwarded, thus substract 1 from timestep.
        double* source_p = module_p->output()[module_p->timestep() - 1];
        memcpy((void*) sink_p, (void*) source_p, size * sizeof(double));
        sink_p += size;
    }
}


void
Network::_backward()
{
    int this_timestep = timestep() - 1;
    std::vector<Module*>::iterator mod_iter;
    double* error_p = outerror()[this_timestep];
    for(mod_iter = _outmodules.begin(); 
        mod_iter != _outmodules.end(); 
        mod_iter++)
    {
        Module* module_p = *mod_iter;
        int size = module_p->outsize();
        int mod_timestep = module_p->timestep() - 1;
        if (!module_p->error_agnostic())
        {
            double* sink_p = module_p->outerror()[mod_timestep];
            memcpy((void*) sink_p, (void*) error_p, size * sizeof(double));
        }
        error_p += size;
    }
    
    std::vector<Component*>::reverse_iterator riter;
    for(riter = _components_sorted.rbegin(); 
        riter != _components_sorted.rend();
        riter++)
    {
        Component* comp_p = *riter;
        if (comp_p->error_agnostic())
        {
            continue;
        }
        comp_p->backward();
    }
    
    double* inerror_p = inerror()[this_timestep];
    for(mod_iter = _inmodules.begin();
        mod_iter != _inmodules.end();
        mod_iter++)
    {
        Module* module_p = *mod_iter;
        int size = module_p->insize();
        // No -1 on the timestep, since the modules have been backwarded 
        // already.
        int mod_timestep = module_p->timestep();
        double* source_p = module_p->inerror()[mod_timestep];
        memcpy((void*) inerror_p, (void*) source_p, size * sizeof(double));
        inerror_p += size;
    }
}


void
Network::incoming_count(std::map<Module*, int>& count)
{
    std::map<Module*, ModuleType>::iterator mod_iter;
    for(mod_iter = _modules.begin(); mod_iter != _modules.end(); mod_iter++)
    {
        pair<Module*, int> item(mod_iter->first, 0);
        count.insert(item);
    }
    
    std::vector<Connection*>::iterator con_iter;
    for(con_iter = _connections.begin(); 
        con_iter != _connections.end();
        con_iter++)
    {
        if ((*con_iter)->get_recurrent())
        {
            continue;
        }
        count[(*con_iter)->outgoing()]++;
    }
}


void
Network::sort()
{
    // Result lists.
    std::vector<Module*> sorted;
    std::vector<Module*>::iterator mod_iter;

    std::vector<Connection*>::iterator con_iter;
    
    // Mapping from nodes to the number of incoming connections.
    std::map<Module*, int> count;
    std::map<Module*, int>::iterator count_iter;
    incoming_count(count);

    // Make up a vector of all nodes with no incoming connections.
    std::vector<Module*> roots;
    for(count_iter = count.begin(); 
        count_iter != count.end(); 
        count_iter++)
    {
        Module* module_p = count_iter->first;
        int count = count_iter->second;
        if (count == 0)
        {
            roots.push_back(module_p);
        }
    }

    while (roots.size() > 0)
    {
        // Pop a root node and insert it into the list.
        Module* current = roots.back();
        roots.pop_back();
        sorted.push_back(current);
        for(con_iter = _outgoing_connections[current].begin(); 
            con_iter != _outgoing_connections[current].end();
            con_iter++)
         {
             if ((*con_iter)->get_recurrent())
             {
                 continue;
             }
             Module* other_p = (*con_iter)->outgoing();
             count[other_p]--;
             if (count[other_p] == 0)
             {
                 roots.push_back(other_p);
             }
         }
    }
    
    for(count_iter = count.begin(); count_iter != count.end(); count_iter++)
    {
        if (count_iter->second != 0)
        {
            // FIXME: error handling, graph has cycle.
            assert(0);
        }
    }

    // Fill the list of sorted components correctly.
    _components_sorted.clear();
    
    // First fill in the recurrent connections.
    // TODO: These should be sorted by recurrency.
    for (con_iter = _connections.begin();
         con_iter != _connections.end();
         con_iter++)
    {
        if (((*con_iter)->get_recurrent()))
        {
            _components_sorted.push_back(*con_iter);
        }
    }
    
    // Then fill in the rest in topological order.
    for(mod_iter = sorted.begin(); mod_iter != sorted.end(); mod_iter++)
    {
        _components_sorted.push_back(*mod_iter);
        for(con_iter = _outgoing_connections[*mod_iter].begin();
            con_iter != _outgoing_connections[*mod_iter].end();
            con_iter++)
        {
            if (((*con_iter)->get_recurrent()))
            {
                continue;
            }
            _components_sorted.push_back(*con_iter);
        }
    }
    init_buffers();
    _dirty = false;
}


void
Network::init_buffers()
{
    std::vector<Module*>::iterator mod_iter;

    _insize = 0;
    for(mod_iter = _inmodules.begin(); 
        mod_iter != _inmodules.end(); 
        mod_iter++)
    {
        _insize += (*mod_iter)->insize();
    }
    
    _outsize = 0;
    for(mod_iter = _outmodules.begin(); 
        mod_iter != _outmodules.end(); 
        mod_iter++)
    {
        _outsize += (*mod_iter)->outsize();
    }

    Module::init_buffers();
}
