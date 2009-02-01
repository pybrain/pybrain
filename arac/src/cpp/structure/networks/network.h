// Part of Arac Neural Network Composition Library.
// (c) 2008 by Justin S Bayer, <bayer.justin@googlemail.com>


#ifndef Arac_STRUCTURE_NETWORKS_NETWORK_INCLUDED
#define Arac_STRUCTURE_NETWORKS_NETWORK_INCLUDED


#include <vector>
#include <map>

#include "../component.h"
#include "../connections/connection.h"
#include "../modules/module.h"
#include "basenetwork.h"


namespace arac {
namespace structure {
namespace networks {
    
    
using namespace arac::structure::modules;
using namespace arac::structure::connections;
using arac::structure::Component;


// TODO: document.

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

    protected:
        
        virtual void _forward();
        virtual void _backward();

        // Fill count with the amount of incoming edges for every module.
        void incoming_count(std::map<Module*, int>& count);
        
        virtual void sort();
        
        void init_buffers();

        std::vector<Module*> _inmodules;
        std::vector<Module*> _outmodules;
        std::vector<Component*> _components_sorted;
        std::map<Module*, ModuleType> _modules;
        std::vector<Connection*> _connections;
        std::map<Module*, std::vector<Connection*> > _outgoing_connections;
    };
 
    
}
}
}


#endif