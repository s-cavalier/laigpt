#include "Tensor.hpp"
#include <xtensor/xio.hpp>

ag::Tensor::Node& ag::Tensor::operator*() {
    return node.get();
}

const ag::Tensor::Node& ag::Tensor::operator*() const {
    return node.get();
}

ag::Tensor::Node* ag::Tensor::operator->() {
    return &node.get();
}

const ag::Tensor::Node* ag::Tensor::operator->() const {
    return &node.get();
}


ag::Tensor::Tensor( std::initializer_list<float> init_values, TensorNetwork& network ) :
    net( network ),
    node( *network.arena.allocate<Node>() ) {

    new ( &node.get() ) Node(network);
    node.get().values = init_values;

}

ag::Tensor::Tensor( std::initializer_list<std::initializer_list<float>> init_values, TensorNetwork& network ) :
    net( network ),
    node( *network.arena.allocate<Node>() ) {

    new ( &node.get() ) Node(network);
    node.get().values = init_values;

}

ag::Tensor::Tensor( const arena_f32arr& array, TensorNetwork& network ) :
    net( network ),
    node( *network.arena.allocate<Node>() ) {

        new ( &node.get() ) Node(array, network);

    }

ag::Tensor::Tensor( arena_f32arr&& array, TensorNetwork& network ) :
    net( network ),
    node( *network.arena.allocate<Node>() ) {

        new ( &node.get() ) Node(std::move(array), network);

    }

std::string ag::Tensor::Node::serialize_graph(bool include_values) const {
    std::ostringstream out;
    std::unordered_map<const Node*, int> idmap;
    int next_id = 0;

    std::function<void(const Node*)> dfs = [&](const Node* n) {
        if (idmap.count(n)) return;  
        int id = next_id++;
        idmap[n] = id;

        for (const ref<Node>& p : n->parents) {
            dfs(&p.get());
        }
    };

    dfs(this);

    std::vector<const Node*> ordered(idmap.size());
    for (auto &kv : idmap) ordered[kv.second] = kv.first;

    for (int id = 0; id < (int)ordered.size(); id++) {
        const Node* n = ordered[id];

        out << "Node " << id << "\n";

        out << "  parents -> [";
        for (size_t i = 0; i < n->parents.size(); i++) {
            const Node* p = &n->parents[i].get();
            int pid = idmap.at(p);
            if (i) out << ", ";
            out << pid;
        }
        out << "]\n";

        if (include_values) {
            out << "  values:\n" << values << '\n';
        } else {
            out << "  values: <omitted>\n";
        }

        out << "\n";
    }

    return out.str();
}


ag::arena_f32arr& ag::Tensor::values() {
    return node.get().values;
}

const ag::arena_f32arr& ag::Tensor::values() const {
    return node.get().values;
}

ag::Tensor ag::Tensor::get_ref() {
    return Tensor(*this);
}

ag::Tensor ag::Tensor::get_clone() const {
    Tensor tmp(*this);
    
    Node* new_node = net.get().arena.allocate<Node>();

    new (new_node) Node( node.get().values, net );

    tmp.node = *new_node;

    return tmp;
}

std::string ag::Tensor::serialize_graph(bool include_values) const {
    return node.get().serialize_graph(include_values);
}