#pragma once

// The PPO (MLP 64-64) pattern built node by node, for the RL graph tests
// (TOFIX195 S4.3).

#include "../src/core/graph_node_factory.h"

#include <algorithm>
#include <cstdlib>
#include <string>
#include <vector>

namespace rl_graph_test {

struct Graph {
    std::vector<gui::MLNode> nodes;
    std::vector<gui::NodeLink> links;
    int next_node = 1, next_pin = 1, next_link = 1;

    int Add(gui::NodeType type, const std::string& name) {
        nodes.push_back(gui::CreateGraphNode(type, name, next_node, next_pin));
        return nodes.back().id;
    }
    gui::MLNode& At(int id) {
        for (auto& n : nodes)
            if (n.id == id) return n;
        std::abort();
    }
    void Link(int from, int to, size_t from_pin = 0, size_t to_pin = 0) {
        links.push_back({next_link++, from, At(from).outputs.at(from_pin).id, to, At(to).inputs.at(to_pin).id});
    }
    void Unlink(int from, int to) {
        std::erase_if(links, [&](const gui::NodeLink& l) { return l.from_node == from && l.to_node == to; });
    }
};

struct Ids {
    int env, p1, pa1, p2, pa2, phead, v1, va1, v2, va2, vhead, ppo;
};

inline Graph Pattern(Ids& id) {
    using gui::NodeType;
    Graph g;
    id.env = g.Add(NodeType::RLEnvironment, "Environment");
    id.p1 = g.Add(NodeType::Dense, "Policy Dense 1");
    id.pa1 = g.Add(NodeType::Tanh, "Policy Tanh 1");
    id.p2 = g.Add(NodeType::Dense, "Policy Dense 2");
    id.pa2 = g.Add(NodeType::Tanh, "Policy Tanh 2");
    id.phead = g.Add(NodeType::PolicyHead, "Policy Head");
    id.v1 = g.Add(NodeType::Dense, "Value Dense 1");
    id.va1 = g.Add(NodeType::Tanh, "Value Tanh 1");
    id.v2 = g.Add(NodeType::Dense, "Value Dense 2");
    id.va2 = g.Add(NodeType::Tanh, "Value Tanh 2");
    id.vhead = g.Add(NodeType::ValueHead, "Value Head");
    id.ppo = g.Add(NodeType::PPO, "PPO");
    for (int d : {id.p1, id.p2, id.v1, id.v2}) g.At(d).parameters["units"] = "64";
    g.Link(id.env, id.p1);
    g.Link(id.p1, id.pa1);
    g.Link(id.pa1, id.p2);
    g.Link(id.p2, id.pa2);
    g.Link(id.pa2, id.phead);
    g.Link(id.env, id.v1);
    g.Link(id.v1, id.va1);
    g.Link(id.va1, id.v2);
    g.Link(id.v2, id.va2);
    g.Link(id.va2, id.vhead);
    g.Link(id.phead, id.ppo, 0, 0);
    g.Link(id.env, id.ppo, 1, 1);
    g.Link(id.vhead, id.ppo, 0, 2);
    return g;
}

}  // namespace rl_graph_test
