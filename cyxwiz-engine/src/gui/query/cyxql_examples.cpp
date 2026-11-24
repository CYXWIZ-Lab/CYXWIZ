#include "cyxql_examples.h"

namespace cyxwiz::query {

const std::vector<Example>& Examples() {
    static const std::vector<Example> examples = {
        {
            "List all nodes",
            "MATCH (n) RETURN n",
            "Returns all nodes in the graph"
        },
        {
            "List all Dense layers",
            "MATCH (n:Dense) RETURN n.name, n.units",
            "Find all Dense layers and show their names and units"
        },
        {
            "List all Conv2D layers",
            "MATCH (n:Conv2D) RETURN n.name, n.filters, n.kernel_size",
            "Find all Conv2D layers with their parameters"
        },
        {
            "Count nodes by type",
            "MATCH (n) RETURN type(n) AS type, count(n) AS count",
            "Count how many nodes of each type exist"
        },
        {
            "Find input to output path",
            "MATCH (a:DataInput)-[*]->(b:Output) RETURN a.name, b.name",
            "Find paths from DataInput to Output nodes"
        },
        {
            "Find residual connections",
            "MATCH (a)-[r:ResidualSkip]->(b) RETURN a.name, b.name",
            "Find all residual/skip connections"
        },
        {
            "Find Conv-BN-ReLU sequences",
            "MATCH (c:Conv2D)-[:TensorFlow]->(bn:BatchNorm)-[:TensorFlow]->(r:ReLU)\nRETURN c.name, bn.name, r.name",
            "Find Conv2D -> BatchNorm -> ReLU patterns"
        },
        {
            "Find Dense layers > 256 units",
            "MATCH (n:Dense) WHERE n.units > 256 RETURN n.name, n.units",
            "Filter Dense layers by number of units"
        },
        {
            "Find Dropout layers",
            "MATCH (n:Dropout) RETURN n.name, n.rate",
            "List all Dropout layers with their rates"
        },
        {
            "Find attention layers",
            "MATCH (n:MultiHeadAttention) RETURN n.name, n.d_model, n.num_heads",
            "Find all Multi-Head Attention layers"
        },
        {
            "Get node properties",
            "MATCH (n) WHERE n.name CONTAINS \"Dense\" RETURN n.name, properties(n)",
            "Get all properties of nodes with 'Dense' in the name"
        },
        {
            "List optimizers",
            "MATCH (n) WHERE type(n) IN [\"SGD\", \"Adam\", \"AdamW\", \"RMSprop\", \"Adagrad\", \"NAdam\"]\n"
            "RETURN n.name, type(n) AS type, n.learning_rate",
            "Find all optimizer nodes"
        }
    };
    return examples;
}

}  // namespace cyxwiz::query
