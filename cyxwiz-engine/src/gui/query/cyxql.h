#pragma once

/**
 * CyxQL - Cypher-inspired Query Language for Node Editor
 *
 * CyxQL provides a powerful way to query and manipulate the visual node graph
 * using a syntax similar to Neo4j's Cypher query language.
 *
 * Example queries:
 *
 *   // Find all Dense layers
 *   MATCH (n:Dense) RETURN n
 *
 *   // Find Dense layers with > 256 units
 *   MATCH (n:Dense) WHERE n.units > 256 RETURN n.name, n.units
 *
 *   // Find path from input to output
 *   MATCH (a:DatasetInput)-[*]->(b:Output) RETURN a, b
 *
 *   // Find residual connections
 *   MATCH (a)-[r:ResidualSkip]->(b) RETURN a.name, b.name
 *
 *   // Find Conv-BN-ReLU sequences
 *   MATCH (c:Conv2D)-[:TensorFlow]->(bn:BatchNorm)-[:TensorFlow]->(r:ReLU)
 *   RETURN c.name, bn.name, r.name
 *
 *   // Count layers by type
 *   MATCH (n) RETURN type(n), count(n)
 *
 * Usage:
 *   #include "query/cyxql.h"
 *
 *   auto result = cyxwiz::query::runQuery(nodeEditor, "MATCH (n:Dense) RETURN n");
 *   if (result.success) {
 *       std::cout << result.toTable() << std::endl;
 *   } else {
 *       std::cerr << result.error << std::endl;
 *   }
 */

#include "cyxql_types.h"
#include "cyxql_lexer.h"
#include "cyxql_parser.h"
#include "cyxql_executor.h"
