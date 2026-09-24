#pragma once

#include <nlohmann/json.hpp>

#include <string>

namespace gui {
enum class NodeType;
}

namespace cyxwiz {

// Preparation Recipe, contract version 1 (TOFIX101 package B, data_studio_design.md).
//
// A recipe is a saved Subgraph whose wrapper node carries
//   recipe_role = "preparation_recipe"
//   recipe_contract_version = "1"
// It has exactly one Dataset input and one Dataset output, and every internal
// node is a stateless single-table preparation step. Sources, exports, splits,
// fitted transforms and models are pipeline stages and stay outside.
//
// Recipes run through the ordinary pipeline: LowerPreparationRecipes replaces
// each recipe wrapper by its steps and rewires the wrapper's links to the bound
// internal pins, so Preview, Run and pipeline execution share the canonical
// operators. A Subgraph without the recipe role is a visual group and is never
// silently made executable.
inline constexpr const char* kRecipeRoleParameter = "recipe_role";
inline constexpr const char* kPreparationRecipeRole = "preparation_recipe";
inline constexpr const char* kRecipeContractParameter = "recipe_contract_version";
inline constexpr const char* kPreparationRecipeContractVersion = "1";

// True when a node of this type may be a step inside a recipe.
bool IsPreparationRecipeStepType(gui::NodeType type);

// Why this subgraph record cannot be a Preparation Recipe, or empty when it can.
// `wrapper` is the wrapper node and `record` its entry in the document's
// "subgraphs" array, both in the saved .cyxgraph form. The recipe role itself
// is not checked here, so the editor can validate before marking.
std::string PreparationRecipeRejection(const nlohmann::json& wrapper,
                                       const nlohmann::json& record);

// Executable view of a saved graph document (the form written by the node
// editor: numeric node types, links with from_node/to_node and pin indexes,
// optional "subgraphs"). Returns {"nodes", "links", "recipe_steps"} where every
// recipe wrapper is replaced by its steps; "recipe_steps" maps each step id
// (as a string) to its recipe wrapper id for progress attribution. Node types
// stay numeric. Throws std::runtime_error with an actionable reason when a
// subgraph is not a valid recipe or the document is inconsistent.
nlohmann::json LowerPreparationRecipes(const nlohmann::json& document);

}  // namespace cyxwiz
