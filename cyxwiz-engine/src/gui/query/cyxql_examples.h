#pragma once

#include <vector>

namespace cyxwiz::query {

// The example queries the Query Console offers. They live beside the parser
// so a test can parse every one: an example that fails to parse is a bug.
struct Example {
    const char* name;
    const char* query;
    const char* description;
};

const std::vector<Example>& Examples();

}  // namespace cyxwiz::query
