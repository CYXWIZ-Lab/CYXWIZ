#pragma once

// A value bound to a ? / $1 of a session query (session_query_engine.h).
// On its own so code that only builds a WHERE clause (the dashboard model)
// needs neither DuckDB nor Arrow.

#include <cstdint>
#include <string>
#include <utility>

namespace cyxwiz {

struct QueryParam {
    enum class Type { Null, Bool, Int, Double, Text };
    Type type = Type::Null;
    bool b = false;
    int64_t i = 0;
    double d = 0;
    std::string s;
    static QueryParam Of(bool v) { QueryParam p; p.type = Type::Bool; p.b = v; return p; }
    static QueryParam Of(int64_t v) { QueryParam p; p.type = Type::Int; p.i = v; return p; }
    static QueryParam Of(double v) { QueryParam p; p.type = Type::Double; p.d = v; return p; }
    static QueryParam Of(std::string v) { QueryParam p; p.type = Type::Text; p.s = std::move(v); return p; }
};

}  // namespace cyxwiz
