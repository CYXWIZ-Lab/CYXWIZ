#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cyxwiz/optimization.h>

#include <string>
#include <vector>

TEST_CASE("Linear programming solves its declared simplex subset",
          "[optimization][linear_programming]") {
    const auto result = cyxwiz::Optimization::SolveLP(
        {2.0}, {{1.0}}, {4.0}, {"<="}, true);

    REQUIRE(result.success);
    REQUIRE(result.status == "Optimal");
    REQUIRE(result.solution.size() == 1);
    REQUIRE(result.solution[0] == Catch::Approx(4.0));
    REQUIRE(result.objective_value == Catch::Approx(8.0));

    const auto minimization = cyxwiz::Optimization::SolveLP(
        {-2.0}, {{1.0}}, {4.0}, {"<="}, false);
    REQUIRE(minimization.success);
    REQUIRE(minimization.solution[0] == Catch::Approx(4.0));
    REQUIRE(minimization.objective_value == Catch::Approx(-8.0));
}

TEST_CASE("Linear programming rejects unsupported constraint semantics",
          "[optimization][linear_programming][validation]") {
    const auto greater_equal = cyxwiz::Optimization::SolveLP(
        {1.0}, {{1.0}}, {2.0}, {">="}, false);
    REQUIRE_FALSE(greater_equal.success);
    REQUIRE(greater_equal.error_message.find("only <=") != std::string::npos);

    const auto mismatched_types = cyxwiz::Optimization::SolveLP(
        {1.0}, {{1.0}}, {2.0}, {}, false);
    REQUIRE_FALSE(mismatched_types.success);
    REQUIRE(mismatched_types.error_message.find("counts must match") !=
            std::string::npos);

    const auto negative_rhs = cyxwiz::Optimization::SolveLP(
        {1.0}, {{1.0}}, {-2.0}, {"<="}, false);
    REQUIRE_FALSE(negative_rhs.success);
    REQUIRE(negative_rhs.error_message.find("non-negative") !=
            std::string::npos);
}
