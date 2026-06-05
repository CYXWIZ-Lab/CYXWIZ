#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cyxwiz/linear_algebra.h>
#include <vector>

namespace {

std::vector<std::vector<double>> ReconstructFromSVD(const cyxwiz::SVDResult& svd) {
    std::vector<std::vector<double>> result(
        static_cast<size_t>(svd.m),
        std::vector<double>(static_cast<size_t>(svd.n), 0.0));

    for (int i = 0; i < svd.m; ++i) {
        for (int j = 0; j < svd.n; ++j) {
            for (int k = 0; k < svd.k; ++k) {
                result[static_cast<size_t>(i)][static_cast<size_t>(j)] +=
                    svd.U[static_cast<size_t>(i)][static_cast<size_t>(k)] *
                    svd.S[static_cast<size_t>(k)] *
                    svd.Vt[static_cast<size_t>(k)][static_cast<size_t>(j)];
            }
        }
    }

    return result;
}

double ColumnDot(const std::vector<std::vector<double>>& matrix, size_t left, size_t right) {
    double result = 0.0;
    for (const auto& row : matrix) {
        result += row[left] * row[right];
    }
    return result;
}

}  // namespace

TEST_CASE("LinearAlgebra CPU SVD reconstructs a rectangular matrix", "[linalg][svd]") {
    const std::vector<std::vector<double>> matrix = {
        {3.0, 0.0},
        {0.0, 2.0},
        {0.0, 0.0},
    };

    cyxwiz::SVDResult svd = cyxwiz::LinearAlgebra::SVD(matrix, false);
    REQUIRE(svd.success);
    REQUIRE(svd.m == 3);
    REQUIRE(svd.n == 2);
    REQUIRE(svd.k == 2);
    REQUIRE(svd.S.size() == 2);
    REQUIRE(svd.S[0] == Catch::Approx(3.0).margin(1e-10));
    REQUIRE(svd.S[1] == Catch::Approx(2.0).margin(1e-10));
    REQUIRE(svd.U.size() == 3);
    REQUIRE(svd.U[0].size() == 2);
    REQUIRE(svd.Vt.size() == 2);
    REQUIRE(svd.Vt[0].size() == 2);

    const std::vector<std::vector<double>> reconstructed = ReconstructFromSVD(svd);
    for (size_t r = 0; r < matrix.size(); ++r) {
        for (size_t c = 0; c < matrix[r].size(); ++c) {
            REQUIRE(reconstructed[r][c] == Catch::Approx(matrix[r][c]).margin(1e-10));
        }
    }
}

TEST_CASE("LinearAlgebra CPU full SVD returns complete bases", "[linalg][svd]") {
    const std::vector<std::vector<double>> matrix = {
        {3.0, 0.0},
        {0.0, 2.0},
        {0.0, 0.0},
    };

    cyxwiz::SVDResult svd = cyxwiz::LinearAlgebra::SVD(matrix, true);
    REQUIRE(svd.success);
    REQUIRE(svd.m == 3);
    REQUIRE(svd.n == 2);
    REQUIRE(svd.k == 2);
    REQUIRE(svd.S.size() == 2);
    REQUIRE(svd.U.size() == 3);
    REQUIRE(svd.U[0].size() == 3);
    REQUIRE(svd.Vt.size() == 2);
    REQUIRE(svd.Vt[0].size() == 2);

    const std::vector<std::vector<double>> reconstructed = ReconstructFromSVD(svd);
    for (size_t r = 0; r < matrix.size(); ++r) {
        for (size_t c = 0; c < matrix[r].size(); ++c) {
            REQUIRE(reconstructed[r][c] == Catch::Approx(matrix[r][c]).margin(1e-10));
        }
    }

    for (size_t col = 0; col < 3; ++col) {
        REQUIRE(ColumnDot(svd.U, col, col) == Catch::Approx(1.0).margin(1e-10));
    }
    REQUIRE(ColumnDot(svd.U, 0, 1) == Catch::Approx(0.0).margin(1e-10));
    REQUIRE(ColumnDot(svd.U, 0, 2) == Catch::Approx(0.0).margin(1e-10));
    REQUIRE(ColumnDot(svd.U, 1, 2) == Catch::Approx(0.0).margin(1e-10));
}

TEST_CASE("LinearAlgebra CPU SVD powers rank condition and low-rank helpers", "[linalg][svd]") {
    const std::vector<std::vector<double>> matrix = {
        {3.0, 0.0},
        {0.0, 2.0},
        {0.0, 0.0},
    };

    cyxwiz::ScalarResult rank = cyxwiz::LinearAlgebra::Rank(matrix);
    REQUIRE(rank.success);
    REQUIRE(rank.value == Catch::Approx(2.0));

    cyxwiz::ScalarResult condition = cyxwiz::LinearAlgebra::ConditionNumber(matrix);
    REQUIRE(condition.success);
    REQUIRE(condition.value == Catch::Approx(1.5).margin(1e-10));

    cyxwiz::MatrixResult low_rank = cyxwiz::LinearAlgebra::LowRankApproximation(matrix, 1);
    REQUIRE(low_rank.success);
    REQUIRE(low_rank.rows == 3);
    REQUIRE(low_rank.cols == 2);
    REQUIRE(low_rank.matrix[0][0] == Catch::Approx(3.0).margin(1e-10));
    REQUIRE(low_rank.matrix[0][1] == Catch::Approx(0.0).margin(1e-10));
    REQUIRE(low_rank.matrix[1][0] == Catch::Approx(0.0).margin(1e-10));
    REQUIRE(low_rank.matrix[1][1] == Catch::Approx(0.0).margin(1e-10));
    REQUIRE(low_rank.matrix[2][0] == Catch::Approx(0.0).margin(1e-10));
    REQUIRE(low_rank.matrix[2][1] == Catch::Approx(0.0).margin(1e-10));
}
