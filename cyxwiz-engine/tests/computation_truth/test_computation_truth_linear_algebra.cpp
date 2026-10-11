// The host LinearAlgebra API (the analysis panels) against PyTorch torch.linalg
// in float64 (TOFIX140, linear algebra on the device).
//
// fixtures/linear_algebra_pytorch.json (generate_linear_algebra_fixtures.py).
// Runs on the active ArrayFire backend (CYXWIZ_TEST_ARRAYFIRE_BACKEND=
// cuda|opencl|cpu). Decompositions with free signs are checked through
// reconstruction, orthonormality and PyTorch's sign-free values.
#include "test_device_selection.h"

#include <cyxwiz/linear_algebra.h>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

using json = nlohmann::json;
using Matrix = std::vector<std::vector<double>>;

constexpr double kTolerance = 1e-8;  // float64 on the device

void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << std::endl;
        std::exit(1);
    }
}

void CheckValue(double actual, double expected, const std::string& what) {
    Check(std::abs(actual - expected) <= kTolerance * (1.0 + std::abs(expected)),
          what + ": " + std::to_string(actual) + ", PyTorch " + std::to_string(expected));
}

void CheckValues(const std::vector<double>& actual, const json& expected, const std::string& what) {
    const auto values = expected.get<std::vector<double>>();
    Check(actual.size() == values.size(), what + ": count " + std::to_string(actual.size()));
    for (size_t i = 0; i < values.size(); ++i) CheckValue(actual[i], values[i], what + "[" + std::to_string(i) + "]");
}

void CheckMatrix(const Matrix& actual, const Matrix& expected, const std::string& what) {
    Check(actual.size() == expected.size(), what + ": rows");
    for (size_t r = 0; r < expected.size(); ++r) {
        Check(actual[r].size() == expected[r].size(), what + ": cols");
        for (size_t c = 0; c < expected[r].size(); ++c) {
            CheckValue(actual[r][c], expected[r][c], what + "(" + std::to_string(r) + "," + std::to_string(c) + ")");
        }
    }
}

Matrix Product(const Matrix& a, const Matrix& b) {
    Matrix out(a.size(), std::vector<double>(b[0].size(), 0.0));
    for (size_t i = 0; i < a.size(); ++i)
        for (size_t k = 0; k < b.size(); ++k)
            for (size_t j = 0; j < b[0].size(); ++j) out[i][j] += a[i][k] * b[k][j];
    return out;
}

Matrix Transposed(const Matrix& a) {
    Matrix out(a[0].size(), std::vector<double>(a.size()));
    for (size_t i = 0; i < a.size(); ++i)
        for (size_t j = 0; j < a[0].size(); ++j) out[j][i] = a[i][j];
    return out;
}

void CheckOrthonormalColumns(const Matrix& q, const std::string& what) {
    const Matrix gram = Product(Transposed(q), q);
    for (size_t i = 0; i < gram.size(); ++i)
        for (size_t j = 0; j < gram.size(); ++j) CheckValue(gram[i][j], i == j ? 1.0 : 0.0, what + " orthonormal");
}

std::filesystem::path FixturePath(const char* argv0) {
    const auto beside = std::filesystem::path(argv0).parent_path() / "computation_truth_fixtures" /
                        "linear_algebra_pytorch.json";
    if (std::filesystem::exists(beside)) return beside;
    return std::filesystem::path(CYXWIZ_LINEAR_ALGEBRA_FIXTURE);
}

void RunCase(const json& c, const std::string& name) {
    using cyxwiz::LinearAlgebra;
    const std::string op = c.at("op").get<std::string>();
    const Matrix A = c.at("inputs").at("A").get<Matrix>();
    const Matrix B = c.at("inputs").contains("B") ? c.at("inputs").at("B").get<Matrix>() : Matrix{};
    const auto expect_matrix = [&](const cyxwiz::MatrixResult& result) {
        Check(result.success, name + ": " + result.error_message);
        CheckMatrix(result.matrix, c.at("matrix").get<Matrix>(), name);
        Check(result.rows == static_cast<int>(result.matrix.size()) &&
                  result.cols == static_cast<int>(result.matrix[0].size()),
              name + ": rows / cols fields");
    };
    const auto expect_scalar = [&](const cyxwiz::ScalarResult& result) {
        Check(result.success, name + ": " + result.error_message);
        CheckValue(result.value, c.at("value").get<double>(), name);
    };

    if (op == "add") return expect_matrix(LinearAlgebra::Add(A, B));
    if (op == "subtract") return expect_matrix(LinearAlgebra::Subtract(A, B));
    if (op == "multiply") return expect_matrix(LinearAlgebra::Multiply(A, B));
    if (op == "scalar_multiply") return expect_matrix(LinearAlgebra::ScalarMultiply(A, c.at("scalar").get<double>()));
    if (op == "transpose") return expect_matrix(LinearAlgebra::Transpose(A));
    if (op == "inverse") return expect_matrix(LinearAlgebra::Inverse(A));
    if (op == "solve") return expect_matrix(LinearAlgebra::Solve(A, B));
    if (op == "least_squares") return expect_matrix(LinearAlgebra::LeastSquares(A, B));
    if (op == "low_rank") return expect_matrix(LinearAlgebra::LowRankApproximation(A, c.at("k").get<int>()));
    if (op == "cholesky") {
        const auto result = LinearAlgebra::Cholesky(A);
        Check(result.success && result.is_positive_definite, name + ": " + result.error_message);
        return CheckMatrix(result.L, c.at("matrix").get<Matrix>(), name);
    }
    if (op == "inverse_singular" || op == "solve_singular") {
        const auto result = op == "inverse_singular" ? LinearAlgebra::Inverse(A) : LinearAlgebra::Solve(A, B);
        Check(!result.success && result.error_message.find("singular") != std::string::npos,
              name + ": singular matrix must be refused, got '" + result.error_message + "'");
        return;
    }
    if (op == "cholesky_not_pd") {
        const auto result = LinearAlgebra::Cholesky(A);
        Check(!result.success && !result.is_positive_definite, name + ": indefinite matrix must be refused");
        Check(!LinearAlgebra::IsPositiveDefinite(A), name + ": IsPositiveDefinite");
        return;
    }
    if (op == "determinant") return expect_scalar(LinearAlgebra::Determinant(A));
    if (op == "trace") return expect_scalar(LinearAlgebra::Trace(A));
    if (op == "rank") return expect_scalar(LinearAlgebra::Rank(A));
    if (op == "frobenius") return expect_scalar(LinearAlgebra::FrobeniusNorm(A));
    if (op == "condition") return expect_scalar(LinearAlgebra::ConditionNumber(A));
    if (op == "singular_values") {
        for (const bool full : {false, true}) {
            const auto svd = LinearAlgebra::SVD(A, full);
            const std::string tag = name + (full ? " full" : " thin");
            Check(svd.success, tag + ": " + svd.error_message);
            CheckValues(svd.S, c.at("values"), tag + " S");
            const size_t m = A.size(), n = A[0].size(), k = std::min(m, n);
            Check(svd.U.size() == m && svd.U[0].size() == (full ? m : k), tag + ": U shape");
            Check(svd.Vt.size() == (full ? n : k) && svd.Vt[0].size() == n, tag + ": Vt shape");
            CheckOrthonormalColumns(svd.U, tag + " U");
            CheckOrthonormalColumns(Transposed(svd.Vt), tag + " V");
            Matrix us(m, std::vector<double>(k));
            for (size_t i = 0; i < m; ++i)
                for (size_t j = 0; j < k; ++j) us[i][j] = svd.U[i][j] * svd.S[j];
            Matrix vt_k(svd.Vt.begin(), svd.Vt.begin() + static_cast<std::ptrdiff_t>(k));
            CheckMatrix(Product(us, vt_k), A, tag + " U S Vt = A");
        }
        return;
    }
    if (op == "eigen_symmetric") {
        const auto eigen = LinearAlgebra::Eigen(A);
        Check(eigen.success, name + ": " + eigen.error_message);
        std::vector<double> values;
        Matrix vectors(A.size(), std::vector<double>(A.size()));
        for (size_t j = 0; j < A.size(); ++j) {
            Check(eigen.eigenvalues[j].imag() == 0.0, name + ": real eigenvalues");
            values.push_back(eigen.eigenvalues[j].real());
            for (size_t i = 0; i < A.size(); ++i) vectors[i][j] = eigen.eigenvectors[i][j].real();
        }
        CheckValues(values, c.at("values"), name + " eigenvalues");
        CheckOrthonormalColumns(vectors, name + " eigenvectors");
        const Matrix av = Product(A, vectors);
        for (size_t i = 0; i < A.size(); ++i)
            for (size_t j = 0; j < A.size(); ++j) CheckValue(av[i][j], values[j] * vectors[i][j], name + " A v = l v");
        return;
    }
    if (op == "eigen_2x2") {
        const auto eigen = LinearAlgebra::Eigen(A);
        Check(eigen.success, name + ": " + eigen.error_message);
        const auto real = c.at("real").get<std::vector<double>>(), imag = c.at("imag").get<std::vector<double>>();
        for (size_t j = 0; j < 2; ++j) {
            CheckValue(eigen.eigenvalues[j].real(), real[j], name + " real");
            CheckValue(std::abs(eigen.eigenvalues[j].imag()), std::abs(imag[j]), name + " |imag|");
            for (size_t i = 0; i < 2; ++i) {
                const std::complex<double> av = A[i][0] * eigen.eigenvectors[0][j] + A[i][1] * eigen.eigenvectors[1][j];
                const std::complex<double> lv = eigen.eigenvalues[j] * eigen.eigenvectors[i][j];
                Check(std::abs(av - lv) <= kTolerance, name + ": A v = l v");
            }
        }
        return;
    }
    if (op == "qr") {
        const auto qr = LinearAlgebra::QR(A);
        Check(qr.success, name + ": " + qr.error_message);
        const size_t m = A.size(), n = A[0].size(), k = std::min(m, n);
        Check(qr.Q.size() == m && qr.Q[0].size() == k && qr.R.size() == k && qr.R[0].size() == n, name + ": reduced shapes");
        CheckOrthonormalColumns(qr.Q, name + " Q");
        Check(LinearAlgebra::IsOrthogonal(qr.Q), name + ": IsOrthogonal(Q)");
        for (size_t i = 0; i < k; ++i)
            for (size_t j = 0; j < i && j < n; ++j) CheckValue(qr.R[i][j], 0.0, name + " R upper triangular");
        std::vector<double> diag;
        for (size_t i = 0; i < k; ++i) diag.push_back(std::abs(qr.R[i][i]));
        CheckValues(diag, c.at("abs_diag_r"), name + " |diag R|");
        return CheckMatrix(Product(qr.Q, qr.R), A, name + " Q R = A");
    }
    if (op == "lu") {
        const auto lu = LinearAlgebra::LU(A);
        Check(lu.success, name + ": " + lu.error_message);
        CheckMatrix(lu.L, c.at("L").get<Matrix>(), name + " L");
        CheckMatrix(lu.U, c.at("U").get<Matrix>(), name + " U");
        Check(lu.P == c.at("perm").get<std::vector<int>>(), name + ": permutation");
        return;
    }
    if (op == "is_symmetric") {
        Check(LinearAlgebra::IsSymmetric(A) == c.at("flag").get<bool>(), name);
        return;
    }
    if (op == "is_orthogonal") {
        Check(LinearAlgebra::IsOrthogonal(A) == c.at("flag").get<bool>(), name);
        return;
    }
    Check(false, name + ": unknown op " + op);
}

}  // namespace

int main(int, char** argv) {
    Check(cyxwiz::test::SelectTestDeviceFromEnvironment(), "requested test device");
    std::ifstream in(FixturePath(argv[0]));
    Check(static_cast<bool>(in), "cannot open the linear algebra fixture");
    const json fixture = json::parse(in);
    size_t passed = 0;
    for (const auto& c : fixture.at("cases")) {
        const std::string name = c.at("op").get<std::string>() + (c.contains("label") ? " " + c.at("label").get<std::string>() : "");
        try {
            RunCase(c, name);
        } catch (const std::exception& error) {
            Check(false, name + ": " + error.what());
        }
        ++passed;
    }
    std::cout << "linear algebra matches PyTorch: " << passed << " cases" << std::endl;
    return 0;
}
