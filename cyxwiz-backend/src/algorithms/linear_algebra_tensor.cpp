// Windows compatibility
#ifdef _WIN32
#define NOMINMAX
#endif

#include "cyxwiz/linear_algebra.h"
#include "arrayfire_backend_utils.h"
#include <spdlog/spdlog.h>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#ifdef CYXWIZ_HAS_ARRAYFIRE
#include <arrayfire.h>
#endif

// Undefine Windows macros that conflict with std::min/max
#ifdef min
#undef min
#endif
#ifdef max
#undef max
#endif

namespace cyxwiz {

#ifdef CYXWIZ_HAS_ARRAYFIRE
static cyxwiz::DataType AfTypeToTensorType(af::dtype dtype) {
    switch (dtype) {
        case af::dtype::f32: return cyxwiz::DataType::Float32;
        case af::dtype::f64: return cyxwiz::DataType::Float64;
        case af::dtype::s32: return cyxwiz::DataType::Int32;
        case af::dtype::s64: return cyxwiz::DataType::Int64;
        case af::dtype::u8:  return cyxwiz::DataType::UInt8;
        default: return cyxwiz::DataType::Float64;
    }
}

static cyxwiz::Tensor AfArrayToTensorWithShape(const af::array& arr, const std::vector<size_t>& shape) {
    cyxwiz::Tensor out(shape, AfTypeToTensorType(arr.type()));
    if (out.NumBytes() > 0) {
        af::array materialized = arr;
        materialized.eval();
        materialized.host(out.Data());
    }
    return out;
}

static std::string BuildTensorContext(const char* tensor_name, const Tensor& tensor) {
    return BuildTensorShapeContext(tensor_name, tensor.Shape());
}

static std::string BuildTensorContext(
    const char* left_name,
    const Tensor& left,
    const char* right_name,
    const Tensor& right)
{
    return BuildTensorContext(left_name, left) + "; " + BuildTensorContext(right_name, right);
}

static void LogLinearAlgebraTensorFallbackOnce(
    const char* operation_name,
    const char* error_message,
    const std::string& tensor_context)
{
    const BackendFallbackReason reason = ClassifyArrayFireBackendFallbackReason(error_message);
    const std::string context = BuildArrayFireBackendFallbackContext(tensor_context);
    if (ShouldLogArrayFireBackendFallbackOnce(operation_name, reason, context)) {
        spdlog::warn("{}",
            BuildArrayFireBackendFallbackMessage(
                operation_name,
                reason,
                reason != BackendFallbackReason::CudaJitParamOverflow,
                error_message,
                context));
    }
}
#endif

static double TensorValueAsDouble(const Tensor& t, size_t idx) {
    switch (t.GetDataType()) {
        case DataType::Float64:
            return static_cast<const double*>(t.Data())[idx];
        case DataType::Float32:
            return static_cast<double>(static_cast<const float*>(t.Data())[idx]);
        case DataType::Int32:
            return static_cast<double>(static_cast<const int32_t*>(t.Data())[idx]);
        case DataType::Int64:
            return static_cast<double>(static_cast<const int64_t*>(t.Data())[idx]);
        case DataType::UInt8:
            return static_cast<double>(static_cast<const uint8_t*>(t.Data())[idx]);
        default:
            return 0.0;
    }
}

static std::vector<std::vector<double>> Tensor2DToMatrix(const Tensor& t) {
    const auto& shape = t.Shape();
    if (shape.size() != 2) {
        return {};
    }

    const size_t rows = shape[0];
    const size_t cols = shape[1];
    std::vector<std::vector<double>> out(rows, std::vector<double>(cols));

    if (rows == 0 || cols == 0) {
        return out;
    }

    if (t.GetDataType() == DataType::Float64) {
        const double* src = static_cast<const double*>(t.Data());
        for (size_t r = 0; r < rows; ++r) {
            std::memcpy(out[r].data(), src + r * cols, cols * sizeof(double));
        }
        return out;
    }

    for (size_t r = 0; r < rows; ++r) {
        for (size_t c = 0; c < cols; ++c) {
            out[r][c] = TensorValueAsDouble(t, r * cols + c);
        }
    }
    return out;
}

static std::vector<std::vector<double>> TensorVectorOr2DToMatrix(const Tensor& t) {
    const auto& shape = t.Shape();
    if (shape.size() == 2) {
        return Tensor2DToMatrix(t);
    }
    if (shape.size() == 1) {
        const size_t n = shape[0];
        std::vector<std::vector<double>> out(n, std::vector<double>(1));
        for (size_t i = 0; i < n; ++i) {
            out[i][0] = TensorValueAsDouble(t, i);
        }
        return out;
    }
    return {};
}

static Tensor MatrixToTensor(const std::vector<std::vector<double>>& mat, bool squeeze_single_col) {
    if (mat.empty()) {
        if (squeeze_single_col) {
            return Tensor({0}, DataType::Float64);
        }
        return Tensor({0, 0}, DataType::Float64);
    }

    const size_t rows = mat.size();
    const size_t cols = mat[0].size();

    if (squeeze_single_col && cols == 1) {
        Tensor out({rows}, DataType::Float64);
        double* dst = static_cast<double*>(out.Data());
        for (size_t r = 0; r < rows; ++r) {
            dst[r] = mat[r][0];
        }
        return out;
    }

    Tensor out({rows, cols}, DataType::Float64);
    double* dst = static_cast<double*>(out.Data());
    for (size_t r = 0; r < rows; ++r) {
        std::memcpy(dst + r * cols, mat[r].data(), cols * sizeof(double));
    }
    return out;
}


// ============================================================================
// Tensor-First Operations
// ============================================================================

TensorResult LinearAlgebra::Multiply(const Tensor& A, const Tensor& B) {
    TensorResult result;

    const auto& shapeA = A.Shape();
    const auto& shapeB = B.Shape();
    if (shapeA.size() != 2 || shapeB.size() != 2) {
        result.error_message = "A and B must be 2D tensors for matrix multiplication";
        return result;
    }

    const size_t colsA = shapeA[1];
    const size_t rowsB = shapeB[0];
    if (colsA != rowsB) {
        result.error_message = "Matrix A columns must equal Matrix B rows for multiplication";
        return result;
    }

#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        af::array aA = A.GetArrayRowMajor2D();
        af::array aB = B.GetArrayRowMajor2D();
        if (aA.type() != af::dtype::f32 && aA.type() != af::dtype::f64) {
            aA = aA.as(af::dtype::f64);
        }
        if (aB.type() != af::dtype::f32 && aB.type() != af::dtype::f64) {
            aB = aB.as(af::dtype::f64);
        }

        af::array aC = af::matmul(aA, aB);
        aC.eval();
        result.tensor = Tensor::FromArrayRowMajor2D(aC);
        result.success = true;
        return result;
    } catch (const af::exception& e) {
        LogLinearAlgebraTensorFallbackOnce(
            "LinearAlgebra::TensorMultiply",
            e.what(),
            BuildTensorContext("A", A, "B", B));
    }
#endif

    auto A_mat = Tensor2DToMatrix(A);
    auto B_mat = Tensor2DToMatrix(B);
    auto cpu_result = Multiply(A_mat, B_mat);
    if (!cpu_result.success) {
        result.error_message = cpu_result.error_message;
        return result;
    }

    result.tensor = MatrixToTensor(cpu_result.matrix, false);
    result.success = true;
    return result;
}

TensorResult LinearAlgebra::Transpose(const Tensor& A) {
    TensorResult result;

    const auto& shape = A.Shape();
    if (shape.size() != 2) {
        result.error_message = "A must be a 2D tensor";
        return result;
    }

    const size_t rows = shape[0];
    const size_t cols = shape[1];

#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        af::array aA = A.GetArray();
        if (aA.type() != af::dtype::f32 && aA.type() != af::dtype::f64) {
            aA = aA.as(af::dtype::f64);
        }

        af::array aT = af::transpose(aA);
        aT.eval();
        result.tensor = AfArrayToTensorWithShape(aT, {cols, rows});
        result.success = true;
        return result;
    } catch (const af::exception& e) {
        LogLinearAlgebraTensorFallbackOnce(
            "LinearAlgebra::TensorTranspose",
            e.what(),
            BuildTensorContext("A", A));
    }
#endif

    auto A_mat = Tensor2DToMatrix(A);
    auto cpu_result = Transpose(A_mat);
    if (!cpu_result.success) {
        result.error_message = cpu_result.error_message;
        return result;
    }

    result.tensor = MatrixToTensor(cpu_result.matrix, false);
    result.success = true;
    return result;
}

TensorResult LinearAlgebra::Inverse(const Tensor& A) {
    TensorResult result;

    const auto& shape = A.Shape();
    if (shape.size() != 2) {
        result.error_message = "A must be a 2D tensor";
        return result;
    }
    if (shape[0] != shape[1]) {
        result.error_message = "Matrix must be square for inversion";
        return result;
    }

    const size_t n = shape[0];

#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        af::array aA = A.GetArray();
        if (aA.type() != af::dtype::f32 && aA.type() != af::dtype::f64) {
            aA = aA.as(af::dtype::f64);
        }

        af::array aInv = af::inverse(aA);
        aInv.eval();
        result.tensor = AfArrayToTensorWithShape(aInv, {n, n});
        result.success = true;
        return result;
    } catch (const af::exception& e) {
        LogLinearAlgebraTensorFallbackOnce(
            "LinearAlgebra::TensorInverse",
            e.what(),
            BuildTensorContext("A", A));
    }
#endif

    auto A_mat = Tensor2DToMatrix(A);
    auto cpu_result = Inverse(A_mat);
    if (!cpu_result.success) {
        result.error_message = cpu_result.error_message;
        return result;
    }

    result.tensor = MatrixToTensor(cpu_result.matrix, false);
    result.success = true;
    return result;
}

ScalarResult LinearAlgebra::FrobeniusNorm(const Tensor& A) {
    ScalarResult result;

    const auto& shape = A.Shape();
    if (shape.size() != 2) {
        result.error_message = "A must be a 2D tensor";
        return result;
    }

#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        af::array aA = A.GetArray();
        if (aA.type() != af::dtype::f32 && aA.type() != af::dtype::f64) {
            aA = aA.as(af::dtype::f64);
        }
        af::array sq = aA * aA;
        sq.eval();
        af::array flat_sq = af::flat(sq);
        flat_sq.eval();
        result.value = std::sqrt(af::sum<double>(flat_sq));
        result.success = true;
        return result;
    } catch (const af::exception& e) {
        LogLinearAlgebraTensorFallbackOnce(
            "LinearAlgebra::TensorFrobeniusNorm",
            e.what(),
            BuildTensorContext("A", A));
    }
#endif

    auto A_mat = Tensor2DToMatrix(A);
    return FrobeniusNorm(A_mat);
}

TensorResult LinearAlgebra::Solve(const Tensor& A, const Tensor& b) {
    TensorResult result;

    const auto& shapeA = A.Shape();
    const auto& shapeB = b.Shape();

    if (shapeA.size() != 2) {
        result.error_message = "A must be a 2D tensor";
        return result;
    }
    if (shapeA[0] != shapeA[1]) {
        result.error_message = "Matrix A must be square for Solve";
        return result;
    }
    if (shapeB.size() != 1 && shapeB.size() != 2) {
        result.error_message = "b must be a 1D or 2D tensor";
        return result;
    }

    const size_t n = shapeA[0];
    const bool b_was_vector = (shapeB.size() == 1);
    const size_t b_rows = b_was_vector ? shapeB[0] : shapeB[0];
    const size_t b_cols = b_was_vector ? 1 : shapeB[1];
    if (b_rows != n) {
        result.error_message = "Dimensions mismatch: A rows must equal b rows";
        return result;
    }

#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        af::array aA = A.GetArray();
        af::array aB = b.GetArray();
        if (aA.type() != af::dtype::f32 && aA.type() != af::dtype::f64) {
            aA = aA.as(af::dtype::f64);
        }
        if (aB.type() != af::dtype::f32 && aB.type() != af::dtype::f64) {
            aB = aB.as(af::dtype::f64);
        }
        if (b_was_vector) {
            aB = af::moddims(aB, static_cast<dim_t>(n), static_cast<dim_t>(1));
        }

        af::array x = af::solve(aA, aB);
        x.eval();
        if (b_was_vector) {
            af::array x_vec = af::moddims(x, static_cast<dim_t>(n));
            x_vec.eval();
            result.tensor = AfArrayToTensorWithShape(x_vec, {n});
        } else {
            result.tensor = AfArrayToTensorWithShape(x, {n, b_cols});
        }
        result.success = true;
        return result;
    } catch (const af::exception& e) {
        LogLinearAlgebraTensorFallbackOnce(
            "LinearAlgebra::TensorSolve",
            e.what(),
            BuildTensorContext("A", A, "b", b));
    }
#endif

    auto A_mat = Tensor2DToMatrix(A);
    auto b_mat = TensorVectorOr2DToMatrix(b);
    auto cpu_result = Solve(A_mat, b_mat);
    if (!cpu_result.success) {
        result.error_message = cpu_result.error_message;
        return result;
    }

    result.tensor = MatrixToTensor(cpu_result.matrix, b_was_vector);
    result.success = true;
    return result;
}

TensorResult LinearAlgebra::LeastSquares(const Tensor& A, const Tensor& b) {
    TensorResult result;

    const auto& shapeA = A.Shape();
    const auto& shapeB = b.Shape();

    if (shapeA.size() != 2) {
        result.error_message = "A must be a 2D tensor";
        return result;
    }
    if (shapeB.size() != 1 && shapeB.size() != 2) {
        result.error_message = "b must be a 1D or 2D tensor";
        return result;
    }

    const size_t rowsA = shapeA[0];
    const size_t colsA = shapeA[1];
    const bool b_was_vector = (shapeB.size() == 1);
    const size_t rowsB = b_was_vector ? shapeB[0] : shapeB[0];
    const size_t colsB = b_was_vector ? 1 : shapeB[1];
    if (rowsA != rowsB) {
        result.error_message = "A and b must have same number of rows";
        return result;
    }

#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        af::array aA = A.GetArray();
        af::array aB = b.GetArray();
        if (aA.type() != af::dtype::f32 && aA.type() != af::dtype::f64) {
            aA = aA.as(af::dtype::f64);
        }
        if (aB.type() != af::dtype::f32 && aB.type() != af::dtype::f64) {
            aB = aB.as(af::dtype::f64);
        }
        if (b_was_vector) {
            aB = af::moddims(aB, static_cast<dim_t>(rowsB), static_cast<dim_t>(1));
        }

        af::array x = af::solve(aA, aB, AF_MAT_NONE);
        x.eval();
        if (b_was_vector) {
            af::array x_vec = af::moddims(x, static_cast<dim_t>(colsA));
            x_vec.eval();
            result.tensor = AfArrayToTensorWithShape(x_vec, {colsA});
        } else {
            result.tensor = AfArrayToTensorWithShape(x, {colsA, colsB});
        }
        result.success = true;
        return result;
    } catch (const af::exception& e) {
        LogLinearAlgebraTensorFallbackOnce(
            "LinearAlgebra::TensorLeastSquares",
            e.what(),
            BuildTensorContext("A", A, "b", b));
    }
#endif

    auto A_mat = Tensor2DToMatrix(A);
    auto b_mat = TensorVectorOr2DToMatrix(b);
    auto cpu_result = LeastSquares(A_mat, b_mat);
    if (!cpu_result.success) {
        result.error_message = cpu_result.error_message;
        return result;
    }

    result.tensor = MatrixToTensor(cpu_result.matrix, b_was_vector);
    result.success = true;
    return result;
}


} // namespace cyxwiz
