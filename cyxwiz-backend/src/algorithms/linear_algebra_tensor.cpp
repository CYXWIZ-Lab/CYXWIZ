// Windows compatibility
#ifdef _WIN32
#define NOMINMAX
#endif

#include "cyxwiz/linear_algebra.h"
#include <cmath>
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
        result.error_message = std::string("LinearAlgebra::TensorMultiply failed on the ArrayFire device: ") + e.what();
        return result;
    }
#else
    result.error_message = "LinearAlgebra::TensorMultiply runs on ArrayFire, and this build has no ArrayFire";
    return result;
#endif
}

TensorResult LinearAlgebra::Transpose(const Tensor& A) {
    TensorResult result;

    const auto& shape = A.Shape();
    if (shape.size() != 2) {
        result.error_message = "A must be a 2D tensor";
        return result;
    }


#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        af::array aA = A.GetArrayRowMajor2D();
        if (aA.type() != af::dtype::f32 && aA.type() != af::dtype::f64) {
            aA = aA.as(af::dtype::f64);
        }

        af::array aT = af::transpose(aA);
        aT.eval();
        result.tensor = Tensor::FromArrayRowMajor2D(aT);
        result.success = true;
        return result;
    } catch (const af::exception& e) {
        result.error_message = std::string("LinearAlgebra::TensorTranspose failed on the ArrayFire device: ") + e.what();
        return result;
    }
#else
    result.error_message = "LinearAlgebra::TensorTranspose runs on ArrayFire, and this build has no ArrayFire";
    return result;
#endif
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


#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        af::array aA = A.GetArrayRowMajor2D();
        if (aA.type() != af::dtype::f32 && aA.type() != af::dtype::f64) {
            aA = aA.as(af::dtype::f64);
        }

        af::array aInv = af::inverse(aA);
        aInv.eval();
        result.tensor = Tensor::FromArrayRowMajor2D(aInv);
        result.success = true;
        return result;
    } catch (const af::exception& e) {
        result.error_message = std::string("LinearAlgebra::TensorInverse failed on the ArrayFire device: ") + e.what();
        return result;
    }
#else
    result.error_message = "LinearAlgebra::TensorInverse runs on ArrayFire, and this build has no ArrayFire";
    return result;
#endif
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
        af::array aA = A.GetArrayRowMajor2D();
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
        result.error_message = std::string("LinearAlgebra::TensorFrobeniusNorm failed on the ArrayFire device: ") + e.what();
        return result;
    }
#else
    result.error_message = "LinearAlgebra::TensorFrobeniusNorm runs on ArrayFire, and this build has no ArrayFire";
    return result;
#endif
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
    if (b_rows != n) {
        result.error_message = "Dimensions mismatch: A rows must equal b rows";
        return result;
    }

#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        af::array aA = A.GetArrayRowMajor2D();
        af::array aB = b.GetArrayRowMajor2D();
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
            result.tensor = Tensor::FromSemanticArray(x_vec, {n});
        } else {
            result.tensor = Tensor::FromArrayRowMajor2D(x);
        }
        result.success = true;
        return result;
    } catch (const af::exception& e) {
        result.error_message = std::string("LinearAlgebra::TensorSolve failed on the ArrayFire device: ") + e.what();
        return result;
    }
#else
    result.error_message = "LinearAlgebra::TensorSolve runs on ArrayFire, and this build has no ArrayFire";
    return result;
#endif
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
    if (rowsA != rowsB) {
        result.error_message = "A and b must have same number of rows";
        return result;
    }

#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        af::array aA = A.GetArrayRowMajor2D();
        af::array aB = b.GetArrayRowMajor2D();
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
            result.tensor = Tensor::FromSemanticArray(x_vec, {colsA});
        } else {
            result.tensor = Tensor::FromArrayRowMajor2D(x);
        }
        result.success = true;
        return result;
    } catch (const af::exception& e) {
        result.error_message = std::string("LinearAlgebra::TensorLeastSquares failed on the ArrayFire device: ") + e.what();
        return result;
    }
#else
    result.error_message = "LinearAlgebra::TensorLeastSquares runs on ArrayFire, and this build has no ArrayFire";
    return result;
#endif
}


} // namespace cyxwiz
