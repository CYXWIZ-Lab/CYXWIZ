#include <cyxwiz/optimization.h>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <limits>
#include <spdlog/spdlog.h>

namespace cyxwiz {

// ============================================================================
// Utility Functions
// ============================================================================

double Optimization::VectorNorm(const std::vector<double>& v) {
    double sum = 0.0;
    for (double x : v) {
        sum += x * x;
    }
    return std::sqrt(sum);
}

// ============================================================================
// Gradient Descent Variants
// ============================================================================

GradientDescentResult Optimization::GradientDescent(
    std::function<double(const std::vector<double>&)> objective,
    std::function<std::vector<double>(const std::vector<double>&)> gradient,
    const std::vector<double>& x0,
    double learning_rate,
    int max_iterations,
    double tolerance
) {
    GradientDescentResult result;

    if (x0.empty()) {
        result.error_message = "Initial point cannot be empty";
        return result;
    }

    std::vector<double> x = x0;
    result.path.push_back(x);
    result.objective_history.push_back(objective(x));

    for (int i = 0; i < max_iterations; i++) {
        std::vector<double> grad = gradient(x);
        double grad_norm = VectorNorm(grad);

        // Update x
        for (size_t j = 0; j < x.size(); j++) {
            x[j] -= learning_rate * grad[j];
        }

        double obj = objective(x);
        result.path.push_back(x);
        result.objective_history.push_back(obj);
        result.iterations = i + 1;
        result.gradient_norm = grad_norm;

        // Check convergence
        if (grad_norm < tolerance) {
            result.converged = true;
            break;
        }
    }

    result.solution = x;
    result.final_objective = objective(x);
    result.success = true;

    return result;
}

GradientDescentResult Optimization::MomentumGD(
    std::function<double(const std::vector<double>&)> objective,
    std::function<std::vector<double>(const std::vector<double>&)> gradient,
    const std::vector<double>& x0,
    double learning_rate,
    double momentum,
    int max_iterations,
    double tolerance
) {
    GradientDescentResult result;

    if (x0.empty()) {
        result.error_message = "Initial point cannot be empty";
        return result;
    }

    std::vector<double> x = x0;
    std::vector<double> velocity(x.size(), 0.0);

    result.path.push_back(x);
    result.objective_history.push_back(objective(x));

    for (int i = 0; i < max_iterations; i++) {
        std::vector<double> grad = gradient(x);
        double grad_norm = VectorNorm(grad);

        // Update velocity and position
        for (size_t j = 0; j < x.size(); j++) {
            velocity[j] = momentum * velocity[j] - learning_rate * grad[j];
            x[j] += velocity[j];
        }

        double obj = objective(x);
        result.path.push_back(x);
        result.objective_history.push_back(obj);
        result.iterations = i + 1;
        result.gradient_norm = grad_norm;

        if (grad_norm < tolerance) {
            result.converged = true;
            break;
        }
    }

    result.solution = x;
    result.final_objective = objective(x);
    result.success = true;

    return result;
}

GradientDescentResult Optimization::Adam(
    std::function<double(const std::vector<double>&)> objective,
    std::function<std::vector<double>(const std::vector<double>&)> gradient,
    const std::vector<double>& x0,
    double learning_rate,
    double beta1,
    double beta2,
    double epsilon,
    int max_iterations,
    double tolerance
) {
    GradientDescentResult result;

    if (x0.empty()) {
        result.error_message = "Initial point cannot be empty";
        return result;
    }

    std::vector<double> x = x0;
    std::vector<double> m(x.size(), 0.0);  // First moment
    std::vector<double> v(x.size(), 0.0);  // Second moment

    result.path.push_back(x);
    result.objective_history.push_back(objective(x));

    for (int i = 0; i < max_iterations; i++) {
        std::vector<double> grad = gradient(x);
        double grad_norm = VectorNorm(grad);

        double t = i + 1;

        // Update biased moments
        for (size_t j = 0; j < x.size(); j++) {
            m[j] = beta1 * m[j] + (1 - beta1) * grad[j];
            v[j] = beta2 * v[j] + (1 - beta2) * grad[j] * grad[j];
        }

        // Bias correction
        double beta1_t = 1.0 - std::pow(beta1, t);
        double beta2_t = 1.0 - std::pow(beta2, t);

        // Update parameters
        for (size_t j = 0; j < x.size(); j++) {
            double m_hat = m[j] / beta1_t;
            double v_hat = v[j] / beta2_t;
            x[j] -= learning_rate * m_hat / (std::sqrt(v_hat) + epsilon);
        }

        double obj = objective(x);
        result.path.push_back(x);
        result.objective_history.push_back(obj);
        result.iterations = i + 1;
        result.gradient_norm = grad_norm;

        if (grad_norm < tolerance) {
            result.converged = true;
            break;
        }
    }

    result.solution = x;
    result.final_objective = objective(x);
    result.success = true;

    return result;
}

GradientDescentResult Optimization::RMSprop(
    std::function<double(const std::vector<double>&)> objective,
    std::function<std::vector<double>(const std::vector<double>&)> gradient,
    const std::vector<double>& x0,
    double learning_rate,
    double decay_rate,
    double epsilon,
    int max_iterations,
    double tolerance
) {
    GradientDescentResult result;

    if (x0.empty()) {
        result.error_message = "Initial point cannot be empty";
        return result;
    }

    std::vector<double> x = x0;
    std::vector<double> cache(x.size(), 0.0);

    result.path.push_back(x);
    result.objective_history.push_back(objective(x));

    for (int i = 0; i < max_iterations; i++) {
        std::vector<double> grad = gradient(x);
        double grad_norm = VectorNorm(grad);

        // Update cache and position
        for (size_t j = 0; j < x.size(); j++) {
            cache[j] = decay_rate * cache[j] + (1 - decay_rate) * grad[j] * grad[j];
            x[j] -= learning_rate * grad[j] / (std::sqrt(cache[j]) + epsilon);
        }

        double obj = objective(x);
        result.path.push_back(x);
        result.objective_history.push_back(obj);
        result.iterations = i + 1;
        result.gradient_norm = grad_norm;

        if (grad_norm < tolerance) {
            result.converged = true;
            break;
        }
    }

    result.solution = x;
    result.final_objective = objective(x);
    result.success = true;

    return result;
}

// ============================================================================
// Convexity Analysis
// ============================================================================

std::vector<std::vector<double>> Optimization::ComputeHessian(
    std::function<double(const std::vector<double>&)> func,
    const std::vector<double>& point,
    double delta
) {
    int n = static_cast<int>(point.size());
    std::vector<std::vector<double>> hessian(n, std::vector<double>(n, 0.0));

    for (int i = 0; i < n; i++) {
        for (int j = 0; j <= i; j++) {
            std::vector<double> pp = point, pm = point, mp = point, mm = point;

            pp[i] += delta;
            pp[j] += delta;

            pm[i] += delta;
            pm[j] -= delta;

            mp[i] -= delta;
            mp[j] += delta;

            mm[i] -= delta;
            mm[j] -= delta;

            double h2 = (func(pp) - func(pm) - func(mp) + func(mm)) / (4.0 * delta * delta);
            hessian[i][j] = h2;
            hessian[j][i] = h2;  // Symmetric
        }
    }

    return hessian;
}

ConvexityResult Optimization::AnalyzeConvexity(
    std::function<double(const std::vector<double>&)> func,
    const std::vector<double>& point,
    double delta
) {
    ConvexityResult result;

    if (point.empty()) {
        result.error_message = "Point cannot be empty";
        return result;
    }

    result.hessian = ComputeHessian(func, point, delta);
    int n = static_cast<int>(point.size());

    // Compute eigenvalues using power iteration (simplified)
    // For proper implementation, use a library like Eigen
    // Here we use a simple diagonal dominance check for 2x2

    if (n == 2) {
        // Closed-form eigenvalues for 2x2 symmetric matrix
        double a = result.hessian[0][0];
        double b = result.hessian[0][1];
        double c = result.hessian[1][1];

        double trace = a + c;
        double det = a * c - b * b;
        double discriminant = trace * trace - 4 * det;

        if (discriminant >= 0) {
            double sqrt_disc = std::sqrt(discriminant);
            result.eigenvalues.push_back((trace + sqrt_disc) / 2.0);
            result.eigenvalues.push_back((trace - sqrt_disc) / 2.0);
        } else {
            // Complex eigenvalues - not symmetric (shouldn't happen for Hessian)
            result.error_message = "Hessian has complex eigenvalues";
            return result;
        }
    } else {
        // For larger matrices, use diagonal elements as approximation
        // (This is a simplification - proper eigenvalue computation would need a library)
        for (int i = 0; i < n; i++) {
            result.eigenvalues.push_back(result.hessian[i][i]);
        }
    }

    // Find min and max eigenvalues
    result.min_eigenvalue = *std::min_element(result.eigenvalues.begin(), result.eigenvalues.end());
    result.max_eigenvalue = *std::max_element(result.eigenvalues.begin(), result.eigenvalues.end());

    // Determine convexity
    double tol = 1e-10;
    if (result.min_eigenvalue > tol) {
        result.is_strictly_convex = true;
        result.is_convex = true;
        result.analysis = "Strictly convex: All eigenvalues positive";
    } else if (result.min_eigenvalue >= -tol) {
        result.is_convex = true;
        result.analysis = "Convex: All eigenvalues non-negative";
    } else if (result.max_eigenvalue < -tol) {
        result.is_strictly_concave = true;
        result.is_concave = true;
        result.analysis = "Strictly concave: All eigenvalues negative";
    } else if (result.max_eigenvalue <= tol) {
        result.is_concave = true;
        result.analysis = "Concave: All eigenvalues non-positive";
    } else {
        result.analysis = "Neither convex nor concave: Hessian is indefinite";
    }

    result.success = true;
    return result;
}

// ============================================================================
// Linear Programming (Simplex Method)
// ============================================================================

LPResult Optimization::SolveLP(
    const std::vector<double>& c,
    const std::vector<std::vector<double>>& A,
    const std::vector<double>& b,
    const std::vector<std::string>& constraint_types,
    bool maximize
) {
    LPResult result;

    if (c.empty() || A.empty() || b.empty()) {
        result.error_message = "Empty input";
        return result;
    }

    int num_vars = static_cast<int>(c.size());
    int num_constraints = static_cast<int>(A.size());

    // Convert to standard form with slack variables
    // For simplicity, assume all constraints are <=
    int num_slack = num_constraints;
    int total_vars = num_vars + num_slack;

    // Build simplex tableau
    std::vector<std::vector<double>> tableau(num_constraints + 1, std::vector<double>(total_vars + 1, 0.0));

    // Fill constraint rows
    for (int i = 0; i < num_constraints; i++) {
        for (int j = 0; j < num_vars; j++) {
            tableau[i][j] = A[i][j];
        }
        tableau[i][num_vars + i] = 1.0;  // Slack variable
        tableau[i][total_vars] = b[i];   // RHS
    }

    // Fill objective row (last row)
    for (int j = 0; j < num_vars; j++) {
        tableau[num_constraints][j] = maximize ? -c[j] : c[j];
    }

    // Basic variables (slack variables initially)
    std::vector<int> basic(num_constraints);
    for (int i = 0; i < num_constraints; i++) {
        basic[i] = num_vars + i;
    }

    // Simplex iterations
    const int max_iters = 1000;
    for (int iter = 0; iter < max_iters; iter++) {
        result.iterations = iter;

        // Find entering variable (most negative in objective row)
        int entering = -1;
        double min_val = -1e-10;
        for (int j = 0; j < total_vars; j++) {
            if (tableau[num_constraints][j] < min_val) {
                min_val = tableau[num_constraints][j];
                entering = j;
            }
        }

        if (entering == -1) {
            // Optimal solution found
            result.status = "Optimal";
            break;
        }

        // Find leaving variable (minimum ratio test)
        int leaving = -1;
        double min_ratio = std::numeric_limits<double>::max();
        for (int i = 0; i < num_constraints; i++) {
            if (tableau[i][entering] > 1e-10) {
                double ratio = tableau[i][total_vars] / tableau[i][entering];
                if (ratio < min_ratio) {
                    min_ratio = ratio;
                    leaving = i;
                }
            }
        }

        if (leaving == -1) {
            result.status = "Unbounded";
            result.error_message = "Problem is unbounded";
            return result;
        }

        // Pivot
        double pivot = tableau[leaving][entering];
        for (int j = 0; j <= total_vars; j++) {
            tableau[leaving][j] /= pivot;
        }

        for (int i = 0; i <= num_constraints; i++) {
            if (i != leaving) {
                double factor = tableau[i][entering];
                for (int j = 0; j <= total_vars; j++) {
                    tableau[i][j] -= factor * tableau[leaving][j];
                }
            }
        }

        basic[leaving] = entering;
    }

    // Extract solution
    result.solution.resize(num_vars, 0.0);
    for (int i = 0; i < num_constraints; i++) {
        if (basic[i] < num_vars) {
            result.solution[basic[i]] = tableau[i][total_vars];
        }
    }

    result.objective_value = -tableau[num_constraints][total_vars];
    if (!maximize) {
        result.objective_value = -result.objective_value;
    }

    // Dual variables (shadow prices)
    result.dual_variables.resize(num_constraints);
    for (int i = 0; i < num_constraints; i++) {
        result.dual_variables[i] = tableau[num_constraints][num_vars + i];
    }

    result.success = true;
    if (result.status.empty()) result.status = "Optimal";

    return result;
}

LPResult Optimization::SolveLPWithBounds(
    const std::vector<double>& c,
    const std::vector<std::vector<double>>& A,
    const std::vector<double>& b,
    const std::vector<std::string>& constraint_types,
    const std::vector<double>& lower_bounds,
    const std::vector<double>& upper_bounds,
    bool maximize
) {
    // Convert bounded LP to standard form by adding bound constraints
    std::vector<std::vector<double>> A_extended = A;
    std::vector<double> b_extended = b;
    std::vector<std::string> types_extended = constraint_types;

    int n = static_cast<int>(c.size());

    // Add upper bound constraints
    for (int i = 0; i < n; i++) {
        if (upper_bounds[i] < std::numeric_limits<double>::max()) {
            std::vector<double> row(n, 0.0);
            row[i] = 1.0;
            A_extended.push_back(row);
            b_extended.push_back(upper_bounds[i]);
            types_extended.push_back("<=");
        }
    }

    // Add lower bound constraints
    for (int i = 0; i < n; i++) {
        if (lower_bounds[i] > -std::numeric_limits<double>::max()) {
            std::vector<double> row(n, 0.0);
            row[i] = -1.0;
            A_extended.push_back(row);
            b_extended.push_back(-lower_bounds[i]);
            types_extended.push_back("<=");
        }
    }

    return SolveLP(c, A_extended, b_extended, types_extended, maximize);
}

// ============================================================================
// Quadratic Programming
// ============================================================================

QPResult Optimization::SolveUnconstrainedQP(
    const std::vector<std::vector<double>>& Q,
    const std::vector<double>& c
) {
    QPResult result;

    int n = static_cast<int>(c.size());

    // For unconstrained QP: minimize 0.5*x'Qx + c'x
    // Optimal: Qx = -c, so x = -Q^(-1)*c

    // Simple Gauss elimination for small matrices
    std::vector<std::vector<double>> aug(n, std::vector<double>(n + 1));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            aug[i][j] = Q[i][j];
        }
        aug[i][n] = -c[i];
    }

    // Forward elimination
    for (int i = 0; i < n; i++) {
        // Find pivot
        int max_row = i;
        for (int k = i + 1; k < n; k++) {
            if (std::abs(aug[k][i]) > std::abs(aug[max_row][i])) {
                max_row = k;
            }
        }
        std::swap(aug[i], aug[max_row]);

        if (std::abs(aug[i][i]) < 1e-12) {
            result.error_message = "Matrix is singular";
            result.status = "Infeasible";
            return result;
        }

        for (int k = i + 1; k < n; k++) {
            double factor = aug[k][i] / aug[i][i];
            for (int j = i; j <= n; j++) {
                aug[k][j] -= factor * aug[i][j];
            }
        }
    }

    // Back substitution
    result.solution.resize(n);
    for (int i = n - 1; i >= 0; i--) {
        result.solution[i] = aug[i][n];
        for (int j = i + 1; j < n; j++) {
            result.solution[i] -= aug[i][j] * result.solution[j];
        }
        result.solution[i] /= aug[i][i];
    }

    // Compute objective value
    result.objective_value = 0.0;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            result.objective_value += 0.5 * result.solution[i] * Q[i][j] * result.solution[j];
        }
        result.objective_value += c[i] * result.solution[i];
    }

    result.success = true;
    result.status = "Optimal";

    return result;
}

QPResult Optimization::SolveQP(
    const std::vector<std::vector<double>>& Q,
    const std::vector<double>& c,
    const std::vector<std::vector<double>>& A,
    const std::vector<double>& b
) {
    QPResult result;

    if (A.empty()) {
        // Unconstrained case
        return SolveUnconstrainedQP(Q, c);
    }

    // For constrained QP, use active set method (simplified)
    // Start with unconstrained solution
    QPResult unconstrained = SolveUnconstrainedQP(Q, c);

    if (!unconstrained.success) {
        return unconstrained;
    }

    // Check if unconstrained solution is feasible
    bool feasible = true;
    for (size_t i = 0; i < A.size(); i++) {
        double sum = 0;
        for (size_t j = 0; j < unconstrained.solution.size(); j++) {
            sum += A[i][j] * unconstrained.solution[j];
        }
        if (sum > b[i] + 1e-10) {
            feasible = false;
            break;
        }
    }

    if (feasible) {
        return unconstrained;
    }

    // If not feasible, use gradient projection (simplified)
    // This is a basic implementation - full active set would be more complex
    std::vector<double> x = unconstrained.solution;

    const int max_iters = 1000;
    double lr = 0.01;

    for (int iter = 0; iter < max_iters; iter++) {
        // Compute gradient: Qx + c
        std::vector<double> grad(x.size(), 0.0);
        for (size_t i = 0; i < x.size(); i++) {
            grad[i] = c[i];
            for (size_t j = 0; j < x.size(); j++) {
                grad[i] += Q[i][j] * x[j];
            }
        }

        // Gradient step
        for (size_t i = 0; i < x.size(); i++) {
            x[i] -= lr * grad[i];
        }

        // Project onto feasible region (simple clipping)
        for (size_t i = 0; i < A.size(); i++) {
            double sum = 0;
            for (size_t j = 0; j < x.size(); j++) {
                sum += A[i][j] * x[j];
            }
            if (sum > b[i]) {
                // Project back
                double scale = b[i] / sum;
                for (size_t j = 0; j < x.size(); j++) {
                    x[j] *= scale;
                }
            }
        }

        result.iterations = iter + 1;
    }

    result.solution = x;

    // Compute objective
    result.objective_value = 0.0;
    int n = static_cast<int>(x.size());
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            result.objective_value += 0.5 * x[i] * Q[i][j] * x[j];
        }
        result.objective_value += c[i] * x[i];
    }

    result.success = true;
    result.status = "Optimal";

    return result;
}

// ============================================================================
// Numerical Differentiation
// ============================================================================

DerivativeResult Optimization::NumericalDerivative(
    std::function<double(double)> func,
    double x,
    double h,
    const std::string& method
) {
    DerivativeResult result;

    double f_plus = func(x + h);
    double f_minus = func(x - h);
    double f_x = func(x);

    result.forward_estimate = (f_plus - f_x) / h;
    result.backward_estimate = (f_x - f_minus) / h;
    result.central_estimate = (f_plus - f_minus) / (2 * h);

    if (method == "forward") {
        result.value = result.forward_estimate;
    } else if (method == "backward") {
        result.value = result.backward_estimate;
    } else {
        result.value = result.central_estimate;
    }

    // Estimate error (difference between methods)
    result.approximation_error = std::abs(result.central_estimate - result.forward_estimate);

    result.success = true;
    return result;
}

DerivativeResult Optimization::NumericalGradient(
    std::function<double(const std::vector<double>&)> func,
    const std::vector<double>& x,
    double h
) {
    DerivativeResult result;
    int n = static_cast<int>(x.size());

    result.gradient.resize(n);

    for (int i = 0; i < n; i++) {
        std::vector<double> x_plus = x;
        std::vector<double> x_minus = x;
        x_plus[i] += h;
        x_minus[i] -= h;

        result.gradient[i] = (func(x_plus) - func(x_minus)) / (2 * h);
    }

    result.success = true;
    return result;
}

DerivativeResult Optimization::SecondDerivative(
    std::function<double(double)> func,
    double x,
    double h
) {
    DerivativeResult result;

    double f_plus = func(x + h);
    double f_minus = func(x - h);
    double f_x = func(x);

    result.value = (f_plus - 2 * f_x + f_minus) / (h * h);
    result.success = true;

    return result;
}

DerivativeResult Optimization::CompareDerivativeMethods(
    std::function<double(double)> func,
    double x,
    double h
) {
    return NumericalDerivative(func, x, h, "central");
}

// ============================================================================
// Numerical Integration
// ============================================================================

IntegralResult Optimization::TrapezoidalRule(
    std::function<double(double)> func,
    double a, double b,
    int n
) {
    IntegralResult result;
    result.method_used = "Trapezoidal";

    double h = (b - a) / n;
    double sum = 0.5 * (func(a) + func(b));

    result.x_points.push_back(a);
    result.y_points.push_back(func(a));

    for (int i = 1; i < n; i++) {
        double x = a + i * h;
        double y = func(x);
        sum += y;
        result.x_points.push_back(x);
        result.y_points.push_back(y);
    }

    result.x_points.push_back(b);
    result.y_points.push_back(func(b));

    result.value = h * sum;
    result.function_evaluations = n + 1;
    result.success = true;

    return result;
}

IntegralResult Optimization::SimpsonsRule(
    std::function<double(double)> func,
    double a, double b,
    int n
) {
    IntegralResult result;
    result.method_used = "Simpson's";

    // Ensure n is even
    if (n % 2 != 0) n++;

    double h = (b - a) / n;
    double sum = func(a) + func(b);

    result.x_points.push_back(a);
    result.y_points.push_back(func(a));

    for (int i = 1; i < n; i++) {
        double x = a + i * h;
        double y = func(x);
        sum += (i % 2 == 0 ? 2 : 4) * y;
        result.x_points.push_back(x);
        result.y_points.push_back(y);
    }

    result.x_points.push_back(b);
    result.y_points.push_back(func(b));

    result.value = h * sum / 3.0;
    result.function_evaluations = n + 1;
    result.success = true;

    return result;
}

IntegralResult Optimization::RombergIntegration(
    std::function<double(double)> func,
    double a, double b,
    int max_iterations,
    double tolerance
) {
    IntegralResult result;
    result.method_used = "Romberg";

    std::vector<std::vector<double>> R(max_iterations, std::vector<double>(max_iterations, 0.0));

    // First trapezoidal estimate
    double h = b - a;
    R[0][0] = h * (func(a) + func(b)) / 2.0;
    result.function_evaluations = 2;

    for (int i = 1; i < max_iterations; i++) {
        h /= 2;

        // Composite trapezoidal rule
        double sum = 0;
        int num_new_points = 1 << (i - 1);  // 2^(i-1)
        for (int k = 0; k < num_new_points; k++) {
            sum += func(a + (2 * k + 1) * h);
        }
        result.function_evaluations += num_new_points;

        R[i][0] = R[i-1][0] / 2.0 + h * sum;

        // Richardson extrapolation
        for (int j = 1; j <= i; j++) {
            double factor = std::pow(4.0, j);
            R[i][j] = (factor * R[i][j-1] - R[i-1][j-1]) / (factor - 1.0);
        }

        // Check convergence
        if (i > 0 && std::abs(R[i][i] - R[i-1][i-1]) < tolerance) {
            result.value = R[i][i];
            result.absolute_error = std::abs(R[i][i] - R[i-1][i-1]);
            result.success = true;
            return result;
        }
    }

    result.value = R[max_iterations-1][max_iterations-1];
    result.success = true;

    return result;
}

IntegralResult Optimization::AdaptiveIntegrate(
    std::function<double(double)> func,
    double a, double b,
    double tolerance,
    int max_depth
) {
    IntegralResult result;
    result.method_used = "Adaptive Simpson's";
    result.function_evaluations = 0;

    // Recursive adaptive integration helper
    std::function<double(double, double, double, double, double, int)> adaptive_helper;

    adaptive_helper = [&](double a, double b, double fa, double fb, double fc, int depth) -> double {
        double c = (a + b) / 2.0;
        double d = (a + c) / 2.0;
        double e = (c + b) / 2.0;

        double fd = func(d);
        double fe = func(e);
        result.function_evaluations += 2;

        double S1 = (b - a) * (fa + 4*fc + fb) / 6.0;
        double S2 = (b - a) * (fa + 4*fd + 2*fc + 4*fe + fb) / 12.0;

        if (depth >= max_depth || std::abs(S2 - S1) < 15 * tolerance) {
            result.x_points.push_back(c);
            result.y_points.push_back(fc);
            return S2 + (S2 - S1) / 15.0;
        }

        return adaptive_helper(a, c, fa, fc, fd, depth + 1) +
               adaptive_helper(c, b, fc, fb, fe, depth + 1);
    };

    double fa = func(a);
    double fb = func(b);
    double fc = func((a + b) / 2.0);
    result.function_evaluations = 3;

    result.x_points.push_back(a);
    result.y_points.push_back(fa);
    result.x_points.push_back(b);
    result.y_points.push_back(fb);

    result.value = adaptive_helper(a, b, fa, fb, fc, 0);
    result.success = true;

    return result;
}

IntegralResult Optimization::GaussianQuadrature(
    std::function<double(double)> func,
    double a, double b,
    int n
) {
    IntegralResult result;
    result.method_used = "Gaussian Quadrature";

    // Gauss-Legendre quadrature nodes and weights for common n
    std::vector<double> nodes, weights;

    switch (n) {
        case 1:
            nodes = {0.0};
            weights = {2.0};
            break;
        case 2:
            nodes = {-0.5773502691896257, 0.5773502691896257};
            weights = {1.0, 1.0};
            break;
        case 3:
            nodes = {-0.7745966692414834, 0.0, 0.7745966692414834};
            weights = {0.5555555555555556, 0.8888888888888888, 0.5555555555555556};
            break;
        case 4:
            nodes = {-0.8611363115940526, -0.3399810435848563, 0.3399810435848563, 0.8611363115940526};
            weights = {0.3478548451374538, 0.6521451548625461, 0.6521451548625461, 0.3478548451374538};
            break;
        case 5:
            nodes = {-0.9061798459386640, -0.5384693101056831, 0.0, 0.5384693101056831, 0.9061798459386640};
            weights = {0.2369268850561891, 0.4786286704993665, 0.5688888888888889, 0.4786286704993665, 0.2369268850561891};
            break;
        default:
            n = 5;
            nodes = {-0.9061798459386640, -0.5384693101056831, 0.0, 0.5384693101056831, 0.9061798459386640};
            weights = {0.2369268850561891, 0.4786286704993665, 0.5688888888888889, 0.4786286704993665, 0.2369268850561891};
    }

    // Transform from [-1, 1] to [a, b]
    double sum = 0.0;
    double c1 = (b - a) / 2.0;
    double c2 = (b + a) / 2.0;

    for (int i = 0; i < n; i++) {
        double x = c1 * nodes[i] + c2;
        double y = func(x);
        sum += weights[i] * y;
        result.x_points.push_back(x);
        result.y_points.push_back(y);
    }

    result.value = c1 * sum;
    result.function_evaluations = n;
    result.success = true;

    return result;
}

IntegralResult Optimization::Integrate(
    std::function<double(double)> func,
    double a, double b,
    const std::string& method,
    int n
) {
    if (method == "trapezoid") {
        return TrapezoidalRule(func, a, b, n);
    } else if (method == "simpson") {
        return SimpsonsRule(func, a, b, n);
    } else if (method == "romberg") {
        return RombergIntegration(func, a, b);
    } else if (method == "adaptive") {
        return AdaptiveIntegrate(func, a, b);
    } else if (method == "gaussian") {
        return GaussianQuadrature(func, a, b, std::min(n, 5));
    } else {
        return SimpsonsRule(func, a, b, n);
    }
}

std::vector<IntegralResult> Optimization::CompareIntegrationMethods(
    std::function<double(double)> func,
    double a, double b,
    int n
) {
    std::vector<IntegralResult> results;
    results.push_back(TrapezoidalRule(func, a, b, n));
    results.push_back(SimpsonsRule(func, a, b, n));
    results.push_back(RombergIntegration(func, a, b));
    results.push_back(GaussianQuadrature(func, a, b, 5));
    return results;
}

// ============================================================================
// Test Functions
// ============================================================================

double Optimization::Rosenbrock(const std::vector<double>& x) {
    if (x.size() < 2) return 0.0;
    double a = 1.0, b = 100.0;
    return (a - x[0]) * (a - x[0]) + b * (x[1] - x[0] * x[0]) * (x[1] - x[0] * x[0]);
}

std::vector<double> Optimization::RosenbrockGradient(const std::vector<double>& x) {
    if (x.size() < 2) return {0.0, 0.0};
    double a = 1.0, b = 100.0;
    double dx = -2.0 * (a - x[0]) - 4.0 * b * x[0] * (x[1] - x[0] * x[0]);
    double dy = 2.0 * b * (x[1] - x[0] * x[0]);
    return {dx, dy};
}

double Optimization::Sphere(const std::vector<double>& x) {
    double sum = 0.0;
    for (double xi : x) {
        sum += xi * xi;
    }
    return sum;
}

std::vector<double> Optimization::SphereGradient(const std::vector<double>& x) {
    std::vector<double> grad(x.size());
    for (size_t i = 0; i < x.size(); i++) {
        grad[i] = 2.0 * x[i];
    }
    return grad;
}

double Optimization::Rastrigin(const std::vector<double>& x) {
    const double PI = 3.14159265358979323846;
    double sum = 10.0 * x.size();
    for (double xi : x) {
        sum += xi * xi - 10.0 * std::cos(2.0 * PI * xi);
    }
    return sum;
}

std::vector<double> Optimization::RastriginGradient(const std::vector<double>& x) {
    const double PI = 3.14159265358979323846;
    std::vector<double> grad(x.size());
    for (size_t i = 0; i < x.size(); i++) {
        grad[i] = 2.0 * x[i] + 20.0 * PI * std::sin(2.0 * PI * x[i]);
    }
    return grad;
}

double Optimization::Beale(const std::vector<double>& x) {
    if (x.size() < 2) return 0.0;
    double t1 = 1.5 - x[0] + x[0] * x[1];
    double t2 = 2.25 - x[0] + x[0] * x[1] * x[1];
    double t3 = 2.625 - x[0] + x[0] * x[1] * x[1] * x[1];
    return t1 * t1 + t2 * t2 + t3 * t3;
}

std::vector<double> Optimization::BealeGradient(const std::vector<double>& x) {
    if (x.size() < 2) return {0.0, 0.0};
    double t1 = 1.5 - x[0] + x[0] * x[1];
    double t2 = 2.25 - x[0] + x[0] * x[1] * x[1];
    double t3 = 2.625 - x[0] + x[0] * x[1] * x[1] * x[1];

    double dx = 2 * t1 * (x[1] - 1) + 2 * t2 * (x[1] * x[1] - 1) + 2 * t3 * (x[1] * x[1] * x[1] - 1);
    double dy = 2 * t1 * x[0] + 2 * t2 * 2 * x[0] * x[1] + 2 * t3 * 3 * x[0] * x[1] * x[1];
    return {dx, dy};
}

double Optimization::Booth(const std::vector<double>& x) {
    if (x.size() < 2) return 0.0;
    double t1 = x[0] + 2 * x[1] - 7;
    double t2 = 2 * x[0] + x[1] - 5;
    return t1 * t1 + t2 * t2;
}

std::vector<double> Optimization::BoothGradient(const std::vector<double>& x) {
    if (x.size() < 2) return {0.0, 0.0};
    double t1 = x[0] + 2 * x[1] - 7;
    double t2 = 2 * x[0] + x[1] - 5;
    double dx = 2 * t1 + 4 * t2;
    double dy = 4 * t1 + 2 * t2;
    return {dx, dy};
}

double Optimization::Himmelblau(const std::vector<double>& x) {
    if (x.size() < 2) return 0.0;
    double t1 = x[0] * x[0] + x[1] - 11;
    double t2 = x[0] + x[1] * x[1] - 7;
    return t1 * t1 + t2 * t2;
}

std::vector<double> Optimization::HimmelblauGradient(const std::vector<double>& x) {
    if (x.size() < 2) return {0.0, 0.0};
    double t1 = x[0] * x[0] + x[1] - 11;
    double t2 = x[0] + x[1] * x[1] - 7;
    double dx = 4 * x[0] * t1 + 2 * t2;
    double dy = 2 * t1 + 4 * x[1] * t2;
    return {dx, dy};
}

// ============================================================================
// Utility Functions for Visualization
// ============================================================================

std::vector<std::vector<double>> Optimization::GenerateContourData(
    std::function<double(const std::vector<double>&)> func,
    double x_min, double x_max,
    double y_min, double y_max,
    int resolution
) {
    std::vector<std::vector<double>> data(resolution, std::vector<double>(resolution));

    double dx = (x_max - x_min) / (resolution - 1);
    double dy = (y_max - y_min) / (resolution - 1);

    for (int i = 0; i < resolution; i++) {
        for (int j = 0; j < resolution; j++) {
            double x = x_min + j * dx;
            double y = y_min + i * dy;
            data[i][j] = func({x, y});
        }
    }

    return data;
}

void Optimization::MeshGrid(
    double x_min, double x_max,
    double y_min, double y_max,
    int nx, int ny,
    std::vector<std::vector<double>>& X,
    std::vector<std::vector<double>>& Y
) {
    X.resize(ny, std::vector<double>(nx));
    Y.resize(ny, std::vector<double>(nx));

    double dx = (x_max - x_min) / (nx - 1);
    double dy = (y_max - y_min) / (ny - 1);

    for (int i = 0; i < ny; i++) {
        for (int j = 0; j < nx; j++) {
            X[i][j] = x_min + j * dx;
            Y[i][j] = y_min + i * dy;
        }
    }
}

} // namespace cyxwiz
