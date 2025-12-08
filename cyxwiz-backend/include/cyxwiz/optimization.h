#pragma once

#include "api_export.h"
#include <vector>
#include <string>
#include <functional>

namespace cyxwiz {

// ============================================================================
// Result Structures
// ============================================================================

struct CYXWIZ_API GradientDescentResult {
    std::vector<double> solution;              // Final x values
    std::vector<double> objective_history;     // f(x) at each iteration
    std::vector<std::vector<double>> path;     // x trajectory
    int iterations = 0;
    double final_objective = 0.0;
    double gradient_norm = 0.0;
    bool converged = false;
    bool success = false;
    std::string error_message;
};

struct CYXWIZ_API ConvexityResult {
    bool is_convex = false;
    bool is_strictly_convex = false;
    bool is_concave = false;
    bool is_strictly_concave = false;
    std::vector<double> eigenvalues;           // Hessian eigenvalues
    std::vector<std::vector<double>> hessian;  // Computed Hessian matrix
    double min_eigenvalue = 0.0;
    double max_eigenvalue = 0.0;
    std::string analysis;                      // Text description
    bool success = false;
    std::string error_message;
};

struct CYXWIZ_API LPResult {
    std::vector<double> solution;              // Optimal x
    double objective_value = 0.0;
    std::vector<double> dual_variables;        // Shadow prices
    std::vector<bool> active_constraints;
    int iterations = 0;
    std::string status;                        // "Optimal", "Unbounded", "Infeasible"
    bool success = false;
    std::string error_message;
};

struct CYXWIZ_API QPResult {
    std::vector<double> solution;
    double objective_value = 0.0;
    std::vector<double> dual_variables;
    std::vector<double> lagrange_multipliers;
    int iterations = 0;
    std::string status;                        // "Optimal", "Unbounded", "Infeasible"
    bool success = false;
    std::string error_message;
};

struct CYXWIZ_API DerivativeResult {
    double value = 0.0;                        // f'(x) or partial derivative
    std::vector<double> gradient;              // For multivariate functions
    std::vector<std::vector<double>> hessian;  // Second derivatives
    double forward_estimate = 0.0;             // Forward difference result
    double backward_estimate = 0.0;            // Backward difference result
    double central_estimate = 0.0;             // Central difference result
    double approximation_error = 0.0;          // Estimated error
    bool success = false;
    std::string error_message;
};

struct CYXWIZ_API IntegralResult {
    double value = 0.0;
    double absolute_error = 0.0;
    int function_evaluations = 0;
    std::string method_used;
    std::vector<double> x_points;              // Evaluation points (for visualization)
    std::vector<double> y_points;              // Function values at points
    bool success = false;
    std::string error_message;
};

// ============================================================================
// Optimization Class
// ============================================================================

class CYXWIZ_API Optimization {
public:
    // ==================== Gradient Descent Variants ====================

    // Basic gradient descent
    static GradientDescentResult GradientDescent(
        std::function<double(const std::vector<double>&)> objective,
        std::function<std::vector<double>(const std::vector<double>&)> gradient,
        const std::vector<double>& x0,
        double learning_rate = 0.01,
        int max_iterations = 1000,
        double tolerance = 1e-6
    );

    // Gradient descent with momentum
    static GradientDescentResult MomentumGD(
        std::function<double(const std::vector<double>&)> objective,
        std::function<std::vector<double>(const std::vector<double>&)> gradient,
        const std::vector<double>& x0,
        double learning_rate = 0.01,
        double momentum = 0.9,
        int max_iterations = 1000,
        double tolerance = 1e-6
    );

    // Adam optimizer
    static GradientDescentResult Adam(
        std::function<double(const std::vector<double>&)> objective,
        std::function<std::vector<double>(const std::vector<double>&)> gradient,
        const std::vector<double>& x0,
        double learning_rate = 0.001,
        double beta1 = 0.9,
        double beta2 = 0.999,
        double epsilon = 1e-8,
        int max_iterations = 1000,
        double tolerance = 1e-6
    );

    // RMSprop optimizer
    static GradientDescentResult RMSprop(
        std::function<double(const std::vector<double>&)> objective,
        std::function<std::vector<double>(const std::vector<double>&)> gradient,
        const std::vector<double>& x0,
        double learning_rate = 0.01,
        double decay_rate = 0.99,
        double epsilon = 1e-8,
        int max_iterations = 1000,
        double tolerance = 1e-6
    );

    // ==================== Convexity Analysis ====================

    // Analyze convexity at a point using numerical Hessian
    static ConvexityResult AnalyzeConvexity(
        std::function<double(const std::vector<double>&)> func,
        const std::vector<double>& point,
        double delta = 1e-5
    );

    // Compute numerical Hessian matrix
    static std::vector<std::vector<double>> ComputeHessian(
        std::function<double(const std::vector<double>&)> func,
        const std::vector<double>& point,
        double delta = 1e-5
    );

    // ==================== Linear Programming ====================

    // Solve linear programming problem:
    // minimize/maximize c'x subject to Ax <= b, x >= 0
    static LPResult SolveLP(
        const std::vector<double>& c,                        // Objective coefficients
        const std::vector<std::vector<double>>& A,           // Constraint matrix
        const std::vector<double>& b,                        // RHS values
        const std::vector<std::string>& constraint_types,    // "<=", ">=", "="
        bool maximize = false                                // true = maximize, false = minimize
    );

    // Solve LP with variable bounds
    static LPResult SolveLPWithBounds(
        const std::vector<double>& c,
        const std::vector<std::vector<double>>& A,
        const std::vector<double>& b,
        const std::vector<std::string>& constraint_types,
        const std::vector<double>& lower_bounds,
        const std::vector<double>& upper_bounds,
        bool maximize = false
    );

    // ==================== Quadratic Programming ====================

    // Solve quadratic programming problem:
    // minimize 0.5 * x'Qx + c'x subject to Ax <= b
    static QPResult SolveQP(
        const std::vector<std::vector<double>>& Q,           // Quadratic term (symmetric)
        const std::vector<double>& c,                        // Linear term
        const std::vector<std::vector<double>>& A,           // Constraint matrix
        const std::vector<double>& b                         // RHS values
    );

    // Solve unconstrained QP (closed-form solution if Q is positive definite)
    static QPResult SolveUnconstrainedQP(
        const std::vector<std::vector<double>>& Q,
        const std::vector<double>& c
    );

    // ==================== Numerical Differentiation ====================

    // Single variable derivative
    static DerivativeResult NumericalDerivative(
        std::function<double(double)> func,
        double x,
        double h = 1e-5,
        const std::string& method = "central"  // "forward", "backward", "central"
    );

    // Gradient of multivariate function
    static DerivativeResult NumericalGradient(
        std::function<double(const std::vector<double>&)> func,
        const std::vector<double>& x,
        double h = 1e-5
    );

    // Second derivative (single variable)
    static DerivativeResult SecondDerivative(
        std::function<double(double)> func,
        double x,
        double h = 1e-5
    );

    // Compare all differentiation methods
    static DerivativeResult CompareDerivativeMethods(
        std::function<double(double)> func,
        double x,
        double h = 1e-5
    );

    // ==================== Numerical Integration ====================

    // Basic integration methods
    static IntegralResult Integrate(
        std::function<double(double)> func,
        double a, double b,
        const std::string& method = "simpson",  // "trapezoid", "simpson", "midpoint"
        int n = 100
    );

    // Composite trapezoidal rule
    static IntegralResult TrapezoidalRule(
        std::function<double(double)> func,
        double a, double b,
        int n = 100
    );

    // Composite Simpson's rule
    static IntegralResult SimpsonsRule(
        std::function<double(double)> func,
        double a, double b,
        int n = 100  // Must be even
    );

    // Romberg integration (higher accuracy)
    static IntegralResult RombergIntegration(
        std::function<double(double)> func,
        double a, double b,
        int max_iterations = 10,
        double tolerance = 1e-10
    );

    // Adaptive integration (adjusts subdivisions based on error)
    static IntegralResult AdaptiveIntegrate(
        std::function<double(double)> func,
        double a, double b,
        double tolerance = 1e-6,
        int max_depth = 50
    );

    // Gaussian quadrature
    static IntegralResult GaussianQuadrature(
        std::function<double(double)> func,
        double a, double b,
        int n = 5  // Number of quadrature points (1-10)
    );

    // Compare all integration methods
    static std::vector<IntegralResult> CompareIntegrationMethods(
        std::function<double(double)> func,
        double a, double b,
        int n = 100
    );

    // ==================== Test Functions (Built-in Presets) ====================

    // Rosenbrock function: f(x,y) = (1-x)^2 + 100(y-x^2)^2
    static double Rosenbrock(const std::vector<double>& x);
    static std::vector<double> RosenbrockGradient(const std::vector<double>& x);

    // Sphere function: f(x) = sum(x_i^2)
    static double Sphere(const std::vector<double>& x);
    static std::vector<double> SphereGradient(const std::vector<double>& x);

    // Rastrigin function: f(x) = 10n + sum(x_i^2 - 10*cos(2*pi*x_i))
    static double Rastrigin(const std::vector<double>& x);
    static std::vector<double> RastriginGradient(const std::vector<double>& x);

    // Beale function: for benchmarking
    static double Beale(const std::vector<double>& x);
    static std::vector<double> BealeGradient(const std::vector<double>& x);

    // Booth function: f(x,y) = (x + 2y - 7)^2 + (2x + y - 5)^2
    static double Booth(const std::vector<double>& x);
    static std::vector<double> BoothGradient(const std::vector<double>& x);

    // Himmelblau function: f(x,y) = (x^2 + y - 11)^2 + (x + y^2 - 7)^2
    static double Himmelblau(const std::vector<double>& x);
    static std::vector<double> HimmelblauGradient(const std::vector<double>& x);

    // ==================== Utility Functions ====================

    // Compute norm of vector
    static double VectorNorm(const std::vector<double>& v);

    // Generate contour data for 2D visualization
    static std::vector<std::vector<double>> GenerateContourData(
        std::function<double(const std::vector<double>&)> func,
        double x_min, double x_max,
        double y_min, double y_max,
        int resolution = 50
    );

    // Generate mesh grid
    static void MeshGrid(
        double x_min, double x_max,
        double y_min, double y_max,
        int nx, int ny,
        std::vector<std::vector<double>>& X,
        std::vector<std::vector<double>>& Y
    );
};

} // namespace cyxwiz
