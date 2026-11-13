#pragma once

// Tools > Optimization and Calculus (TOFIX129 D10): one window for the
// operations of core/optimization_presentation, in three groups. Minimize:
// the training optimizers on a typed formula, their paths on its contour map.
// Programs: linear and quadratic programs of any size (two variables drawn).
// Calculus: integral, derivative and curvature of a typed formula. Each
// answer is worked out in the background as the inputs change (Stop ends
// it): a sentence, its tables and plots; Copy and Export CSV. A portfolio
// read from a table of returns can be kept as a node of the graph.

#include "../../core/optimization_presentation.h"
#include "../data_studio/explore_tab.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace cyxwiz {

class OptimizationPanel {
public:
    OptimizationPanel();

    void Render();

    bool IsVisible() const { return visible_; }
    void SetVisible(bool v) { visible_ = v; }
    // Opens the window on the operation `id` ("optimizers", "linear", ...).
    void Open(const std::string& id);
    // Keep as a node: (the table it reads, the node) -> what happened.
    using AddToGraph = std::function<std::string(const std::string&, const optcalc::GraphHandOff&)>;
    void SetAddToGraph(AddToGraph add) { add_to_graph_ = std::move(add); }

private:
    struct Job {
        optcalc::Op op = optcalc::Op::Optimizers;
        optcalc::MinimizeRequest minimize_request;
        optcalc::MinimizeAnswer minimize;
        optcalc::SurfaceMap map;                 // Minimize, Curvature, a QP of 2 variables
        optcalc::ProgramRequest program_request;
        optcalc::ProgramAnswer program;
        bool from_table = false;                 // a portfolio read from a returns table
        optcalc::PortfolioChoice portfolio;
        std::string portfolio_note;
        optcalc::IntegralAnswer integral;
        optcalc::DerivativeAnswer derivative;
        optcalc::CurvatureAnswer curvature;
        std::string at, point_x, point_y;        // the inputs the labels name, as asked
        std::string error;                       // "Cancelled." only: a refusal is an answer
        double milliseconds = 0;
        bool ok() const { return error.empty(); }
    };

    void RenderOperations();
    void RenderInputs();
    void RenderMinimizeInputs();
    void RenderProgramInputs();
    void RenderPortfolioInputs();
    void RenderLineInputs();
    void RenderCurvatureInputs();
    void RenderAnswer();
    void RenderMinimize(const Job& job);
    void RenderProgram(const Job& job);
    void RenderIntegral(const Job& job);
    void RenderDerivative(const Job& job);
    void RenderCurvature(const Job& job);
    // Copy and Export CSV of the answer shown.
    void RenderActions(const Job& job);
    bool RenderFormula(const char* id, std::string& text, const char* label);
    bool RenderRange();
    void SetOp(optcalc::Op op);
    void ApplyExample(const optcalc::FormulaExample& e);
    void ApplyExample(const optcalc::LineExample& e);
    optcalc::ProgramRequest& Program();
    void ResizeProgram(int variables);
    void Changed();
    std::string Key() const;

    bool visible_ = false;
    optcalc::Op op_ = optcalc::Op::Optimizers;

    // Minimize and Curvature: a formula of x and y and the range shown.
    std::string surface_ = "(1 - x)^2 + 100*(y - x^2)^2";
    double x_min_ = -2.5, x_max_ = 2.0, y_min_ = -1.0, y_max_ = 3.0;
    std::string minima_ = "(1, 1)";
    double x0_ = -2.0, y0_ = 2.0;
    int steps_ = 1000;
    std::vector<optcalc::OptimizerChoice> optimizers_;
    size_t shown_steps_ = 0;                     // 0: whole paths
    std::string point_x_ = "0", point_y_ = "0";  // Curvature

    // Programs.
    optcalc::ProgramRequest linear_, quadratic_;
    std::vector<std::string> lower_text_, upper_text_;   // the shown program's bounds as typed
    std::string bounds_error_;
    bool paste_open_ = false;
    std::string paste_;
    std::string paste_error_;
    int qp_source_ = 0;                          // 0 typed, 1 a returns table
    optcalc::PortfolioChoice portfolio_;
    std::vector<std::string> tables_;            // the session's tables, read when the list opens
    std::vector<std::string> table_columns_;     // the chosen table's number columns
    std::vector<char> picked_;
    std::string columns_for_, columns_error_;
    uint64_t columns_task_ = 0;
    int refresh_ = 0;                            // Read again
    std::shared_ptr<int> alive_ = std::make_shared<int>(0);
    AddToGraph add_to_graph_;
    std::string graph_message_;                  // what Keep as a node did

    // Integral and Derivative: a formula of x.
    std::string line_ = "sin(x)";
    std::string a_ = "0", b_ = "pi";
    int rule_ = 0;                               // QuadratureRule
    int n_ = 100;
    std::string at_ = "1";
    int difference_ = 2;                         // DifferenceRule: Central
    std::string step_text_;                      // "" : automatic

    std::string key_;
    bool dirty_ = true;
    std::string stopped_key_;
    ExploreJob<Job> job_;
    std::shared_ptr<const Job> shown_;
    std::string shown_key_;
    std::string message_;
    bool message_error_ = false;
};

}  // namespace cyxwiz
