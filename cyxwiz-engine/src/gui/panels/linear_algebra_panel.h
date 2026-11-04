#pragma once

// Tools > Linear Algebra (TOFIX129 D9): one window for the matrix operations
// (core/linear_algebra_presentation): the operations in four groups, matrix A
// typed or pasted, the second input an operation takes (B, b or k), and the
// answer, worked out in the background as A changes: its sentence, figures,
// bars, a tab per result matrix and the check that the factors give A back,
// or the reason it cannot run; Copy, Export CSV and Use as A.

#include "../../core/linear_algebra_presentation.h"
#include "../data_studio/explore_tab.h"

#include <memory>
#include <string>

namespace cyxwiz {

class LinearAlgebraPanel {
public:
    LinearAlgebraPanel();

    void Render();

    bool IsVisible() const { return visible_; }
    void SetVisible(bool v) { visible_ = v; }
    // Opens the window on the operation `id` ("svd", "multiply", ...).
    void Open(const std::string& id);

private:
    struct Job {
        linalg::TableAnswer result;  // the answer, and the inputs read from tables
        std::string error;           // "Cancelled." only: a refusal is an answer
        double milliseconds = 0;
        bool ok() const { return error.empty(); }
    };
    // A table chosen as an input: its number columns come from a background query.
    struct TableInput {
        linalg::TableChoice choice;
        std::vector<std::string> columns;   // the table's number columns
        std::vector<char> picked;           // per column
        std::string columns_for;            // the table they were read for
        std::string error;
        uint64_t task = 0;
    };

    void RenderOperations();
    void RenderInputs();
    void RenderSecond();
    void RenderAnswer();
    void RenderTable(const linalg::Table& table);
    // The table picker, its number columns and (for A) the matrix to take; true when the choice changed.
    bool RenderTableInput(const char* id, TableInput& input, bool matrix_choice);
    void RenderRead(const linalg::TableMatrix& read, const char* title);
    // A typed grid; true when a value changed.
    bool RenderGrid(const char* id, linalg::Matrix& m, bool rows_fixed);
    void SetOp(linalg::Op op);
    void LoadExample();
    void Changed();
    void Export(const linalg::Table& table);

    bool visible_ = false;
    linalg::Op op_ = linalg::Op::SVD;
    int source_ = 0;            // 0 typed, 1 paste, 2 from a table
    int b_source_ = 0;          // 0 typed, 1 from a table
    TableInput a_table_, b_table_;
    std::vector<std::string> tables_;   // the session's tables, refreshed when the list opens
    int refresh_ = 0;           // Read again
    linalg::Matrix a_, b_;
    double k_ = 2.5;
    int low_rank_ = 1;
    bool edited_ = false;       // A was changed by hand: opening another operation keeps it
    std::string paste_;
    std::string paste_note_;
    bool paste_error_ = false;
    int random_seed_ = 1;

    std::string key_;           // the request's key (rebuilt when an input changes)
    bool dirty_ = true;
    linalg::Request request_;
    std::string stopped_key_;
    ExploreJob<Job> job_;
    std::shared_ptr<const Job> shown_;   // the last answer, kept while the next is worked out
    std::shared_ptr<int> alive_ = std::make_shared<int>(0);
    std::string shown_key_;
    int tab_ = 0;
    std::string message_;
    bool message_error_ = false;
};

}  // namespace cyxwiz
