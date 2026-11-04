// Tools > Linear Algebra (TOFIX129 D9): the window's answers against numpy
// (fixtures/generate_linear_algebra_presentation_fixtures.py): the result
// matrices, the figures, the singular values and eigenvalues, the refusals
// and their words, the check that the factors give A back; the operation
// list, pasted text, the numbers as shown, Copy and CSV, and a stop.

#include "../src/core/linear_algebra_presentation.h"
#include "computation_truth/test_device_selection.h"

#include <nlohmann/json.hpp>

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <set>
#include <string>

using namespace cyxwiz;
using nlohmann::json;

namespace {

int g_checks = 0;

void Check(bool condition, const std::string& message) {
    ++g_checks;
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(1);
    }
}

// float32 devices hold about 7 digits.
double Tolerance() { return 2e-4; }

void Near(double a, double b, const std::string& what) {
    Check(std::fabs(a - b) <= Tolerance() * std::max(1.0, std::fabs(b)), what + ": " + std::to_string(a) + " vs " + std::to_string(b));
}

linalg::Matrix ToMatrix(const json& j) {
    linalg::Matrix m;
    for (const auto& row : j) m.push_back(row.get<std::vector<double>>());
    return m;
}

void NearMatrix(const linalg::Matrix& got, const json& expected, const std::string& what) {
    const linalg::Matrix e = ToMatrix(expected);
    Check(got.size() == e.size() && (got.empty() || got[0].size() == e[0].size()), what + ": the size");
    for (size_t i = 0; i < e.size(); ++i)
        for (size_t j = 0; j < e[i].size(); ++j) Near(got[i][j], e[i][j], what + " [" + std::to_string(i) + "," + std::to_string(j) + "]");
}

std::string FigureOf(const linalg::Answer& a, const std::string& name) {
    for (const auto& f : a.figures)
        if (f.name == name) return f.value;
    return "";
}

// The largest difference a verify line reports.
double VerifiedDifference(const linalg::Answer& a) {
    const auto at = a.verify.rfind(' ');
    return at == std::string::npos ? NAN : std::atof(a.verify.c_str() + at + 1);
}

}  // namespace

int main() {
    if (!test::SelectTestDeviceFromEnvironment()) return 1;

    // ---- the operations
    std::set<std::string> ids;
    for (const auto& info : linalg::Ops()) {
        Check(ids.insert(info.id).second, std::string("one id per operation: ") + info.id);
        Check(linalg::FindOp(info.id) && linalg::FindOp(info.id)->op == info.op, std::string("FindOp: ") + info.id);
        Check(*info.hint != '\0', std::string("a hint: ") + info.id);
    }
    Check(linalg::Ops().size() == 17, "17 operations");
    Check(linalg::Groups().size() == 4 && linalg::Groups()[3] == "Factor", "four groups");

    std::ifstream in(CYXWIZ_LINEAR_ALGEBRA_PRESENTATION_FIXTURE);
    Check(in.good(), "cannot open the fixture");
    const json f = json::parse(in);
    for (const auto& c : f["cases"]) {
        const std::string name = c["name"];
        const linalg::OpInfo* info = linalg::FindOp(c["op"]);
        Check(info != nullptr, name + ": a known operation");
        linalg::Request r;
        r.op = info->op;
        r.a = ToMatrix(c["a"]);
        if (c.contains("b")) r.b = ToMatrix(c["b"]);
        if (c.contains("k")) r.k = c["k"];
        if (c.contains("low_rank")) r.low_rank = c["low_rank"];
        const linalg::Answer a = linalg::Run(r);
        const json& e = c["expected"];
        if (e.contains("error")) {
            Check(!a.ok(), name + ": refused");
            Check(a.error.find(e["error"].get<std::string>()) != std::string::npos, name + ": the reason \"" + a.error + "\"");
            continue;
        }
        Check(a.ok(), name + ": " + a.error);
        Check(!a.sentence.empty(), name + ": a sentence");
        if (e.contains("matrix")) {
            Check(!a.tabs.empty(), name + ": a result tab");
            const auto& tab = a.tabs[0];
            NearMatrix(tab.values, e["matrix"], name + ": the result");
        }
        if (e.contains("condition")) Near(std::atof(FigureOf(a, "condition").c_str()), e["condition"], name + ": condition");
        if (e.contains("det")) {
            const std::string det = FigureOf(a, "det");
            if (r.op == linalg::Op::LU) {
                // det is not a figure of LU; its check is the reconstruction.
            } else {
                Near(std::atof(det.c_str()), e["det"], name + ": det");
            }
        }
        if (e.contains("trace")) Near(std::atof(FigureOf(a, "trace").c_str()), e["trace"], name + ": trace");
        if (e.contains("rank")) {
            const int rank = e["rank"];
            const std::string shown = FigureOf(a, "rank");
            if (!shown.empty()) Check(shown.rfind(std::to_string(rank) + " of", 0) == 0, name + ": rank " + shown);
            else Check(a.sentence.find("rank " + std::to_string(rank)) != std::string::npos, name + ": rank in \"" + a.sentence + "\"");
        }
        if (e.contains("norm")) Near(std::atof(FigureOf(a, "‖A‖").c_str()), e["norm"], name + ": norm");
        if (e.contains("values")) {
            const auto values = e["values"].get<std::vector<double>>();
            Check(a.bars.size() == values.size(), name + ": a bar per value");
            for (size_t k = 0; k < values.size(); ++k) Near(a.bars[k].value, values[k], name + ": value " + std::to_string(k + 1));
        }
        if (e.contains("share"))
            for (size_t k = 0; k < e["share"].size(); ++k) Near(a.bars[k].share, e["share"][k], name + ": share " + std::to_string(k + 1));
        if (e.contains("low_rank_matrix")) {
            const auto& low = a.tabs.back();
            Check(low.title.rfind("Low rank", 0) == 0, name + ": the low rank tab");
            NearMatrix(low.values, e["low_rank_matrix"], name + ": low rank");
        }
        if (e.contains("abs_diag_r")) {
            const auto& R = a.tabs[1].values;
            for (size_t k = 0; k < e["abs_diag_r"].size(); ++k) Near(std::fabs(R[k][k]), e["abs_diag_r"][k], name + ": |R| diagonal");
        }
        if (e.contains("miss")) Near(std::atof(FigureOf(a, "‖A x - b‖").c_str()), e["miss"], name + ": the miss");
        if (a.verified && r.op != linalg::Op::LeastSquares)
            Check(VerifiedDifference(a) <= 1e-3 * std::max(1.0, std::fabs(r.a[0][0])) + 1e-3, name + ": " + a.verify);
    }

    // ---- words
    {
        linalg::Request r;
        r.op = linalg::Op::SVD;
        r.a = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}, {10, 11, 12}};
        const linalg::Answer a = linalg::Run(r);
        Check(a.sentence.find("rank 2 of 3") != std::string::npos && a.sentence.find("One direction holds 99.74 %") != std::string::npos,
              "svd: the sentence \"" + a.sentence + "\"");
        Check(FigureOf(a, "condition") == "∞ (singular)", "svd: a singular A's condition");
        Check(a.verified && VerifiedDifference(a) < 1e-3, "svd: U S V^T gives A back");
        Check(a.tabs.size() == 4 && a.tabs[1].title == "U (4 × 3)" && a.tabs[2].title == "V^T (3 × 3)", "svd: the tabs");
        r.op = linalg::Op::Cholesky;
        r.a = {{1, 2}, {2, 1}};
        const linalg::Answer c = linalg::Run(r);
        Check(c.try_instead == std::vector<std::string>({"eigen", "lu"}), "cholesky refused: points to Eigen and LU");
        r.op = linalg::Op::Solve;
        r.a = {{1, 2}, {2, 4}};
        r.b = {{5}, {6}};
        Check(linalg::Run(r).try_instead == std::vector<std::string>({"least_squares"}), "a singular solve points to least squares");
        r.op = linalg::Op::QR;
        r.a = {{12, -51, 4}, {6, 167, -68}, {-4, 24, -41}, {-1, 1, 0}};
        const linalg::Answer q = linalg::Run(r);
        Check(q.tabs[0].title == "Q (4 × 3)" && q.tabs[1].title == "R (3 × 3)", "qr: the reduced Q");
        Check(linalg::Run({linalg::Op::Add, {}, {}}).error == "A is empty.", "an empty A");
        Check(linalg::Run({linalg::Op::Add, {{1, 2}, {3}}, {}}).error == "A has rows of different lengths.", "a ragged A");
        Check(linalg::Run({linalg::Op::Rank, {{1, NAN}}, {}}).error == "A has a value that is not a number.", "a NaN");
    }

    // ---- a stop
    {
        linalg::Request r;
        r.op = linalg::Op::SVD;
        r.a = {{1, 2}, {3, 4}};
        Check(linalg::Run(r, [] { return true; }).error == "Cancelled.", "a stop ends the answer");
    }

    // ---- pasted text
    {
        linalg::Matrix m;
        std::string error;
        Check(linalg::ParseMatrix("1\t2\t3\r\n4\t5\t6\r\n", m, error) && m == linalg::Matrix({{1, 2, 3}, {4, 5, 6}}), "paste: tabs");
        Check(linalg::ParseMatrix("a,b\n1,2\n3,4", m, error) && m == linalg::Matrix({{1, 2}, {3, 4}}), "paste: a header line");
        Check(linalg::ParseMatrix(",x,y\nr1,1,2\nr2,3,4", m, error) && m == linalg::Matrix({{1, 2}, {3, 4}}), "paste: names of rows and columns");
        Check(linalg::ParseMatrix("1 2; 3 4", m, error) && m.size() == 1 && m[0].size() == 4, "paste: one line, spaces and semicolons");
        Check(!linalg::ParseMatrix("1 2\n3", m, error) && error == "Row 2 has 1 values, row 1 has 2.", "paste: rows of different lengths");
        Check(!linalg::ParseMatrix("1 x\n3 4", m, error) && error.find("\"x\" in row 1") != std::string::npos, "paste: a word: " + error);
        Check(!linalg::ParseMatrix("  \n", m, error), "paste: nothing");
        std::string big;
        for (int i = 0; i < 65; ++i) big += "1\n";
        Check(!linalg::ParseMatrix(big, m, error) && error.find("up to 64 × 64") != std::string::npos, "paste: too large");
    }

    // ---- numbers, Copy and CSV
    Check(linalg::Number(2.5) == "2.5000" && linalg::Number(-0.00001) == "-1.0000e-05" && linalg::Number(1e-12) == "0" &&
              linalg::Number(1e-12, 1e-11) == "1.0000e-12" && linalg::Number(2e7) == "2.0000e+07",
          "numbers as shown");
    {
        linalg::Request r;
        r.op = linalg::Op::Multiply;
        r.a = {{1, 2}, {3, 4}};
        r.b = {{5, 6}, {7, 8}};
        const linalg::Answer a = linalg::Run(r);
        Check(linalg::CopyText(a, 0).find("\tc1\tc2\nr1\t19\t22\nr2\t43\t50\n") != std::string::npos, "copy: tab-separated rows");
        Check(linalg::Csv(a.tabs[0]) == ",c1,c2\nr1,19,22\nr2,43,50\n", "csv: " + linalg::Csv(a.tabs[0]));
    }

    std::cout << "linear algebra presentation: " << g_checks << " checks against numpy. OK\n";
    return 0;
}
