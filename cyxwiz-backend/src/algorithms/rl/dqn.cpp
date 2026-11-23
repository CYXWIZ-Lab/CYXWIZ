#include "cyxwiz/rl/dqn.h"

#include "../arrayfire_indexed.h"

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include <arrayfire.h>

// Windows headers define these; ArrayFire's min/max must win.
#ifdef max
#undef max
#endif
#ifdef min
#undef min
#endif

namespace cyxwiz::rl {

namespace {

af::array Array(const Tensor& t) { return t.GetSemanticArray(); }

Tensor Make(const af::array& a, std::vector<size_t> shape) {
    af::array evaluated = a;
    evaluated.eval();
    return Tensor::FromSemanticArray(evaluated, std::move(shape));
}

void Require(bool ok, const std::string& what) {
    if (!ok) throw std::invalid_argument(what);
}

// B values ([B] or [B, 1]) as an ArrayFire column of B rows
af::array Column(const Tensor& t, size_t rows, const char* name) {
    Require(t.NumElements() == rows,
            std::string("DQN: ") + name + " needs " + std::to_string(rows) + " values, got " +
                std::to_string(t.NumElements()));
    return af::moddims(Array(t), static_cast<dim_t>(rows));
}

// The Q values [B, A]: B and A
void RequireQ(const Tensor& q, const char* name, size_t& rows, size_t& choices) {
    Require(q.Shape().size() == 2 && q.Shape()[0] > 0 && q.Shape()[1] > 0,
            std::string("DQN: ") + name + " must be [B, A] with at least one row and one choice");
    rows = q.Shape()[0];
    choices = q.Shape()[1];
}

}  // namespace

Tensor DqnTarget(const Tensor& next_q, const Tensor& rewards, const Tensor& dones, float gamma) {
    size_t rows = 0, choices = 0;
    RequireQ(next_q, "the next Q values", rows, choices);
    const af::array best = af::moddims(af::max(Array(next_q), 1), static_cast<dim_t>(rows));
    const af::array y = Column(rewards, rows, "rewards") + (1.0f - Column(dones, rows, "dones")) * gamma * best;
    return Make(af::moddims(y, static_cast<dim_t>(rows), 1), {rows, 1});
}

DqnLoss DqnLossAndGradient(const Tensor& q, const Tensor& actions, const Tensor& target) {
    size_t rows = 0, choices = 0;
    RequireQ(q, "the Q values", rows, choices);
    const dim_t b = static_cast<dim_t>(rows);
    const dim_t a = static_cast<dim_t>(choices);
    // Each row's chosen column: 1 there, 0 elsewhere
    const af::array chosen_index = af::tile(af::moddims(Column(actions, rows, "actions"), b, 1), 1, static_cast<unsigned>(a));
    const af::array columns = af::tile(af::moddims(af::range(af::dim4(a), 0, f32), 1, a), static_cast<unsigned>(b), 1);
    const af::array mask = (columns == chosen_index).as(f32);
    const af::array chosen = af::moddims(af::sum(Array(q) * mask, 1), b);
    const af::array d = chosen - Column(target, rows, "the target");
    const af::array inside = (af::abs(d) < 1.0f).as(f32);
    const af::array per_row = inside * 0.5f * d * d + (1.0f - inside) * (af::abs(d) - 0.5f);
    const float n = static_cast<float>(rows);
    DqnLoss out;
    out.loss = Make(af::sum(per_row) / n, {1});
    out.mean_q = Make(af::sum(chosen) / n, {1});
    const af::array grad_chosen = af::clamp(d, -1.0, 1.0) / n;
    out.grad_q = Make(mask * af::tile(af::moddims(grad_chosen, b, 1), 1, static_cast<unsigned>(a)), {rows, choices});
    return out;
}

Tensor DqnGreedyActions(const Tensor& q) {
    size_t rows = 0, choices = 0;
    RequireQ(q, "the Q values", rows, choices);
    af::array best, index;
    MaxWithIndex(best, index, Array(q), 1);
    return Make(af::moddims(index.as(f32), static_cast<dim_t>(rows)), {rows});
}

float DqnExplorationRate(float initial_eps, float final_eps, float fraction, double progress_remaining) {
    const double done = 1.0 - progress_remaining;
    if (!(fraction > 0.0f) || done > fraction) return final_eps;
    return static_cast<float>(initial_eps + done * (final_eps - initial_eps) / fraction);
}

}  // namespace cyxwiz::rl
