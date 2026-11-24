#include "replay_buffer.h"

#include "algorithms/device_random.h"

#include <arrayfire.h>

#include <algorithm>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

namespace cyxwiz::rl {

// Columns: observation | action | reward | next observation | done
struct ReplayBuffer::Table {
    af::array rows;  // [capacity, width] f32
};

namespace {

size_t Width(size_t obs, size_t act) { return 2 * obs + act + 2; }

std::string Megabytes(uint64_t bytes) {
    char text[32];
    std::snprintf(text, sizeof(text), "%.0f MB", static_cast<double>(bytes) / 1e6);
    return text;
}

}  // namespace

ReplayBuffer::ReplayBuffer(int64_t capacity, size_t observation_size, size_t action_size,
                           const std::string& learner)
    : table_(std::make_unique<Table>()), capacity_(capacity), observation_size_(observation_size),
      action_size_(action_size) {
    if (capacity < 1) throw std::invalid_argument("The replay buffer needs room for at least one step.");
    if (observation_size < 1 || action_size < 1) throw std::invalid_argument("The replay buffer needs observations and actions.");
    try {
        table_->rows = af::constant(0.0f, static_cast<dim_t>(capacity),
                                    static_cast<dim_t>(Width(observation_size, action_size)), f32);
        table_->rows.eval();
        af::sync();
    } catch (const std::exception& e) {  // af::exception, or the host's bad_alloc on the CPU device
        throw std::runtime_error("The replay buffer of " + std::to_string(capacity) + " steps (" + Megabytes(Bytes()) +
                                 ") does not fit on the training device (" + e.what() +
                                 "). Lower Buffer size on the " + learner + " node.");
    }
}

ReplayBuffer::~ReplayBuffer() = default;

uint64_t ReplayBuffer::Bytes() const { return BytesFor(capacity_, observation_size_, action_size_); }

uint64_t ReplayBuffer::BytesFor(int64_t capacity, size_t observation_size, size_t action_size) {
    if (capacity <= 0) return 0;
    return static_cast<uint64_t>(capacity) * Width(observation_size, action_size) * sizeof(float);
}

void ReplayBuffer::Add(const float* observations, const float* actions, const float* rewards,
                       const float* next_observations, const uint8_t* terminated, size_t rows) {
    if (rows == 0) return;
    const size_t obs = observation_size_, act = action_size_, width = Width(obs, act);
    // Only the last `capacity` of a batch larger than the buffer survive
    const size_t skip = rows > static_cast<size_t>(capacity_) ? rows - static_cast<size_t>(capacity_) : 0;
    const size_t count = rows - skip;
    // Column-major for ArrayFire: column c of row r at c * count + r
    std::vector<float> block(count * width);
    for (size_t k = 0; k < count; ++k) {
        const size_t r = skip + k;
        size_t c = 0;
        const auto put = [&](float v) { block[c++ * count + k] = v; };
        for (size_t i = 0; i < obs; ++i) put(observations[r * obs + i]);
        for (size_t i = 0; i < act; ++i) put(actions[r * act + i]);
        put(rewards[r]);
        for (size_t i = 0; i < obs; ++i) put(next_observations[r * obs + i]);
        put(terminated[r] ? 1.0f : 0.0f);
    }
    const af::array device(static_cast<dim_t>(count), static_cast<dim_t>(width), block.data());
    // Up to the end of the table, then from its start
    const size_t first = std::min(count, static_cast<size_t>(capacity_ - next_));
    table_->rows(af::seq(static_cast<double>(next_), static_cast<double>(next_ + first) - 1.0), af::span) =
        device(af::seq(0.0, static_cast<double>(first) - 1.0), af::span);
    if (first < count)
        table_->rows(af::seq(0.0, static_cast<double>(count - first) - 1.0), af::span) =
            device(af::seq(static_cast<double>(first), static_cast<double>(count) - 1.0), af::span);
    table_->rows.eval();
    next_ = static_cast<int64_t>((static_cast<size_t>(next_) + count) % static_cast<size_t>(capacity_));
    size_ = std::min<int64_t>(capacity_, size_ + static_cast<int64_t>(count));
}

ReplayBuffer::Batch ReplayBuffer::Sample(size_t batch, uint64_t seed) const {
    if (size_ < 1) throw std::logic_error("The replay buffer is empty.");
    if (batch < 1) throw std::invalid_argument("A replay batch needs at least one row.");
    // Uniform rows in [0, size): 32-bit Philox draws (the same on every
    // device) modulo the size; the bias is below size / 2^32
    af::randomEngine engine = cyxwiz::DeviceRandomEngine(seed);
    const af::array draws = af::randu(af::dim4(static_cast<dim_t>(batch)), u32, engine);
    const af::array index = draws % static_cast<unsigned>(size_);
    af::array rows = af::lookup(table_->rows, index, 0);
    rows.eval();
    const size_t obs = observation_size_, act = action_size_;
    const auto columns = [&](size_t from, size_t count) {
        af::array part = rows(af::span, af::seq(static_cast<double>(from), static_cast<double>(from + count) - 1.0));
        part.eval();
        return Tensor::FromSemanticArray(part, {batch, count});
    };
    Batch out;
    out.observations = columns(0, obs);
    out.actions = columns(obs, act);
    out.rewards = columns(obs + act, 1);
    out.next_observations = columns(obs + act + 1, obs);
    out.dones = columns(2 * obs + act + 1, 1);
    return out;
}

}  // namespace cyxwiz::rl
