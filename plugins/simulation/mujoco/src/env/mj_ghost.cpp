#include "mj_ghost.h"

namespace cyxwiz::plugin::mujoco {

MjGhost::~MjGhost() {
    if (data_) mj_deleteData(data_);
}

bool MjGhost::Fits(const mjModel* m) const {
    return data_ && m == model_ && m->nq == nq_ && m->nv == nv_ && m->nbody == nbody_;
}

bool MjGhost::Apply(const mjModel* m, const std::vector<mjtNum>& state) {
    if (!m || state.size() != static_cast<size_t>(mj_stateSize(m, mjSTATE_FULLPHYSICS))) return false;
    if (!Fits(m)) {
        if (data_) mj_deleteData(data_);
        data_ = mj_makeData(m);
        model_ = m;
        nq_ = m->nq;
        nv_ = m->nv;
        nbody_ = m->nbody;
        if (!data_) {
            state_.clear();
            return false;
        }
    }
    mj_setState(m, data_, state.data(), mjSTATE_FULLPHYSICS);
    mj_forward(m, data_);
    state_ = state;
    return true;
}

mjData* MjGhost::Data(const mjModel* m) const { return !state_.empty() && Fits(m) ? data_ : nullptr; }

}  // namespace cyxwiz::plugin::mujoco
