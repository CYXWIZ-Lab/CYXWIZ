#include "rl_common.h"

#include <cyxwiz/fused_mlp.h>
#include <cyxwiz/sequential.h>

namespace cyxwiz::rl {

std::unique_ptr<FusedMlp> FuseNetwork(IExecutableModel& model, std::string& why) {
    SequentialModel* sequential = model.AsSequentialModel();
    if (!sequential) {
        why = "not a sequential network";
        return nullptr;
    }
    return FusedMlp::For(*sequential, why);
}

}  // namespace cyxwiz::rl
