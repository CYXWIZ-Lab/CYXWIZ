#pragma once

// The jobs that run the active model on their own thread take turns with
// Train and Test: Explain, Feature Importance and generation from this
// session's model. The model's modules keep each forward pass's inputs for
// the backward pass, so two passes at once would mix them. A job holds a
// claim while it runs; the claim ends when the last copy is released.

#include <memory>
#include <string>

namespace cyxwiz {

class ActiveModelUse {
public:
    ~ActiveModelUse();
    ActiveModelUse(const ActiveModelUse&) = delete;
    ActiveModelUse& operator=(const ActiveModelUse&) = delete;

    // A claim for `what` ("Explain", ...), or null with `why` set when Train,
    // a test or another job has the model.
    static std::shared_ptr<ActiveModelUse> TryClaim(const std::string& what, bool training, bool testing,
                                                    std::string* why);

private:
    ActiveModelUse() = default;
};

// Why another job cannot run the active model now, or "" when it is free.
std::string ActiveModelBusy(bool training, bool testing);

}  // namespace cyxwiz
