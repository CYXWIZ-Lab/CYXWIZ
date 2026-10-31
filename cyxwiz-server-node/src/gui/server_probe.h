// server_probe.h - whether each service behind a typed server address answers
// (TOFIX168 step 3b). Settings > Network and the sign-in dialog ask it every
// frame; it probes off the UI thread, once the address has settled.
#pragma once

#include "common/server_address_view.h"

#include <chrono>
#include <future>
#include <string>

namespace cyxwiz::servernode::gui {

class ServerProbe {
public:
    // The field changed: wait for the typing to settle before asking.
    void Edited();
    // Ask about `address` when it is usable and not asked yet.
    void Update(const std::string& address, bool usable);
    // No server address: ask the separate addresses (central and sign-in).
    void UpdateSeparate(const std::string& central, const std::string& api_base_url);
    // Ask again (Check again).
    void Forget() { probed_.clear(); }
    // The answer for `address` (or SeparateKey): Checking while asking,
    // NotChecked before.
    cyxwiz::protocol::ServerReach ReachFor(const std::string& address) const;
    static std::string SeparateKey(const std::string& central, const std::string& api_base_url) {
        return "separate:" + central + "|" + api_base_url;
    }

private:
    void Start(const std::string& key, cyxwiz::protocol::ServerEndpoints endpoints, std::string address);
    using Clock = std::chrono::steady_clock;
    Clock::time_point edited_at_{};
    std::string probed_;
    std::string answered_;
    cyxwiz::protocol::ServerReach reach_;
    std::future<cyxwiz::protocol::ServerReach> future_;
};

}  // namespace cyxwiz::servernode::gui
