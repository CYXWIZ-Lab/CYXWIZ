// server_probe.cpp
#include "gui/server_probe.h"

#include "auth/auth_manager.h"
#include "common/server_address_probe.h"

namespace cyxwiz::servernode::gui {

namespace proto = cyxwiz::protocol;

void ServerProbe::Edited() { edited_at_ = Clock::now(); }

void ServerProbe::Update(const std::string& address, bool usable) {
    if (future_.valid() && future_.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
        reach_ = future_.get();
        answered_ = probed_;
    }
    if (!usable) return;
    Start(address, {}, address);
}

void ServerProbe::UpdateSeparate(const std::string& central, const std::string& api_base_url) {
    if (future_.valid() && future_.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
        reach_ = future_.get();
        answered_ = probed_;
    }
    proto::ServerEndpoints endpoints;
    endpoints.central = central;
    endpoints.api_base_url = api_base_url;
    Start(SeparateKey(central, api_base_url), endpoints, {});
}

void ServerProbe::Start(const std::string& key, proto::ServerEndpoints endpoints, std::string address) {
    if (key == probed_ || future_.valid()) return;
    // Probe 600 ms after the last keystroke.
    if (Clock::now() - edited_at_ < std::chrono::milliseconds(600)) return;
    probed_ = key;
    future_ = std::async(std::launch::async, [endpoints, address]() {
        const proto::HttpGetFn get = [](const std::string& url, std::string* body) {
            return auth::AuthManager::HttpGet(url, body);
        };
        return address.empty() ? proto::ProbeEndpoints(endpoints, proto::ServiceSet::NodeServices, get)
                               : proto::ProbeServer(address, proto::ServiceSet::NodeServices, get);
    });
}

proto::ServerReach ServerProbe::ReachFor(const std::string& address) const {
    const bool asking = future_.valid() && address == probed_;
    if (address == answered_ && !asking) return reach_;
    proto::ServerReach reach;
    if (asking) {
        reach.central = reach.gateway = reach.api = proto::ServiceReach::Checking;
    }
    return reach;
}

}  // namespace cyxwiz::servernode::gui
