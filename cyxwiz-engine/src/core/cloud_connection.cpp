#include "cloud_connection.h"

#include <atomic>
#include <mutex>

namespace cyxwiz::cloud {

namespace {
std::atomic<network::DataStreamClient*> g_client{nullptr};
std::mutex g_error_mutex;
std::string g_error;
}  // namespace

void SetClient(network::DataStreamClient* client) {
    g_client.store(client);
    if (client) SetLastConnectError({});
}

network::DataStreamClient* Client() { return g_client.load(); }

void SetLastConnectError(const std::string& text) {
    std::lock_guard<std::mutex> lock(g_error_mutex);
    g_error = text;
}

std::string LastConnectError() {
    std::lock_guard<std::mutex> lock(g_error_mutex);
    return g_error;
}

}  // namespace cyxwiz::cloud
