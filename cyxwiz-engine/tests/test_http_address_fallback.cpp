// HTTP clients try every address a name resolves to. A server listening on
// 127.0.0.1 only must answer through "localhost", which Windows resolves to
// ::1 first: the second address is the one that connects. cmake/CyxWizHttplib
// keeps cpp-httplib's Windows non-blocking resolver out, which tried only the
// first address.
#include <httplib.h>

#include <cstdio>
#include <cstdlib>
#include <thread>

namespace {
int failures = 0;
void Check(bool ok, const char* what) {
    if (!ok) {
        std::printf("FAILED: %s\n", what);
        ++failures;
    }
}
}  // namespace

int main() {
#if defined(_WIN32) && defined(CPPHTTPLIB_USE_NON_BLOCKING_GETADDRINFO)
    Check(false, "the non-blocking resolver is off on Windows (it tries only the first address)");
#endif

    httplib::Server server;
    server.Get("/ping", [](const httplib::Request&, httplib::Response& res) { res.set_content("pong", "text/plain"); });
    const int port = server.bind_to_any_port("127.0.0.1");
    Check(port > 0, "the IPv4-only server binds a port");
    std::thread serving([&] { server.listen_after_bind(); });
    server.wait_until_ready();

    httplib::Client client("localhost", port);
    client.set_connection_timeout(5);
    client.set_read_timeout(5);
    const auto res = client.Get("/ping");
    Check(res && res->status == 200 && res->body == "pong",
          "localhost reaches a server on 127.0.0.1 only, after ::1 is refused");
    if (!res) std::printf("  error: %s\n", httplib::to_string(res.error()).c_str());

    server.stop();
    serving.join();
    if (failures == 0) std::printf("test_http_address_fallback: all checks passed\n");
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
