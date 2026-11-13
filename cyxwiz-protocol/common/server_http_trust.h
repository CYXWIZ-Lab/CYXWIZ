#pragma once

// HTTPS clients to the server side verify it the same way the gRPC channels
// do (server_address.h CreateServerChannel): the server's CA when one is
// set (and only it), else the system's trusted roots; verification is never
// off. A template, so cyxwiz-protocol needs no HTTP library of its own:
// pass any cpp-httplib client (httplib::SSLClient, or httplib::Client made
// from an https:// URL), built with CPPHTTPLIB_OPENSSL_SUPPORT.

#include "server_address.h"

namespace cyxwiz::protocol {

template <class HttpsClient>
void TrustServer(HttpsClient& client) {
    client.enable_server_certificate_verification(true);
    const std::string ca = ServerCaFile();
    if (!ca.empty()) client.set_ca_cert_path(ca.c_str());
}

}  // namespace cyxwiz::protocol
