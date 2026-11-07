#pragma once

// What a screen shows about the server's private CA (TOFIX192 T1c): the
// "Certificate authority (optional)" row under the server address, in the
// Engine's Preferences > Servers and the server node's Settings > Network.
// Both render this, so they word it the same way. No ImGui; it reads the
// CA file to show its fingerprint.

#include <string>

namespace cyxwiz::protocol {

struct ServerCaView {
    bool shown = false;              // the address is https://, so a CA can matter
    bool empty = true;               // no CA set: the system's trusted roots
    std::string fingerprint;         // SHA-256 of the certificate, "00:9C:...:F1:16"
    std::string fingerprint_short;   // "00:9C:8E:68:...:F1:16", the full one on hover
    std::string problem;             // why the file cannot be used ("" when it can)
    std::string trusts;              // what a set CA does ("" when none is set)
    std::string compare;             // where to check the fingerprint ("" without one)
    std::string leave_empty;         // when to leave it empty (always shown)
};

// The row for `ca_path` under the server address `address`.
ServerCaView DescribeServerCa(const std::string& address, const std::string& ca_path);

// The SHA-256 fingerprint of the first certificate in a PEM file, as
// `openssl x509 -noout -fingerprint -sha256` and new-certs.sh show print it;
// "" with `error` set when the file holds no readable certificate.
std::string CertificateFingerprint(const std::string& pem_path, std::string* error);

}  // namespace cyxwiz::protocol
