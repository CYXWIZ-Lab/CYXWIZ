#include "server_ca_view.h"

#include <openssl/err.h>
#include <openssl/evp.h>
#include <openssl/pem.h>
#include <openssl/x509.h>

#include <cstdio>
#include <fstream>
#include <sstream>

namespace cyxwiz::protocol {
namespace {

std::string Trim(const std::string& s) {
    const auto first = s.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) return {};
    const auto last = s.find_last_not_of(" \t\r\n");
    return s.substr(first, last - first + 1);
}

}  // namespace

std::string CertificateFingerprint(const std::string& pem_path, std::string* error) {
    std::ifstream in(pem_path, std::ios::binary);
    if (!in) {
        *error = "cannot read " + pem_path;
        return {};
    }
    std::ostringstream text;
    text << in.rdbuf();
    const std::string pem = text.str();

    BIO* bio = BIO_new_mem_buf(pem.data(), static_cast<int>(pem.size()));
    X509* cert = bio ? PEM_read_bio_X509(bio, nullptr, nullptr, nullptr) : nullptr;
    if (bio) BIO_free(bio);
    if (!cert) {
        ERR_clear_error();  // the parse failure stays queued on this thread otherwise
        *error = pem_path + " holds no PEM certificate (a CA file is the server's ca.crt)";
        return {};
    }
    unsigned char digest[EVP_MAX_MD_SIZE];
    unsigned int length = 0;
    const bool ok = X509_digest(cert, EVP_sha256(), digest, &length) == 1;
    X509_free(cert);
    if (!ok) {
        *error = "cannot compute the fingerprint of " + pem_path;
        return {};
    }
    std::string hex;
    char byte[4];
    for (unsigned int i = 0; i < length; ++i) {
        std::snprintf(byte, sizeof(byte), i == 0 ? "%02X" : ":%02X", digest[i]);
        hex += byte;
    }
    error->clear();
    return hex;
}

ServerCaView DescribeServerCa(const std::string& address, const std::string& ca_path) {
    ServerCaView v;
    v.shown = Trim(address).rfind("https://", 0) == 0;
    const std::string path = Trim(ca_path);
    v.empty = path.empty();
    v.leave_empty = "Leave empty for a server with a public certificate.";
    if (v.empty) return v;
    v.trusts = "Trusts this server's private CA (a test server's ca.crt).";
    v.fingerprint = CertificateFingerprint(path, &v.problem);
    if (!v.problem.empty()) return v;
    // "AA:BB:CC:DD:...:YY:ZZ": the first four and the last two groups
    v.fingerprint_short = v.fingerprint.size() > 18
                              ? v.fingerprint.substr(0, 11) + ":...:" + v.fingerprint.substr(v.fingerprint.size() - 5)
                              : v.fingerprint;
    v.compare = "compare with `new-certs.sh show`";
    return v;
}

const ServerCaView& ServerCaViewCache::Get(const std::string& address, const std::string& ca_path) {
    const auto now = std::chrono::steady_clock::now();
    std::error_code ec;
    if (!valid_ || address != address_ || ca_path != path_) {
        address_ = address;
        path_ = ca_path;
        stamp_ = std::filesystem::last_write_time(Trim(ca_path), ec);
        view_ = DescribeServerCa(address, ca_path);
        valid_ = true;
        checked_ = now;
    } else if (now - checked_ >= recheck_) {
        checked_ = now;
        const auto stamp = std::filesystem::last_write_time(Trim(ca_path), ec);
        if (stamp != stamp_) {
            stamp_ = stamp;
            view_ = DescribeServerCa(address, ca_path);
        }
    }
    return view_;
}

}  // namespace cyxwiz::protocol
