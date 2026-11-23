// The "Certificate authority" row under the server address: shown for an
// https:// server, its fingerprint as `openssl x509 -fingerprint -sha256`
// prints it, and a readable reason when the file is not a certificate.
#include "common/server_ca_view.h"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>

namespace {
int failures = 0;
void Check(bool ok, const char* what) {
    if (!ok) {
        std::printf("FAILED: %s\n", what);
        ++failures;
    }
}

// A self-signed EC P-256 certificate made for this test (CN "CyxWiz test
// fixture CA"); its fingerprint below is what openssl prints for it.
const char* kFixtureCa =
    "-----BEGIN CERTIFICATE-----\n"
    "MIIBuzCCAWGgAwIBAgIULtxdxWEDswP6VPXKwD90rlwBkx4wCgYIKoZIzj0EAwIw\n"
    "MjEPMA0GA1UECgwGQ3l4V2l6MR8wHQYDVQQDDBZDeXhXaXogdGVzdCBmaXh0dXJl\n"
    "IENBMCAXDTI2MTEwNjAzMTUzNVoYDzIxMjYxMDEzMDMxNTM1WjAyMQ8wDQYDVQQK\n"
    "DAZDeXhXaXoxHzAdBgNVBAMMFkN5eFdpeiB0ZXN0IGZpeHR1cmUgQ0EwWTATBgcq\n"
    "hkjOPQIBBggqhkjOPQMBBwNCAASg975RWB9jiBveuCytNZT7IiGb5SMBSZPriF7o\n"
    "DrgmgrgaQlmiBNqNdwOtg8Flo7fbsodcrsiiprVSWXpfIyB/o1MwUTAdBgNVHQ4E\n"
    "FgQUbiGkzOYRCcOLmVb7goalm7hYid0wHwYDVR0jBBgwFoAUbiGkzOYRCcOLmVb7\n"
    "goalm7hYid0wDwYDVR0TAQH/BAUwAwEB/zAKBggqhkjOPQQDAgNIADBFAiAyIt6x\n"
    "09XJ1Vm0BWD58kQP6axHGWBQ02HqKjZFZ4SshQIhAJpQuQ1wc9dvb5c2e3z5x+zZ\n"
    "alfI79Oszg8QLPiGUiTT\n"
    "-----END CERTIFICATE-----\n";
const char* kFixtureFingerprint =
    "00:9C:8E:68:78:B3:B9:92:DE:23:E9:58:FF:D2:A4:8F:DC:03:E8:58:09:CE:5E:35:DF:DE:3F:D2:97:B2:F1:16";
}  // namespace

int main() {
    using cyxwiz::protocol::CertificateFingerprint;
    using cyxwiz::protocol::DescribeServerCa;

    const auto dir = std::filesystem::temp_directory_path() / "cyxwiz_test_server_ca_view";
    std::filesystem::create_directories(dir);
    const std::string ca = (dir / "ca.crt").string();
    const std::string notes = (dir / "notes.txt").string();
    std::ofstream(ca, std::ios::binary) << kFixtureCa;
    std::ofstream(notes) << "not a certificate";

    std::string error;
    Check(CertificateFingerprint(ca, &error) == kFixtureFingerprint && error.empty(), "the fingerprint openssl prints");

    // Shown only where it can matter
    Check(DescribeServerCa("https://192.168.1.180", "").shown, "an https server shows the row");
    Check(!DescribeServerCa("http://192.168.1.180:8000", "").shown, "a plain server does not");
    Check(!DescribeServerCa("", "").shown, "no server, no row");

    // Empty: the system's roots
    auto v = DescribeServerCa("https://192.168.1.180", "  ");
    Check(v.empty && v.problem.empty() && v.fingerprint.empty(), "empty means the system's roots");
    Check(v.leave_empty.find("public certificate") != std::string::npos && v.trusts.empty(),
          "empty: only when to leave it empty");

    // A CA: its fingerprint, short on the surface and whole on hover
    v = DescribeServerCa("https://192.168.1.180", ca);
    Check(!v.empty && v.problem.empty() && v.fingerprint == kFixtureFingerprint, "a CA file: its fingerprint");
    Check(v.fingerprint_short == "00:9C:8E:68:...:F1:16", "the short form: first four, last two");
    Check(v.trusts.find("private CA") != std::string::npos && v.compare.find("new-certs.sh show") != std::string::npos &&
              !v.leave_empty.empty(),
          "set: what it trusts, where to compare, when to leave it empty");

    // Not a certificate, or no file: the reason, no fingerprint
    v = DescribeServerCa("https://192.168.1.180", notes);
    Check(!v.problem.empty() && v.fingerprint.empty(), "a file that is not a certificate is named");
    v = DescribeServerCa("https://192.168.1.180", (dir / "missing.crt").string());
    Check(!v.problem.empty(), "a missing file is named");

    // The screens' cache: the same view until the path or the file changes
    {
        using namespace std::chrono_literals;
        const std::string swapped = (dir / "swapped.crt").string();
        std::ofstream(swapped) << "not a certificate";
        cyxwiz::protocol::ServerCaViewCache slow(1h);
        Check(!slow.Get("https://192.168.1.180", swapped).problem.empty(), "cache: first look reads the file");
        std::ofstream(swapped, std::ios::binary) << kFixtureCa;
        std::filesystem::last_write_time(swapped, std::filesystem::last_write_time(swapped) + 1h);
        Check(!slow.Get("https://192.168.1.180", swapped).problem.empty(),
              "cache: not read again before the recheck time");
        Check(slow.Get("https://192.168.1.180", ca).fingerprint == kFixtureFingerprint,
              "cache: a new path is read at once");

        cyxwiz::protocol::ServerCaViewCache now(0ms);
        std::ofstream(swapped) << "not a certificate";
        std::filesystem::last_write_time(swapped, std::filesystem::last_write_time(swapped) + 2h);
        Check(!now.Get("https://192.168.1.180", swapped).problem.empty(), "cache: reads the file");
        std::ofstream(swapped, std::ios::binary) << kFixtureCa;
        std::filesystem::last_write_time(swapped, std::filesystem::last_write_time(swapped) + 3h);
        Check(now.Get("https://192.168.1.180", swapped).fingerprint == kFixtureFingerprint,
              "cache: a changed file is read again");
    }

    std::filesystem::remove_all(dir);
    if (failures == 0) std::printf("test_server_ca_view: all checks passed\n");
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
