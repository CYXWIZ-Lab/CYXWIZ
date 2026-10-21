#include "backend_pack_metadata_verifier.h"
#include "installer_setup_embedded_trust.h"

#include <iostream>
#include <string>

int main() {
    const auto embedded = cyxwiz::runtime::EmbeddedInstallerTrustJson();
    if (embedded.empty()) {
        std::cerr << "embedded setup trust fixture is empty\n";
        return 1;
    }
    std::string error;
    const auto trust = cyxwiz::runtime::BackendPackTrustStore::LoadJson(
        embedded, error);
    const auto* key = trust ? trust->Find("embedded-test-key") : nullptr;
    if (!trust || !key || !key->installer || key->catalog || key->pack ||
        key->revoked) {
        std::cerr << (error.empty() ? "embedded installer role differs" : error)
                  << '\n';
        return 1;
    }
    return 0;
}
