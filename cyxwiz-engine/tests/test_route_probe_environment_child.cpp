#include <cstdlib>
#include <iostream>
#include <string>

namespace {

const char* Value(const char* name) {
    const char* value = std::getenv(name);
    return value ? value : "<unset>";
}

}  // namespace

int main(int argc, char** argv) {
    if (argc == 3 && std::string(argv[1]) == "--enumerate-backend") {
        const std::string backend = argv[2];
        const std::string mode = Value("CYXWIZ_TEST_ROUTE_INVENTORY_MODE");
        std::cout << "probe_event schema=1 backend=" << backend
                  << " operation=route_inventory stage=enumeration_complete\n";
        if (mode == "unknown_field") {
            std::cout
                << "route_inventory_json={\"schema_version\":1,\"backend\":\""
                << backend
                << "\",\"routes\":[],\"unknown\":true}\n";
            return 0;
        }
        if (mode == "duplicate_id") {
            std::cout
                << "route_inventory_json={\"schema_version\":1,\"backend\":\""
                << backend
                << "\",\"routes\":["
                   "{\"device_id\":0,\"name\":\"First\",\"kind\":\"gpu\","
                   "\"identity_confidence\":\"backend_local\",\"provider\":null,"
                   "\"driver_version\":null,\"physical_fingerprint\":null,"
                   "\"metadata_status\":\"not_queried\",\"metadata_error_code\":0,"
                   "\"metadata_message\":null},"
                   "{\"device_id\":0,\"name\":\"Duplicate\",\"kind\":\"gpu\","
                   "\"identity_confidence\":\"backend_local\",\"provider\":null,"
                   "\"driver_version\":null,\"physical_fingerprint\":null,"
                   "\"metadata_status\":\"not_queried\",\"metadata_error_code\":0,"
                   "\"metadata_message\":null}]}\n";
            return 0;
        }
        std::cout
            << "route_inventory_json={\"schema_version\":1,\"backend\":\""
            << backend
            << "\",\"routes\":[{\"device_id\":0,\"name\":\"Fixture GPU\","
               "\"kind\":\"gpu\",\"identity_confidence\":\"stable_hardware\","
               "\"provider\":\"Fixture Provider\",\"driver_version\":\"1.2.3\","
               "\"physical_fingerprint\":\"pci:1234:5678:00:01.0\","
               "\"metadata_status\":\"available\",\"metadata_error_code\":0,"
               "\"metadata_message\":null}]}\n";
        return 0;
    }
    std::cout << "runtime_root=" << Value("CYXWIZ_ACTIVE_RUNTIME_ROOT")
              << '\n'
              << "runtime_set=" << Value("CYXWIZ_RUNTIME_SET_ID") << '\n'
              << "runtime_generation="
              << Value("CYXWIZ_RUNTIME_GENERATION") << '\n'
              << "base_pack=" << Value("CYXWIZ_BASE_PACK_ID") << '\n'
              << "opencl_pack=" << Value("CYXWIZ_RUNTIME_PACK_OPENCL")
              << '\n'
              << "af_path=" << Value("AF_PATH") << '\n'
              << "python_path=" << Value("PYTHONPATH") << '\n';
    return 0;
}
