#include <cstdlib>
#include <iostream>

namespace {

const char* Value(const char* name) {
    const char* value = std::getenv(name);
    return value ? value : "<unset>";
}

}  // namespace

int main() {
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
