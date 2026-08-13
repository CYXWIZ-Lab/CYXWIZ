#include <catch2/catch_session.hpp>

#ifdef CYXWIZ_HAS_ARRAYFIRE
#include <arrayfire.h>
#endif

int main(int argc, char* argv[]) {
#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        // Keep the shared suite deterministic. oneAPI execution is qualified
        // by the isolated operation probe, where a failed event cannot poison
        // unrelated tests in this process.
        af::setBackend(AF_BACKEND_CPU);
        af::setDevice(0);
    } catch (const af::exception&) {
        return 2;
    }
#endif

    return Catch::Session().run(argc, argv);
}
