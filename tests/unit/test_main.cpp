#include <catch2/catch_session.hpp>

#include <arrayfire.h>

int main(int argc, char* argv[]) {
    try {
        // Keep the shared suite deterministic. oneAPI execution is qualified
        // by the isolated operation probe, where a failed event cannot poison
        // unrelated tests in this process.
        af::setBackend(AF_BACKEND_CPU);
        af::setDevice(0);
    } catch (const af::exception&) {
        return 2;
    }

    return Catch::Session().run(argc, argv);
}
