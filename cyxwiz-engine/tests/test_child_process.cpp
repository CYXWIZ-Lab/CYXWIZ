// RunHidden (core/child_process): the Python checks and venv creation start
// their programs with it, without a shell or a console window. The test runs
// itself as the child (--echo-args prints its arguments, one per line), so
// the quoting is checked on every platform without Python.

#include "../src/core/child_process.h"

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

namespace {

int failures = 0;

void Check(bool ok, const std::string& what) {
    if (!ok) {
        std::cerr << "FAIL: " << what << "\n";
        ++failures;
    }
}

std::string WithoutCarriageReturns(std::string text) {
    std::string out;
    for (char c : text)
        if (c != '\r') out += c;
    return out;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc >= 2 && std::strcmp(argv[1], "--echo-args") == 0) {
        for (int i = 2; i < argc; ++i) std::cout << "[" << argv[i] << "]\n";
        std::cerr << "this goes to stderr\n";
        return 0;
    }
    if (argc >= 3 && std::strcmp(argv[1], "--exit") == 0) return std::atoi(argv[2]);

    const std::string self = argv[0];
    const std::vector<std::string> awkward = {"plain", "with space", "with \"quote\"", "trailing\\",
                                              "C:\\Program Files\\Python312\\", "", "a\\\\\"b"};
    std::vector<std::string> args = {self, "--echo-args"};
    args.insert(args.end(), awkward.begin(), awkward.end());
    std::string output;
    const int code = cyxwiz::core::RunHidden(args, &output);
    std::string expected;
    for (const auto& arg : awkward) expected += "[" + arg + "]\n";
    Check(code == 0, "the child exits 0, got " + std::to_string(code));
    Check(WithoutCarriageReturns(output) == expected,
          "every argument arrives as it was given, stdout only; got:\n" + output);

    Check(cyxwiz::core::RunHidden({self, "--exit", "3"}) == 3, "the exit code comes back");
    Check(cyxwiz::core::RunHidden({"cyxwiz-no-such-program-for-this-test"}) == -1,
          "a program that does not exist gives -1");
    Check(cyxwiz::core::RunHidden({}) == -1, "no program gives -1");

    if (failures) return 1;
    std::cout << "child process: arguments, stdout, exit code, missing program. OK\n";
    return 0;
}
