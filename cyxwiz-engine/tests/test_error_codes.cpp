#include "../src/core/error_codes.h"

#include <cstdlib>
#include <iostream>
#include <string>

namespace {

void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << "\n";
        std::exit(1);
    }
}

} // namespace

int main() {
    using namespace cyxwiz;

    Check(std::string(errors::Compiler::MissingTrainingPathNode) == "CW-C-0101",
          "compiler code should be stable");
    Check(std::string(errors::Compiler::GenericIssue) == "CW-C-0001",
          "generic compiler issue code should be stable");
    Check(std::string(errors::Runtime::UnsupportedNode) == "CW-R-0201",
          "runtime unsupported-node code should be stable");
    Check(std::string(errors::Data::RequiredColumnMissing) == "CW-D-0101",
          "data missing-column code should be stable");
    Check(std::string(errors::Data::VocabularyCoverageWarning) == "CW-D-0304",
          "data vocabulary-coverage code should be stable");
    Check(std::string(errors::Gpu::KernelExecutionFailed) == "CW-G-0501",
          "GPU kernel failure code should be stable");

    const std::string formatted = errors::FormatError(
        errors::Runtime::UnsupportedNode,
        "Node type 'ExportSQL' is not supported",
        "SQL database export is not implemented",
        "Use DataConvert or DataOutput for supported table exports");

    Check(formatted ==
              "[CW-R-0201] Node type 'ExportSQL' is not supported. "
              "Detail: SQL database export is not implemented. "
              "Hint: Use DataConvert or DataOutput for supported table exports",
          "formatted error should include code, detail, and hint");

    Check(errors::FormatError(errors::Runtime::ExecutionFailed,
                              "[CW-D-0101] required column missing") ==
              "[CW-D-0101] required column missing",
          "formatter should not double-prefix coded messages");

    std::cout << "test_error_codes passed\n";
    return 0;
}
