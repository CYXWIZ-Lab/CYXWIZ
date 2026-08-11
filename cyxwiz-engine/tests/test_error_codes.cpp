#include "../src/core/error_codes.h"

#include <cstdlib>
#include <iostream>
#include <set>
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

    Check(errors::DiagnosticCodeCatalog.size() == 66,
          "diagnostic catalog should expose every registered stable code");
    std::set<std::string> catalog_codes;
    for (const auto& descriptor : errors::DiagnosticCodeCatalog) {
        const std::string code = descriptor.code ? descriptor.code : "";
        const std::string name =
            descriptor.symbolic_name ? descriptor.symbolic_name : "";
        Check(code.size() == 9 && code.rfind("CW-", 0) == 0 &&
                  code[4] == '-',
              "catalog codes should use canonical CW-X-NNNN form");
        Check(!name.empty(), "catalog entries should have symbolic names");
        Check(catalog_codes.insert(code).second,
              "catalog codes should be unique");
    }
    const auto* descriptor =
        errors::FindDiagnosticCode(errors::Compiler::MissingTrainingPathNode);
    Check(descriptor &&
              std::string(descriptor->symbolic_name) ==
                  "Compiler.MissingTrainingPathNode",
          "catalog lookup should return the registered symbolic name");
    Check(errors::FindDiagnosticCode("CW-C-9999") == nullptr,
          "unregistered codes should not invent symbolic descriptions");
    Check(errors::DiagnosticFamilyName('G') == "gpu_backend" &&
              errors::DiagnosticFamilyName('P') == "native_cpu_backend",
          "GPU and native CPU diagnostic families should remain distinct");

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
