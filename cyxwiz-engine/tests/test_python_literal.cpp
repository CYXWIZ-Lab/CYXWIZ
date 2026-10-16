// Python string literal (TOFIX133 P0 item 16): Windows paths, quotes and
// newlines passed to the debugger stay data instead of becoming code.
#include "../src/core/python_literal.h"

#include <cstdlib>
#include <iostream>
#include <string>

namespace {
void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(1);
    }
}
}  // namespace

int main() {
    using cyxwiz::PythonStringLiteral;
    // \U would start a unicode escape and fail to compile in Python.
    Check(PythonStringLiteral(R"(C:\Users\me\train.py)") == R"('C:\\Users\\me\\train.py')", "Windows path");
    Check(PythonStringLiteral("it's") == R"('it\'s')", "single quote");
    // A newline would end the call and run the rest as a new statement.
    Check(PythonStringLiteral("x > 1')\nimport os\n#") == R"('x > 1\')\nimport os\n#')", "newline stays inside");
    Check(PythonStringLiteral(std::string("a\0b", 3)) == R"('a\x00b')", "control characters escaped");
    Check(PythonStringLiteral("caf\xC3\xA9") == "'caf\xC3\xA9'", "UTF-8 passes through");
    std::cout << "python literal: paths, quotes, newlines, control characters. OK\n";
    return 0;
}
