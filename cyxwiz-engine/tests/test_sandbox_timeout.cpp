#include "../src/scripting/python_sandbox.h"
#include "../src/scripting/python_engine.h"
#include <iostream>
#include <chrono>

int main() {
    std::cout << "=== Python Sandbox Timeout Test ===" << std::endl;

    // Initialize Python
    std::cout << "\n1. Initializing Python interpreter..." << std::endl;
    scripting::PythonEngine engine;
    if (!engine.Initialize()) {
        std::cerr << "Failed to initialize Python!" << std::endl;
        return 1;
    }
    std::cout << "   Python initialized successfully" << std::endl;

    // Acquire GIL for sandbox operations (engine releases it after init)
    engine.AcquireGIL();

    // Create sandbox with 3-second timeout
    std::cout << "\n2. Creating sandbox with 3-second timeout..." << std::endl;
    scripting::PythonSandbox::Config config;
    config.timeout = std::chrono::seconds(3);
    config.max_memory_mb = 512;

    scripting::PythonSandbox sandbox(config);

    // Test 1: Simple code that completes quickly
    std::cout << "\n3. Test 1: Simple code (should succeed)..." << std::endl;
    {
        auto start = std::chrono::steady_clock::now();
        auto result = sandbox.Execute("x = 1 + 1\nprint(f'Result: {x}')");
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - start
        ).count();

        std::cout << "   Success: " << (result.success ? "YES" : "NO") << std::endl;
        std::cout << "   Output: " << result.output << std::endl;
        std::cout << "   Timeout exceeded: " << (result.timeout_exceeded ? "YES" : "NO") << std::endl;
        std::cout << "   Time: " << elapsed << "ms" << std::endl;
    }

    // Test 2: Infinite loop (should timeout)
    std::cout << "\n4. Test 2: Infinite loop (should timeout after 3s)..." << std::endl;
    {
        auto start = std::chrono::steady_clock::now();
        auto result = sandbox.Execute(R"(
print("Starting infinite loop...")
i = 0
while True:
    i += 1
    if i % 1000000 == 0:
        print(f"Iteration: {i}")
)");
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - start
        ).count();

        std::cout << "   Success: " << (result.success ? "YES" : "NO") << std::endl;
        std::cout << "   Timeout exceeded: " << (result.timeout_exceeded ? "YES" : "NO") << std::endl;
        std::cout << "   Error: " << result.error_message << std::endl;
        std::cout << "   Output (partial): " << result.output.substr(0, 200) << std::endl;
        std::cout << "   Time: " << elapsed << "ms" << std::endl;

        if (result.timeout_exceeded && elapsed >= 2900 && elapsed <= 4000) {
            std::cout << "   PASS: Timeout worked correctly!" << std::endl;
        } else {
            std::cout << "   FAIL: Timeout did not work as expected" << std::endl;
        }
    }

    // Test 3: Long-running computation (should timeout)
    std::cout << "\n5. Test 3: Heavy computation (should timeout after 3s)..." << std::endl;
    {
        auto start = std::chrono::steady_clock::now();
        auto result = sandbox.Execute(R"(
print("Starting heavy computation...")
result = 0
for i in range(10**12):  # Will never finish
    result += i
    if i % 10000000 == 0:
        print(f"Progress: {i}")
print(f"Final result: {result}")
)");
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - start
        ).count();

        std::cout << "   Success: " << (result.success ? "YES" : "NO") << std::endl;
        std::cout << "   Timeout exceeded: " << (result.timeout_exceeded ? "YES" : "NO") << std::endl;
        std::cout << "   Error: " << result.error_message << std::endl;
        std::cout << "   Time: " << elapsed << "ms" << std::endl;

        if (result.timeout_exceeded && elapsed >= 2900 && elapsed <= 4000) {
            std::cout << "   PASS: Timeout worked correctly!" << std::endl;
        } else {
            std::cout << "   FAIL: Timeout did not work as expected" << std::endl;
        }
    }

    std::cout << "\n=== Tests Complete ===" << std::endl;

    // Release GIL before shutdown
    engine.ReleaseGIL();

    return 0;
}
