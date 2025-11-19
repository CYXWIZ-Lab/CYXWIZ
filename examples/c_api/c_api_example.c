/**
 * Example: Using CyxWiz from C
 *
 * This demonstrates the pure C API (extern "C" interface)
 * to the CyxWiz backend library.
 *
 * Compile: gcc c_api_example.c -I../cyxwiz-backend/include -L../build/lib -lcyxwiz-backend -o example
 */

#include <cyxwiz/cyxwiz_c.h>
#include <stdio.h>
#include <stdlib.h>

int main(void) {
    printf("CyxWiz C API Example\n");
    printf("====================\n\n");

    // Initialize the backend
    if (!cyxwiz_initialize()) {
        printf("Error: Failed to initialize CyxWiz\n");
        printf("Error message: %s\n", cyxwiz_get_last_error());
        return 1;
    }

    printf("Version: %s\n\n", cyxwiz_get_version());

    // Check available devices
    int cpu_count = cyxwiz_device_get_count(CYXWIZ_DEVICE_CPU);
    int cuda_count = cyxwiz_device_get_count(CYXWIZ_DEVICE_CUDA);

    printf("Available devices:\n");
    printf("  CPU devices: %d\n", cpu_count);
    printf("  CUDA devices: %d\n", cuda_count);
    printf("\n");

    // Create a CPU device
    CyxWizDevice* device = cyxwiz_device_create(CYXWIZ_DEVICE_CPU, 0);
    if (!device) {
        printf("Error: Failed to create device\n");
        cyxwiz_shutdown();
        return 1;
    }

    cyxwiz_device_set_active(device);
    printf("Created and activated CPU device\n\n");

    // Create tensors
    size_t shape[] = {3, 4};  // 3x4 matrix
    CyxWizTensor* tensor_a = cyxwiz_tensor_zeros(shape, 2, CYXWIZ_DTYPE_FLOAT32);
    CyxWizTensor* tensor_b = cyxwiz_tensor_ones(shape, 2, CYXWIZ_DTYPE_FLOAT32);

    if (!tensor_a || !tensor_b) {
        printf("Error: Failed to create tensors\n");
        printf("Error message: %s\n", cyxwiz_get_last_error());
        cyxwiz_device_destroy(device);
        cyxwiz_shutdown();
        return 1;
    }

    printf("Created two tensors:\n");
    printf("  Tensor A: %zu x %zu (zeros)\n", shape[0], shape[1]);
    printf("  Tensor B: %zu x %zu (ones)\n", shape[0], shape[1]);
    printf("  Elements: %zu\n", cyxwiz_tensor_num_elements(tensor_a));
    printf("  Bytes: %zu\n", cyxwiz_tensor_num_bytes(tensor_a));
    printf("\n");

    // Perform tensor addition
    CyxWizTensor* tensor_c = cyxwiz_tensor_add(tensor_a, tensor_b);
    if (!tensor_c) {
        printf("Error: Failed to add tensors\n");
        printf("Error message: %s\n", cyxwiz_get_last_error());
    } else {
        printf("Successfully added tensors: A + B = C\n");

        // Access tensor data
        const float* data = (const float*)cyxwiz_tensor_data_const(tensor_c);
        printf("Result (first 5 elements): ");
        for (int i = 0; i < 5 && i < (int)cyxwiz_tensor_num_elements(tensor_c); i++) {
            printf("%.1f ", data[i]);
        }
        printf("\n\n");

        cyxwiz_tensor_destroy(tensor_c);
    }

    // Create an optimizer
    CyxWizOptimizer* optimizer = cyxwiz_optimizer_create(CYXWIZ_OPTIMIZER_ADAM, 0.001);
    if (optimizer) {
        printf("Created Adam optimizer\n");
        printf("  Learning rate: %.4f\n", cyxwiz_optimizer_get_learning_rate(optimizer));

        // Change learning rate
        cyxwiz_optimizer_set_learning_rate(optimizer, 0.01);
        printf("  Updated learning rate: %.4f\n", cyxwiz_optimizer_get_learning_rate(optimizer));
        printf("\n");

        cyxwiz_optimizer_destroy(optimizer);
    }

    // Memory statistics
    printf("Memory statistics:\n");
    printf("  Allocated: %zu bytes\n", cyxwiz_memory_get_allocated_bytes());
    printf("  Peak: %zu bytes\n", cyxwiz_memory_get_peak_bytes());
    printf("\n");

    // Cleanup
    cyxwiz_tensor_destroy(tensor_a);
    cyxwiz_tensor_destroy(tensor_b);
    cyxwiz_device_destroy(device);
    cyxwiz_shutdown();

    printf("Example completed successfully!\n");
    return 0;
}
