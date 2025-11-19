#include <iostream>
#include <cuda_runtime.h>

int main() {
    std::cout << "Direct CUDA Memory Query Test\n";
    std::cout << "==============================\n\n";

    // Get device count
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        std::cerr << "Failed to get device count: " << cudaGetErrorString(err) << "\n";
        return 1;
    }

    std::cout << "Found " << deviceCount << " CUDA device(s)\n\n";

    for (int i = 0; i < deviceCount; i++) {
        cudaSetDevice(i);

        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);

        std::cout << "Device " << i << ": " << prop.name << "\n";
        std::cout << "  Compute Capability: " << prop.major << "." << prop.minor << "\n";

        size_t free_bytes = 0, total_bytes = 0;
        err = cudaMemGetInfo(&free_bytes, &total_bytes);
        if (err == cudaSuccess) {
            double total_gb = total_bytes / (1024.0 * 1024.0 * 1024.0);
            double free_gb = free_bytes / (1024.0 * 1024.0 * 1024.0);
            double used_gb = total_gb - free_gb;

            std::cout << "  Total Memory: " << total_gb << " GB\n";
            std::cout << "  Free Memory:  " << free_gb << " GB\n";
            std::cout << "  Used Memory:  " << used_gb << " GB\n";
        } else {
            std::cerr << "  Failed to get memory info: " << cudaGetErrorString(err) << "\n";
        }
        std::cout << "\n";
    }

    return 0;
}
