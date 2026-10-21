/**
 * ASan Memory Test
 *
 * This standalone test verifies AddressSanitizer is working correctly.
 * Run with: test_asan_memory [test_name]
 *
 * Tests:
 *   safe      - No memory errors (should pass)
 *   heap-oob  - Heap buffer overflow (ASan should catch)
 *   uaf       - Use after free (ASan should catch)
 *   stack-oob - Stack buffer overflow (ASan should catch)
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>

// Safe operations - no memory errors
void test_safe() {
    printf("Running safe memory test...\n");

    int* arr = new int[10];
    for (int i = 0; i < 10; i++) {
        arr[i] = i * i;
    }

    int sum = 0;
    for (int i = 0; i < 10; i++) {
        sum += arr[i];
    }

    delete[] arr;
    printf("Safe test passed! Sum = %d\n", sum);
}

// Heap buffer overflow - ASan should catch this
void test_heap_oob() {
    printf("Running heap-buffer-overflow test...\n");
    printf("ASan should report: heap-buffer-overflow\n");

    int* arr = new int[10];
    arr[10] = 42;  // Out of bounds write!
    delete[] arr;
}

// Use after free - ASan should catch this
void test_use_after_free() {
    printf("Running use-after-free test...\n");
    printf("ASan should report: heap-use-after-free\n");

    int* ptr = new int(42);
    delete ptr;
    int val = *ptr;  // Use after free!
    printf("Value: %d\n", val);
}

// Stack buffer overflow - ASan should catch this
void test_stack_oob() {
    printf("Running stack-buffer-overflow test...\n");
    printf("ASan should report: stack-buffer-overflow\n");

    int arr[10];
    arr[10] = 42;  // Out of bounds write!
}

void print_usage() {
    printf("ASan Memory Test\n");
    printf("================\n");
    printf("Usage: test_asan_memory [test_name]\n\n");
    printf("Available tests:\n");
    printf("  safe      - No memory errors (should pass)\n");
    printf("  heap-oob  - Heap buffer overflow (ASan catches)\n");
    printf("  uaf       - Use after free (ASan catches)\n");
    printf("  stack-oob - Stack buffer overflow (ASan catches)\n");
    printf("\nExample: test_asan_memory safe\n");
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        print_usage();
        return 0;
    }

    const char* test = argv[1];

    if (strcmp(test, "safe") == 0) {
        test_safe();
    } else if (strcmp(test, "heap-oob") == 0) {
        test_heap_oob();
    } else if (strcmp(test, "uaf") == 0) {
        test_use_after_free();
    } else if (strcmp(test, "stack-oob") == 0) {
        test_stack_oob();
    } else {
        printf("Unknown test: %s\n\n", test);
        print_usage();
        return 1;
    }

    return 0;
}
