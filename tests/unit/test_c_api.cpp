#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <cyxwiz/cyxwiz_c.h>
#include <string>

TEST_CASE("C API tensor matmul returns matrix product", "[c_api]") {
    const size_t shape_a[] = {2, 3};
    const size_t shape_b[] = {3, 2};
    const float data_a[] = {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f
    };
    const float data_b[] = {
        7.0f, 8.0f,
        9.0f, 10.0f,
        11.0f, 12.0f
    };

    CyxWizTensor* a = cyxwiz_tensor_create_with_data(
        shape_a, 2, data_a, CYXWIZ_DTYPE_FLOAT32);
    CyxWizTensor* b = cyxwiz_tensor_create_with_data(
        shape_b, 2, data_b, CYXWIZ_DTYPE_FLOAT32);

    REQUIRE(a != nullptr);
    REQUIRE(b != nullptr);

    CyxWizTensor* c = cyxwiz_tensor_matmul(a, b);
    REQUIRE(c != nullptr);
    REQUIRE(cyxwiz_tensor_num_dimensions(c) == 2);

    size_t shape_c[2] = {};
    cyxwiz_tensor_get_shape(c, shape_c);
    REQUIRE(shape_c[0] == 2);
    REQUIRE(shape_c[1] == 2);

    const float* out = static_cast<const float*>(cyxwiz_tensor_data_const(c));
    REQUIRE(out != nullptr);
    REQUIRE(out[0] == Catch::Approx(58.0f));
    REQUIRE(out[1] == Catch::Approx(64.0f));
    REQUIRE(out[2] == Catch::Approx(139.0f));
    REQUIRE(out[3] == Catch::Approx(154.0f));

    cyxwiz_tensor_destroy(c);
    cyxwiz_tensor_destroy(b);
    cyxwiz_tensor_destroy(a);
}

TEST_CASE("C API tensor matmul reports invalid shapes", "[c_api]") {
    const size_t shape_a[] = {2, 3};
    const size_t shape_b[] = {2, 2};
    const float data_a[] = {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f
    };
    const float data_b[] = {
        1.0f, 2.0f,
        3.0f, 4.0f
    };

    CyxWizTensor* a = cyxwiz_tensor_create_with_data(
        shape_a, 2, data_a, CYXWIZ_DTYPE_FLOAT32);
    CyxWizTensor* b = cyxwiz_tensor_create_with_data(
        shape_b, 2, data_b, CYXWIZ_DTYPE_FLOAT32);

    REQUIRE(a != nullptr);
    REQUIRE(b != nullptr);

    cyxwiz_clear_last_error();
    CyxWizTensor* c = cyxwiz_tensor_matmul(a, b);
    REQUIRE(c == nullptr);

    std::string error = cyxwiz_get_last_error();
    REQUIRE(error.find("columns") != std::string::npos);

    cyxwiz_tensor_destroy(b);
    cyxwiz_tensor_destroy(a);
}

TEST_CASE("C API tensor factories validate public inputs", "[c_api]") {
    cyxwiz_clear_last_error();
    CyxWizTensor* missing_shape = cyxwiz_tensor_create(nullptr, 2, CYXWIZ_DTYPE_FLOAT32);
    REQUIRE(missing_shape == nullptr);
    std::string shape_error = cyxwiz_get_last_error();
    REQUIRE(shape_error.find("shape") != std::string::npos);

    const size_t shape[] = {2, 2};

    cyxwiz_clear_last_error();
    CyxWizTensor* missing_data = cyxwiz_tensor_create_with_data(
        shape, 2, nullptr, CYXWIZ_DTYPE_FLOAT32);
    REQUIRE(missing_data == nullptr);
    std::string data_error = cyxwiz_get_last_error();
    REQUIRE(data_error.find("data") != std::string::npos);

    cyxwiz_clear_last_error();
    CyxWizTensor* invalid_dtype = cyxwiz_tensor_zeros(
        shape, 2, static_cast<CyxWizDataType>(999));
    REQUIRE(invalid_dtype == nullptr);
    std::string dtype_error = cyxwiz_get_last_error();
    REQUIRE(dtype_error.find("data type") != std::string::npos);

    cyxwiz_clear_last_error();
    CyxWizTensor* scalar = cyxwiz_tensor_create(nullptr, 0, CYXWIZ_DTYPE_FLOAT32);
    REQUIRE(scalar != nullptr);
    REQUIRE(cyxwiz_tensor_num_dimensions(scalar) == 0);
    cyxwiz_tensor_destroy(scalar);
}
