#include "tensor/tensor.h"
#include <chrono>
#include <gtest/gtest.h>

class TensorTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Setup code if needed
    }
};

// Basic tensor operations
TEST_F(TensorTest, BasicTensorCreation)
{
    dlf::Tensor<int> tensor({2, 3}, 1);
    auto& shape = tensor.shape();
    EXPECT_EQ(shape[0], 2);
    EXPECT_EQ(shape[1], 3);

    // Verify all elements are initialized to 1
    for (int i = 0; i < 2; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            EXPECT_EQ(tensor.view(i).view(j), 1);
        }
    }
}

TEST_F(TensorTest, TensorSize)
{
    dlf::Tensor<int> tensor({2, 3}, 1);
    EXPECT_EQ(tensor.size(), 6);

    // Test with different dimensions
    dlf::Tensor<int> tensor3d({2, 3, 4}, 1);
    EXPECT_EQ(tensor3d.size(), 24);
}

TEST_F(TensorTest, TensorEmpty)
{
    dlf::Tensor<int> tensor({1, 1}, 1);
    EXPECT_FALSE(tensor.empty());

    // Test with empty tensor (not possible with our implementation)
    // dlf::Tensor<int> empty_tensor({0, 0}, 1); // This would throw an exception
}

TEST_F(TensorTest, TensorStrides)
{
    dlf::Tensor<int> tensor({2, 3}, 1);
    auto strides = tensor.strides();
    EXPECT_EQ(strides[0], 3);
    EXPECT_EQ(strides[1], 1);

    // Test with 3D tensor
    dlf::Tensor<int> tensor3d({2, 3, 4}, 1);
    auto strides3d = tensor3d.strides();
    EXPECT_EQ(strides3d[0], 12); // 3 * 4
    EXPECT_EQ(strides3d[1], 4);  // 4
    EXPECT_EQ(strides3d[2], 1);  // 1
}

// Tensor operations
TEST_F(TensorTest, TensorReshape)
{
    dlf::Tensor<int> tensor({2, 3}, 1);
    tensor.reshape({3, 2});
    auto& shape = tensor.shape();
    EXPECT_EQ(shape[0], 3);
    EXPECT_EQ(shape[1], 2);

    // Test invalid reshape
    EXPECT_THROW(tensor.reshape({4, 2}), std::invalid_argument);
}

TEST_F(TensorTest, TensorTransform)
{
    dlf::Tensor<int> tensor({2, 3}, 1);
    tensor.transform([](const int& x) { return x + 1; });

    // Verify all elements are incremented
    for (int i = 0; i < 2; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            EXPECT_EQ(tensor.view(i).view(j), 2);
        }
    }
}

TEST_F(TensorTest, TensorPermute)
{
    dlf::Tensor<int> tensor({2, 3}, 1);
    auto permuted_shape = tensor.permute({1, 0});
    EXPECT_EQ(permuted_shape[0], 3);
    EXPECT_EQ(permuted_shape[1], 2);

    // Test with 3D tensor
    dlf::Tensor<int> tensor3d({2, 3, 4}, 1);
    auto permuted_shape3d = tensor3d.permute({2, 0, 1});
    EXPECT_EQ(permuted_shape3d[0], 4);
    EXPECT_EQ(permuted_shape3d[1], 2);
    EXPECT_EQ(permuted_shape3d[2], 3);
}

// Error handling
TEST_F(TensorTest, TensorOutOfRange)
{
    dlf::Tensor<int> tensor({2, 3}, 1);
    EXPECT_THROW(tensor.view(2).view(0), std::out_of_range);
    EXPECT_THROW(tensor.view(0).view(3), std::out_of_range);
}

// Tensor operations with different data types
TEST_F(TensorTest, TensorFloatOperations)
{
    dlf::Tensor<float> tensor({2, 3}, 1.0f);
    tensor.transform([](const float& x) { return x * 2.0f; });

    for (int i = 0; i < 2; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            EXPECT_FLOAT_EQ(tensor.view(i).view(j), 2.0f);
        }
    }
}

// Tensor copy and move operations
TEST_F(TensorTest, TensorCopy)
{
    dlf::Tensor<int> tensor1({2, 3}, 1);
    dlf::Tensor<int> tensor2 = tensor1; // Copy constructor

    // Verify both tensors have the same values
    for (int i = 0; i < 2; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            EXPECT_EQ(tensor1.view(i).view(j), tensor2.view(i).view(j));
        }
    }

    // Create a new tensor with different values
    dlf::Tensor<int> tensor3({2, 3}, 42);
    EXPECT_NE(tensor1.view(0).view(0), tensor3.view(0).view(0));
}

TEST_F(TensorTest, TensorMove)
{
    dlf::Tensor<int> tensor1({2, 3}, 1);
    dlf::Tensor<int> tensor2 = std::move(tensor1); // Move constructor

    // Verify tensor2 has the original values
    for (int i = 0; i < 2; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            EXPECT_EQ(tensor2.view(i).view(j), 1);
        }
    }
}

// Tensor element access and modification
TEST_F(TensorTest, TensorElementAccess)
{
    dlf::Tensor<int> tensor({2, 3}, 0);

    // Test element access
    EXPECT_EQ(tensor.view(0).view(0), 0);

    // Test const access
    const dlf::Tensor<int>& const_tensor = tensor;
    EXPECT_EQ(const_tensor.view(0).view(0), 0);

    // Test at() method
    EXPECT_EQ(tensor.at({0, 0}), 0);
    tensor.at({0, 0}) = 42;
    EXPECT_EQ(tensor.at({0, 0}), 42);

    // Test nested view modification
    tensor.view(0).view(0) = 100;
    EXPECT_EQ(tensor.view(0).view(0), 100);
}

// Tensor comparison
TEST_F(TensorTest, TensorComparison)
{
    dlf::Tensor<int> tensor1({2, 3}, 1);
    dlf::Tensor<int> tensor2({2, 3}, 1);
    dlf::Tensor<int> tensor3({2, 3}, 2);

    // Test equality
    EXPECT_TRUE(tensor1 == tensor2);
    EXPECT_FALSE(tensor1 == tensor3);

    // Test inequality
    EXPECT_FALSE(tensor1 != tensor2);
    EXPECT_TRUE(tensor1 != tensor3);
}

// Tensor serialization
TEST_F(TensorTest, TensorSerialization)
{
    dlf::Tensor<int> tensor({2, 3}, 1);

    // Test serialization to string
    std::string serialized = tensor.serialize();
    EXPECT_FALSE(serialized.empty());

    // Test deserialization
    dlf::Tensor<int> deserialized = dlf::Tensor<int>::deserialize(serialized);
    EXPECT_EQ(tensor, deserialized);
}

// Tensor performance
TEST_F(TensorTest, TensorPerformance)
{
    const int size = 100;
    dlf::Tensor<int> tensor({size, size}, 1);

    // Measure transform performance
    auto start = std::chrono::high_resolution_clock::now();
    tensor.transform([](const int& x) { return x * 2; });
    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Transform operation took " << duration.count() << " ms" << std::endl;

    // Verify the operation completed
    EXPECT_EQ(tensor.view(0).view(0), 2);
}

// Test 3D tensor access
TEST_F(TensorTest, Tensor3DAccess)
{
    dlf::Tensor<int> tensor({2, 3, 4}, 1);

    // Test 3D access
    EXPECT_EQ(tensor.view(0).view(0).view(0), 1);
    tensor.view(0).view(0).view(0) = 42;
    EXPECT_EQ(tensor.view(0).view(0).view(0), 42);

    // Test const 3D access
    const dlf::Tensor<int>& const_tensor = tensor;
    EXPECT_EQ(const_tensor.view(0).view(0).view(0), 42);

    // Test out of bounds
    EXPECT_THROW(tensor.view(2).view(0).view(0), std::out_of_range);
    EXPECT_THROW(tensor.view(0).view(3).view(0), std::out_of_range);
    EXPECT_THROW(tensor.view(0).view(0).view(4), std::out_of_range);
}
