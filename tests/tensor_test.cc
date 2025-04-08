#include <gtest/gtest.h>
#include <data/tensor.hpp>
#include <utils/logger.hpp>

class TensorTest : public ::testing::Test {
protected:
    void SetUp() override {
        dlf::Logger::getInstance().initialize("tensor_test.log");
    }
};

// Basic tensor operations
TEST_F(TensorTest, BasicTensorCreation) {
    dlf::Matrix<int> tensor({2, 3}, 1);
    auto& shape = tensor.shape();
    EXPECT_EQ(shape[0], 2);
    EXPECT_EQ(shape[1], 3);
    
    // Verify all elements are initialized to 1
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 3; ++j) {
            EXPECT_EQ(tensor[i][j], 1);
        }
    }
}

TEST_F(TensorTest, TensorSize) {
    dlf::Matrix<int> tensor({2, 3}, 1);
    EXPECT_EQ(tensor.size(), 6);
    
    // Test with different dimensions
    dlf::Matrix<int> tensor3d({2, 3, 4}, 1);
    EXPECT_EQ(tensor3d.size(), 24);
}

TEST_F(TensorTest, TensorEmpty) {
    dlf::Matrix<int> tensor({0, 0}, 1);
    EXPECT_TRUE(tensor.empty());
    
    // Test with non-empty tensor
    dlf::Matrix<int> non_empty({1, 1}, 1);
    EXPECT_FALSE(non_empty.empty());
}

TEST_F(TensorTest, TensorStrides) {
    dlf::Matrix<int> tensor({2, 3}, 1);
    auto strides = tensor.strides();
    EXPECT_EQ(strides[0], 3);
    EXPECT_EQ(strides[1], 1);
    
    // Test with 3D tensor
    dlf::Matrix<int> tensor3d({2, 3, 4}, 1);
    auto strides3d = tensor3d.strides();
    EXPECT_EQ(strides3d[0], 12);  // 3 * 4
    EXPECT_EQ(strides3d[1], 4);   // 4
    EXPECT_EQ(strides3d[2], 1);   // 1
}

// Tensor operations
TEST_F(TensorTest, TensorReshape) {
    dlf::Matrix<int> tensor({2, 3}, 1);
    tensor.reshape({3, 2});
    auto& shape = tensor.shape();
    EXPECT_EQ(shape[0], 3);
    EXPECT_EQ(shape[1], 2);
    
    // Test invalid reshape
    EXPECT_THROW(tensor.reshape({4, 2}), std::invalid_argument);
}

TEST_F(TensorTest, TensorTransform) {
    dlf::Matrix<int> tensor({2, 3}, 1);
    tensor.transform([](const int& x) { return x + 1; });
    
    // Verify all elements are incremented
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 3; ++j) {
            std::cout << "tensor.at(" << i << ", " << j << ") = " << tensor.at(i, j) << std::endl;
            EXPECT_EQ(tensor.at(i, j), 2);
        }
    }
}

TEST_F(TensorTest, TensorPermute) {
    dlf::Matrix<int> tensor({2, 3}, 1);
    auto permuted_shape = tensor.permute({1, 0});
    EXPECT_EQ(permuted_shape[0], 3);
    EXPECT_EQ(permuted_shape[1], 2);
    
    // Test with 3D tensor
    dlf::Matrix<int> tensor3d({2, 3, 4}, 1);
    auto permuted_shape3d = tensor3d.permute({2, 0, 1});
    EXPECT_EQ(permuted_shape3d[0], 4);
    EXPECT_EQ(permuted_shape3d[1], 2);
    EXPECT_EQ(permuted_shape3d[2], 3);
}

// Error handling
TEST_F(TensorTest, TensorOutOfRange) {
    dlf::Matrix<int> tensor({2, 3}, 1);
    EXPECT_THROW(tensor.at(2, 0), std::out_of_range);
    EXPECT_THROW(tensor.at(0, 3), std::out_of_range);
    EXPECT_THROW(tensor.at(-1, 0), std::out_of_range);
}

// Tensor operations with different data types
TEST_F(TensorTest, TensorFloatOperations) {
    dlf::Matrix<float> tensor({2, 3}, 1.0f);
    tensor.transform([](const float& x) { return x * 2.0f; });
    
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 3; ++j) {
            EXPECT_FLOAT_EQ(tensor.at(i, j), 2.0f);
        }
    }
}

// Tensor copy and move operations
TEST_F(TensorTest, TensorCopy) {
    dlf::Matrix<int> tensor1({2, 3}, 1);
    dlf::Matrix<int> tensor2 = tensor1;  // Copy constructor
    
    // Verify both tensors have the same values
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 3; ++j) {
            EXPECT_EQ(tensor1.at(i, j), tensor2.at(i, j));
        }
    }
    
    // Create a new tensor with different values
    dlf::Matrix<int> tensor3({2, 3}, 42);
    EXPECT_NE(tensor1.at(0, 0), tensor3.at(0, 0));
}

TEST_F(TensorTest, TensorMove) {
    dlf::Matrix<int> tensor1({2, 3}, 1);
    dlf::Matrix<int> tensor2 = std::move(tensor1);  // Move constructor
    
    // Verify tensor2 has the original values
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 3; ++j) {
            EXPECT_EQ(tensor2.at(i, j), 1);
        }
    }
    
    // Verify tensor1 is in a valid but unspecified state
    EXPECT_TRUE(tensor1.empty());
}

// Tensor element access and modification
TEST_F(TensorTest, TensorElementAccess) {
    dlf::Matrix<int> tensor({2, 3}, 0);
    
    // Test element access
    EXPECT_EQ(tensor.at(0, 0), 0);
    
    // Test const access
    const dlf::Matrix<int>& const_tensor = tensor;
    EXPECT_EQ(const_tensor.at(0, 0), 0);
}

// Tensor comparison
TEST_F(TensorTest, TensorComparison) {
    dlf::Matrix<int> tensor1({2, 3}, 1);
    dlf::Matrix<int> tensor2({2, 3}, 1);
    dlf::Matrix<int> tensor3({2, 3}, 2);
    
    // Test equality
    EXPECT_TRUE(tensor1 == tensor2);
    EXPECT_FALSE(tensor1 == tensor3);
    
    // Test inequality
    EXPECT_FALSE(tensor1 != tensor2);
    EXPECT_TRUE(tensor1 != tensor3);
}

// Tensor serialization
TEST_F(TensorTest, TensorSerialization) {
    dlf::Matrix<int> tensor({2, 3}, 1);
    
    // Test serialization to string
    std::string serialized = tensor.serialize();
    EXPECT_FALSE(serialized.empty());

    std::cout << "Serialized tensor: " << serialized << std::endl;

    // Test deserialization
    dlf::Matrix<int> deserialized = dlf::Matrix<int>::deserialize(serialized);
    EXPECT_EQ(tensor, deserialized);
}

// Tensor performance
TEST_F(TensorTest, TensorPerformance) {
    const int size = 1000;
    dlf::Matrix<int> tensor({size, size}, 1);
    
    // Measure transform performance
    auto start = std::chrono::high_resolution_clock::now();
    tensor.transform([](const int& x) { return x * 2; });
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    DLF_LOG_INFO("Transform operation took {} ms", duration.count());
    
    // Verify the operation completed
    EXPECT_EQ(tensor.at(0, 0), 2);
}
