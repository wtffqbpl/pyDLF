#include <gtest/gtest.h>
#include <data/tensor.hpp>

TEST(tensor_test, basic_tensor_creation) {
  dlf::Matrix<int> tensor({2, 3}, 1);

  auto &shape = tensor.shape();
  EXPECT_EQ(shape[0], 2);
  EXPECT_EQ(shape[1], 3);
}

TEST(tensor_test, tensor_size) {
  dlf::Matrix<int> tensor({2, 3}, 1);

  EXPECT_EQ(tensor.size(), 6);
}

TEST(tensor_test, tensor_empty) {
  dlf::Matrix<int> tensor({0, 0}, 1);

  EXPECT_TRUE(tensor.empty());
}

TEST(tensor_test, tensor_strides) {
  dlf::Matrix<int> tensor({2, 3}, 1);

  auto strides = tensor.strides();
  EXPECT_EQ(strides[0], 3);
  EXPECT_EQ(strides[1], 1);
}

TEST(tensor_test, tensor_reshape) {
  dlf::Matrix<int> tensor({2, 3}, 1);
  tensor.reshape({3, 2});

  auto &shape = tensor.shape();
  EXPECT_EQ(shape[0], 3);
  EXPECT_EQ(shape[1], 2);
}

TEST(tensor_test, tensor_transform) {
  dlf::Matrix<int> tensor({2, 3}, 1);
  tensor.transform([](const int &x) { return x + 1; });

  EXPECT_EQ(tensor[0][0], 2);
  EXPECT_EQ(tensor[1][2], 2);
}

TEST(tensor_test, tensor_permute) {
  dlf::Matrix<int> tensor({2, 3}, 1);
  auto permuted_shape = tensor.permute({1, 0});

  EXPECT_EQ(permuted_shape[0], 3);
  EXPECT_EQ(permuted_shape[1], 2);
}

TEST(tensor_test, tensor_out_of_range) {
  dlf::Matrix<int> tensor({2, 3}, 1);

  EXPECT_THROW(tensor[2][0], std::out_of_range);
  EXPECT_THROW(tensor[0][3], std::out_of_range);
}
