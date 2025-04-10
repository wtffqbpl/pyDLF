#include <gtest/gtest.h>

#include <armadillo>
#include <iostream>

TEST(ArmadilloTest, Basic)
{
    using namespace arma;

    arma::fmat a1 = "1, 2, 3;"
                    "4, 5, 6;"
                    "7, 8, 9;";

    arma::fmat a2 = a1;

    std::cout << a2 * a1 << std::endl;
}