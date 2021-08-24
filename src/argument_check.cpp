
#include "Eigen/Dense"
#include <iostream>

using Eigen::MatrixXd;
using Eigen::VectorXd;

VectorXd testFn(VectorXd &x)
{
    x(0) = 5;
    return x;
}

int main()
{    
	VectorXd x_pre;
    x_pre << 1,  1, 1, 0, 0;

    VectorXd x_post = testFn(x_pre);

    std::cout << x_post;
}
