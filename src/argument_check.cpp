
#include "Eigen/Dense"
#include <iostream>

using Eigen::MatrixXd;
using Eigen::VectorXd;

VectorXd testFn(VectorXd &x)
{
    x(0) = 5;
    return x;
}

VectorXd testFn2(VectorXd &y)
{
    VectorXd z(4);
    z << 0, 5, 10, 15;
    z(2) = z(1) + y(1);
    std::cout << "z in testFn2 " << z << "    y(1) = " << y(1) << std::endl;
    return z;
}

int main()
{    
	VectorXd x_pre(5);
    x_pre << 1,  1, 1, 0, 0;
    std::cout << x_pre << std::endl;

    VectorXd x_post(5);
    x_post = testFn(x_pre);

    std::cout << x_post << std::endl;

    VectorXd z(4);
    z = testFn2(x_post);

    std::cout << "z in main: " << z << std::endl;
}
