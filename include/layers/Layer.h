#ifndef LAYER_H
#define LAYER_H

#include <Eigen/Dense>

class Layer {
public:
    virtual Eigen::MatrixXd forward(const Eigen::MatrixXd& input) = 0;
    virtual Eigen::MatrixXd backward(const Eigen::MatrixXd& grad_output) = 0;
    virtual void updateParameters(double learning_rate) = 0;
    virtual ~Layer() {}
};

#endif
