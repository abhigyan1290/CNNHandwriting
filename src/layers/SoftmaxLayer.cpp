#include "layers/SoftmaxLayer.h"
#include <cmath>

// Softmax Layer Implementation
SoftmaxLayer::SoftmaxLayer() {}

Eigen::MatrixXd SoftmaxLayer::forward(const Eigen::MatrixXd& input) {
    Eigen::MatrixXd shifted = input;
    for(int i = 0; i < input.rows(); ++i) {
        double max_val = input.row(i).maxCoeff();
        shifted.row(i) = input.row(i).array() - max_val;
    }
    Eigen::MatrixXd exps = shifted.array().exp();
    Eigen::VectorXd sums = exps.rowwise().sum();
    output_cache = exps.array().colwise() / sums.array();
    return output_cache;
}

Eigen::MatrixXd SoftmaxLayer::backward(const Eigen::MatrixXd& grad_output) {
    return grad_output;
}

void SoftmaxLayer::updateParameters(double learning_rate) {
}
