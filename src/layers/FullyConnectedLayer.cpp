#include "layers/FullyConnectedLayer.h"

// Fully Connected Layer Implementation
FullyConnectedLayer::FullyConnectedLayer(int input_size, int output_size)
    : in_size(input_size), out_size(output_size), 
      weights(Eigen::MatrixXd::Random(output_size, input_size) * std::sqrt(2.0 / input_size)),
      biases(Eigen::VectorXd::Zero(output_size)),
      grad_weights(Eigen::MatrixXd::Zero(output_size, input_size)),
      grad_biases(Eigen::VectorXd::Zero(output_size)) {}

Eigen::MatrixXd FullyConnectedLayer::forward(const Eigen::MatrixXd& input) {
    input_cache = input;
    Eigen::MatrixXd output = (weights * input.transpose()).colwise() + biases;
    return output.transpose();
}

Eigen::MatrixXd FullyConnectedLayer::backward(const Eigen::MatrixXd& grad_output) {

    grad_weights += grad_output.transpose() * input_cache;
    grad_biases += grad_output.colwise().sum();

    Eigen::MatrixXd grad_input = grad_output * weights;
    return grad_input;
}

void FullyConnectedLayer::updateParameters(double learning_rate) {
    weights -= learning_rate * (grad_weights / input_cache.rows());
    biases -= learning_rate * (grad_biases / input_cache.rows());

    grad_weights.setZero();
    grad_biases.setZero();
}
