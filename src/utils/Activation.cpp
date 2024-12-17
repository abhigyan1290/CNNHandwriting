#include "utils/Activation.h"

#include "utils/Activation.h"

#include "utils/Activation.h"

// ReLU Implementation
Eigen::MatrixXd ReLU::forward(const Eigen::MatrixXd& input) {
    input_cache = input;
    return input.array().max(0.0);
}

Eigen::MatrixXd ReLU::backward(const Eigen::MatrixXd& grad_output) {
    Eigen::MatrixXd grad_input = grad_output;
    grad_input = (input_cache.array() > 0.0).select(grad_input, 0.0);
    return grad_input;
}

// No need to implement updateParameters in .cpp since it's empty in the header


Eigen::MatrixXd Softmax::forward(const Eigen::MatrixXd& input) {
    // For numerical stability, subtract the max from each row
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

Eigen::MatrixXd Softmax::backward(const Eigen::MatrixXd& grad_output) {
    // Typically combined with cross-entropy loss for gradient simplification
    return grad_output;
}

