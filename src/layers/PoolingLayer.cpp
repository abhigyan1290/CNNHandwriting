#include "layers/PoolingLayer.h"

// Max Pooling Implementation
PoolingLayer::PoolingLayer(int pool_size_, int stride_) : pool_size(pool_size_), stride(stride_) {}

Eigen::MatrixXd PoolingLayer::forward(const Eigen::MatrixXd& input) {
    input_cache = input;
    int input_size = std::sqrt(input.cols()); 
    int output_size = (input_size - pool_size) / stride + 1;
    int channels = input.rows();

    Eigen::MatrixXd output = Eigen::MatrixXd::Zero(channels, output_size * output_size);

    for(int c = 0; c < channels; ++c) {
        Eigen::MatrixXd input_matrix = input.row(c).reshaped(input_size, input_size);
        Eigen::MatrixXd pooled = Eigen::MatrixXd::Zero(output_size, output_size);

        for(int i = 0; i < output_size; ++i) {
            for(int j = 0; j < output_size; ++j) {
                Eigen::MatrixXd region = input_matrix.block(i * stride, j * stride, pool_size, pool_size);
                pooled(i, j) = region.maxCoeff();
            }
        }
        output.row(c) = pooled.reshaped(1, output_size * output_size);
    }

    return output;
}

Eigen::MatrixXd PoolingLayer::backward(const Eigen::MatrixXd& grad_output) {
    return Eigen::MatrixXd(); 
}

void PoolingLayer::updateParameters(double learning_rate) {
}
