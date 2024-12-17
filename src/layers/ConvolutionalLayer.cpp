#include "layers/ConvolutionalLayer.h"
#include <cmath>
#include <iostream>

// Initialize filters with small random values
ConvolutionalLayer::ConvolutionalLayer(int input_channels, int output_channels, int kernel_size, int stride, int padding)
    : input_channels(input_channels), output_channels(output_channels), kernel_size(kernel_size), stride(stride), padding(padding)
{
    std::default_random_engine generator;
    std::normal_distribution<double> distribution(0.0, 1.0 / std::sqrt(kernel_size * kernel_size));

    for (int i = 0; i < output_channels; ++i) {
        Eigen::MatrixXd filter = Eigen::MatrixXd::Zero(input_channels, kernel_size * kernel_size);
        for (int c = 0; c < input_channels; ++c) {
            for (int j = 0; j < kernel_size * kernel_size; ++j) {
                filter(c, j) = distribution(generator);
            }
        }
        filters.push_back(filter);
        biases.push_back(0.0);
        grad_filters.emplace_back(Eigen::MatrixXd::Zero(input_channels, kernel_size * kernel_size));
        grad_biases.emplace_back(0.0);
    }
}


Eigen::MatrixXd ConvolutionalLayer::forward(const Eigen::MatrixXd& input) {
    input_cache = input;
    int input_size = std::sqrt(input.cols()); // Assuming square input
    int output_size = (input_size - kernel_size + 2 * padding) / stride + 1;

    Eigen::MatrixXd output = Eigen::MatrixXd::Zero(output_channels, output_size * output_size);

    for (int oc = 0; oc < output_channels; ++oc) {
        Eigen::MatrixXd conv_result = Eigen::MatrixXd::Zero(output_size, output_size);
        for (int ic = 0; ic < input_channels; ++ic) {
            // ...
        }
        conv_result.array() += biases[oc];
        output.row(oc) = conv_result.reshaped(1, output_size * output_size);
    }
    return output;
}

Eigen::MatrixXd ConvolutionalLayer::backward(const Eigen::MatrixXd& grad_output) {
    return Eigen::MatrixXd(); 
}

void ConvolutionalLayer::updateParameters(double learning_rate) {
    for (int oc = 0; oc < output_channels; ++oc) {
        filters[oc] -= learning_rate * grad_filters[oc];
        biases[oc] -= learning_rate * grad_biases[oc];

        grad_filters[oc].setZero();
        grad_biases[oc] = 0.0;
    }
}

