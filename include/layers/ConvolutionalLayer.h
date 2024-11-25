#ifndef CONVOLUTIONALLAYER_H
#define CONVOLUTIONALLAYER_H

#include "Layer.h"
#include <vector>
#include <Eigen/Dense>
#include <random>

class ConvolutionalLayer : public Layer{
public:
    ConvolutionalLayer(int input_channels, int output_channels, int kernel_size, int stride = 1, int padding = 0);

    Eigen::MatrixXd forward(const Eigen::MatrixXd& input) override;
    Eigen::MatrixXd backward(const Eigen::MatrixXd& grad_output) override;
    void updateParameters(double learning_rate) override;
private:
    int input_channels;
    int output_channels;
    int kernel_size;
    int stride;
    int padding;

    std::vector<Eigen::MatrixXd> filters;
    std::vector<double> biases;

    std::vector<Eigen::MatrixXd> grad_filters;
    std::vector<double> grad_biases;

    Eigen::MatrixXd input_cache;
    Eigen::MatrixXd applyFilter(const Eigen::MatrixXd& input, const Eigen::MatrixXd& filter, double bias) const;

};

#endif