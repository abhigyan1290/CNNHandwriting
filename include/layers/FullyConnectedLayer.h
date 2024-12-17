#ifndef FULLYCONNECTEDLAYER_H
#define FULLYCONNECTEDLAYER_H

#include "Layer.h"
#include <Eigen/Dense>
#include <vector>
#include <random>

class FullyConnectedLayer : public Layer {
public:
    FullyConnectedLayer(int input_size, int output_size);
    
    Eigen::MatrixXd forward(const Eigen::MatrixXd& input) override;
    Eigen::MatrixXd backward(const Eigen::MatrixXd& grad_output) override;
    void updateParameters(double learning_rate) override;

private:
    int in_size;
    int out_size;

    Eigen::MatrixXd weights;         
    Eigen::VectorXd biases;          

    Eigen::MatrixXd grad_weights;   
    Eigen::VectorXd grad_biases;    

    Eigen::MatrixXd input_cache;     
};

#endif // FULLYCONNECTEDLAYER_H
