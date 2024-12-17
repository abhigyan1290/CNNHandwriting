#ifndef POOLINGLAYER_H
#define POOLINGLAYER_H

#include "Layer.h"
#include <Eigen/Dense>

class PoolingLayer : public Layer {
public:
    PoolingLayer(int pool_size, int stride);
    
    Eigen::MatrixXd forward(const Eigen::MatrixXd& input) override;
    Eigen::MatrixXd backward(const Eigen::MatrixXd& grad_output) override;
    void updateParameters(double learning_rate) override;

private:
    int pool_size;
    int stride;
    Eigen::MatrixXd input_cache;
};

#endif // POOLINGLAYER_H
