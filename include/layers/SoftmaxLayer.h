#ifndef SOFTMAXLAYER_H
#define SOFTMAXLAYER_H

#include "Layer.h"
#include "utils/Activation.h"

class SoftmaxLayer : public Layer {
public:
    SoftmaxLayer();
    
    Eigen::MatrixXd forward(const Eigen::MatrixXd& input) override;
    Eigen::MatrixXd backward(const Eigen::MatrixXd& grad_output) override;
    void updateParameters(double learning_rate) override;

private:
    Eigen::MatrixXd output_cache;
};

#endif // SOFTMAXLAYER_H
