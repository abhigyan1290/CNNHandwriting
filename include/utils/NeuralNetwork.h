#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include <vector>
#include <memory>
#include "layers/Layer.h"         // Ensure correct path
#include "utils/Activation.h"     // Ensure correct path
#include "utils/Loss.h"           // Ensure correct path
#include "utils/Optimizer.h"      // Ensure correct path

class NeuralNetwork {
public:
    NeuralNetwork();

    void addLayer(std::shared_ptr<Layer> layer);

    Eigen::MatrixXd forward(const Eigen::MatrixXd& input);

    void backward(const Eigen::MatrixXd& grad_output);

    void updateParameters(double learning_rate);

    std::vector<std::shared_ptr<Layer>>& getLayers();

private:
    std::vector<std::shared_ptr<Layer>> layers;
};

#endif // NEURALNETWORK_H
