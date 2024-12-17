#include "utils/NeuralNetwork.h"

NeuralNetwork::NeuralNetwork() {}

void NeuralNetwork::addLayer(std::shared_ptr<Layer> layer) {
    layers.push_back(layer);
}

Eigen::MatrixXd NeuralNetwork::forward(const Eigen::MatrixXd& input) {
    Eigen::MatrixXd output = input;
    for(auto& layer : layers) {
        output = layer->forward(output);
    }
    return output;
}

void NeuralNetwork::backward(const Eigen::MatrixXd& grad_output) {
    Eigen::MatrixXd grad = grad_output;
    // Iterate in reverse order
    for(auto it = layers.rbegin(); it != layers.rend(); ++it) {
        grad = (*it)->backward(grad);
    }
}

void NeuralNetwork::updateParameters(double learning_rate) {
    for(auto& layer : layers) {
        layer->updateParameters(learning_rate);
    }
}

std::vector<std::shared_ptr<Layer>>& NeuralNetwork::getLayers() {
    return layers;
}
