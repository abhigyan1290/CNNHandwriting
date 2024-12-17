#include "utils/Optimizer.h"
#include "layers/Layer.h" 

SGD::SGD(double learning_rate) : lr(learning_rate) {}

void SGD::update(std::vector<std::shared_ptr<Layer>>& layers) {
    for(auto& layer : layers) {
        layer->updateParameters(lr);
    }
}

