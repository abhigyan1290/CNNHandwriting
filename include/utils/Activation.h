#ifndef ACTIVATION_H
#define ACTIVATION_H

#include "layers/Layer.h" // Ensure the correct path based on your project structure
#include <Eigen/Dense>

// Activation class inherits from Layer
class Activation : public Layer {
public:
    virtual Eigen::MatrixXd forward(const Eigen::MatrixXd& input) = 0;
    virtual Eigen::MatrixXd backward(const Eigen::MatrixXd& grad_output) = 0;
    
    // Provide a default implementation for updateParameters
    // If Activation layers don't have parameters, this can be empty
    virtual void updateParameters(double learning_rate) override {}
    
    virtual ~Activation() {}
};

// ReLU class inherits from Activation
class ReLU : public Activation {
public:
    Eigen::MatrixXd forward(const Eigen::MatrixXd& input) override;
    Eigen::MatrixXd backward(const Eigen::MatrixXd& grad_output) override;
    
    // Implement updateParameters (empty if no parameters)
    void updateParameters(double learning_rate) override {}
    
private:
    Eigen::MatrixXd input_cache;
};

// Softmax class inherits from Activation
class Softmax : public Activation {
public:
    Eigen::MatrixXd forward(const Eigen::MatrixXd& input) override;
    Eigen::MatrixXd backward(const Eigen::MatrixXd& grad_output) override;
    
    // Implement updateParameters (empty if no parameters)
    void updateParameters(double learning_rate) override {}
    
private:
    Eigen::MatrixXd output_cache;
};

#endif // ACTIVATION_H
