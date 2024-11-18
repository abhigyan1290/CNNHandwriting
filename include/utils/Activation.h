#ifndef ACTIVATION_H
#define ACTIVATION_H

#include <Eigen/Dense>

class Activation{
public:
    virtual Eigen::MatrixXd forward(const Eigen::MatrixXd& input) = 0;
    virtual Eigen::MatrixXd backward(const Eigen::MatrixXd& grad_output) = 0;
    virtual ~Activation() {}

};

class ReLU : public Activation{
public:
    virtual Eigen::MatrixXd forward(const Eigen::MatrixXd& input) override;
    virtual Eigen::MatrixXd backward(const Eigen::MatrixXd& grad_output) override;

private:
    Eigen::MatrixXd input_cache;

};

class Softmax : public Activation{
public:
    virtual Eigen::MatrixXd forward(const Eigen::MatrixXd& input) override;
    virtual Eigen::MatrixXd backward(const Eigen::MatrixXd& grad_output) override;

private:
    Eigen::MatrixXd output_cache;
};

#endif