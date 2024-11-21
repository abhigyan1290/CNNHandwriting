#ifndef LOSS_H
#define LOSS_H

#include <Eigen/Dense>

class Loss {
public: 
    virtual double compute(const Eigen::MatrixXd& predictions, const Eigen::MatrixXd& targets) = 0;
    virtual Eigen::MatrixXd derivative(const Eigen::MatrixXd& predictions, const Eigen::MatrixXd& targets) = 0;
    virtual ~Loss() {}
};

class CrossEntropyLoss : public Loss{
public: 
    double compute(const Eigen::MatrixXd& predictions, const Eigen::MatrixXd& targets) override;
    Eigen::MatrixXd derivative(const Eigen::MatrixXd& predictions, const Eigen::MatrixXd& targets) override;
};

#endif