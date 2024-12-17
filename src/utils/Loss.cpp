#include "utils/Loss.h"
#include <cmath>

double CrossEntropyLoss::compute(const Eigen::MatrixXd& predictions, const Eigen::MatrixXd& targets) {
    double loss = 0.0;
    for(int i = 0; i < predictions.rows(); ++i){
        for(int j = 0; j < predictions.cols(); ++j){
            loss -= targets(i,j) * std::log(predictions(i,j) + 1e-15);
        }
    }
    return loss / predictions.rows();
}

Eigen::MatrixXd CrossEntropyLoss::derivative(const Eigen::MatrixXd& predictions, const Eigen::MatrixXd& targets) {
    Eigen::MatrixXd grad = predictions - targets;
    grad /= predictions.rows();
    return grad;
}
