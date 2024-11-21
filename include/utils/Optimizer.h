#ifndef OPTIMIZER_H
#define OPTIMIZER_H


#include <Eigen/Dense>
#include <vector>
#include <memory>

class Layer;

class Optimizer{
public:
    virtual void update(std::vector<std::shared_ptr<Layer>>& layers) = 0;
    virtual ~Optimizer() {}
};

class SGD : public Optimizer{
public: 
    SGD(double learning_rate);
    void update(std::vector<std::shared_ptr<Layer>>& layers) override;

private:
    double lr;
};

#endif