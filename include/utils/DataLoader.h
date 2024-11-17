#ifndef DATALOADER_H
#define DATALOADER_H

#include <vector>
#include <string>
#include <Eigen/Dense>

struct Batch{
    std::vector<Eigen::VectorXd> inputs;
    std::vector<Eigen::VectorXd> labels;
};

class DataLoader{
public:
    DataLoader(const std::string& images_path, const std::string& labels_path);

    int getNumSamples() const;

    Batch getBatch(int start_index, int batch_size) const;

private:
    std::vector<Eigen::VectorXd> images;
    std::vector<Eigen::VectorXd> labels;

    int reverseInt(int i) const;
    void loadImages(const std::string& path);
    void loadLabels(const std::string& path);
    Eigen::VectorXd oneHotEncode(int label) const;
};

#endif