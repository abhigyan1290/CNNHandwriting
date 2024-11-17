#include "utils/DataLoader.h"
#include <fstream>
#include <stdexcept>
#include <iostream>

DataLoader::DataLoader(const std::string& images_path, const std::string& labels_path){
    loadImages(images_path);
    loadLabels(labels_path);

    if(images.size() != labels.size()){
        throw std::runtime_error("Number of images and labels don't match!");
    }
}

int DataLoader::getNumSamples() const{
    return images.size();
}

Batch DataLoader::getBatch(int start_index, int batch_size) const{
    Batch batch;
    for (int i = start_index; i < start_index + batch_size && i < images.size(); ++i){
        batch.inputs.push_back(images[i]);
        batch.labels.push_back(labels[i]);
    }
    return batch;
}

int DataLoader::reverseInt(int i) const{
    unsigned char c1, c2, c3, c4;
    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;
    return((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

void DataLoader::loadImages(const std::string& path){
    std::ifstream file(path, std::ios::binary);
    if(file.is_open()){
        int magic_number = 0;
        int number_of_images = 0;
        int rows = 0;
        int cols = 0;
        file.read((char*)&magic_number, sizeof(magic_number));
        magic_number = reverseInt(magic_number);
        if (magic_number != 2051){
            throw std::runtime_error("Invalid MNIST images");
        }
        file.read((char*)&number_of_images, sizeof(number_of_images));
        number_of_images = reverseInt(number_of_images);
        file.read((char*)&rows, sizeof(rows));
        rows = reverseInt(rows);
        file.read((char*)&cols, sizeof(cols));
        cols = reverseInt(cols);

        images.reserve(number_of_images);
        for(int i = 0; i < number_of_images; ++i){
            Eigen::VectorXd img(rows * cols);
            for (int j = 0; j < rows * cols; ++j){
                unsigned char pixel = 0;
                file.read((char*)&pixel, sizeof(pixel));
                img(j) = static_cast<double>(pixel) / 255.0;
            }
            images.push_back(img);
        }

    } else{
        throw std::runtime_error("Error opening file" + path);
    }
}

void DataLoader::loadLabels(const std::string& path){
    std::ifstream file(path, std::ios::binary);
    if (file.is_open()){
        int magic_number = 0;
        int number_of_labels = 0;
        file.read((char*)&magic_number, sizeof(magic_number));
        magic_number = reverseInt(magic_number);
        if (magic_number != 2049){
            throw std::runtime_error("Invalid MNIST labels");
        }
        file.read((char*)&number_of_labels, sizeof(number_of_labels));
        number_of_labels = reverseInt(number_of_labels);

        labels.reserve(number_of_labels);
        for(int i = 0;i < number_of_labels; ++i){
            unsigned char label = 0;
            file.read((char*)&label, sizeof(label));
            labels.push_back(oneHotEncode(static_cast<int>(label)));
        }
    } else {
        throw std::runtime_error("Error opening file" + path);
    }
}

Eigen::VectorXd DataLoader::oneHotEncode(int label) const{
    Eigen::VectorXd encoded = Eigen::VectorXd::Zero(10);
    if(label < 0 || label > 9){
        throw std::runtime_error("Label out of range");
    }
    encoded(label) = 1.0;
    return encoded;
}