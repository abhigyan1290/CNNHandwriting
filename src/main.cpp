#include <iostream>
#include "utils/DataLoader.h"
#include <unistd.h>   // For getcwd
#include <limits.h>   // For PATH_MAX

int main() {
    char cwd[PATH_MAX];
    if (getcwd(cwd, sizeof(cwd)) != NULL) {
        std::cout << "Current working directory: " << cwd << std::endl;
    } else {
        perror("getcwd() error");
        return 1;
    }

    try {
        // Initialize DataLoader for training data
        DataLoader trainData("/home/abhigyan1290/projects/CNN-Handwriting/data/train-images.idx3-ubyte",
                            "/home/abhigyan1290/projects/CNN-Handwriting/data/train-labels.idx1-ubyte");
        
        // Initialize DataLoader for test data
        DataLoader testData("/home/abhigyan1290/projects/CNN-Handwriting/data/t10k-images.idx3-ubyte",
                           "/home/abhigyan1290/projects/CNN-Handwriting/data/t10k-labels.idx1-ubyte");

        // Print the number of training and test samples
        std::cout << "Loaded " << trainData.getNumSamples() << " training samples." << std::endl;
        std::cout << "Loaded " << testData.getNumSamples() << " test samples." << std::endl;

        // Fetch the first batch from training data
        int batch_size = 10;
        Batch batch = trainData.getBatch(0, batch_size);
        std::cout << "First batch size: " << batch.inputs.size() << std::endl;

        // Print the first input and label
        std::cout << "First input (first 10 pixels): ";
        for(int i = 0; i < 10; ++i) {
            std::cout << batch.inputs[0](i) << " ";
        }
        std::cout << std::endl;

        std::cout << "First label: " << batch.labels[0].transpose() << std::endl;
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }
    
    return 0;
}
