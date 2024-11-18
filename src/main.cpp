#include <iostream>
#include "utils/DataLoader.h"
#include "utils/Activation.h"   // Include Activation header
#include <unistd.h>             // For getcwd
#include <limits.h>             // For PATH_MAX
#include <Eigen/Dense>          // Include Eigen for matrix operations

int main() {
    // Print current working directory
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

        // -----------------------------
        // Activation Functions Testing
        // -----------------------------

        // Initialize Activation Functions
        ReLU relu;
        Softmax softmax;

        // Create a sample input matrix for ReLU
        // Example: A 3x4 matrix with both positive and negative values
        Eigen::MatrixXd relu_input(3, 4);
        relu_input << -1, 2, -3, 4,
                      5, -6, 7, -8,
                      9, 10, -11, 12;
        std::cout << "\nReLU Activation Test:" << std::endl;
        std::cout << "Input:\n" << relu_input << std::endl;

        // Apply ReLU
        Eigen::MatrixXd relu_output = relu.forward(relu_input);
        std::cout << "Output after ReLU:\n" << relu_output << std::endl;

        // Create a sample input matrix for Softmax
        // Example: A 2x3 matrix (2 samples, 3 classes)
        Eigen::MatrixXd softmax_input(2, 3);
        softmax_input << 1.0, 2.0, 3.0,
                         1.0, 2.0, 3.0;
        std::cout << "\nSoftmax Activation Test:" << std::endl;
        std::cout << "Input:\n" << softmax_input << std::endl;

        // Apply Softmax
        Eigen::MatrixXd softmax_output = softmax.forward(softmax_input);
        std::cout << "Output after Softmax:\n" << softmax_output << std::endl;

    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }
    
    return 0;
}
