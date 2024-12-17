#include <iostream>
#include <vector>
#include <memory>
#include "utils/DataLoader.h"
#include "utils/NeuralNetwork.h"
#include "utils/Activation.h"
#include "utils/Loss.h"
#include "utils/Optimizer.h"
#include "layers/ConvolutionalLayer.h"
#include "layers/PoolingLayer.h"
#include "layers/FullyConnectedLayer.h"
#include "layers/SoftmaxLayer.h"

double evaluate(NeuralNetwork& network, DataLoader& testData, int batch_size) {
    int num_batches = testData.getNumSamples() / batch_size;
    int correct = 0;
    for(int batch_idx = 0; batch_idx < num_batches; ++batch_idx) {
        // Get batch
        Batch batch = testData.getBatch(batch_idx * batch_size, batch_size);
        
        // Convert batch inputs and labels to matrices
        Eigen::MatrixXd input_batch(batch_size, batch.inputs[0].size());
        Eigen::MatrixXd label_batch(batch_size, batch.labels[0].size());
        for(int i = 0; i < batch_size; ++i) {
            input_batch.row(i) = batch.inputs[i];
            label_batch.row(i) = batch.labels[i];
        }

        // Forward pass
        Eigen::MatrixXd predictions = network.forward(input_batch);

        // Get predicted classes
        for(int i = 0; i < batch_size; ++i) {
            int predicted = 0;
            predictions.row(i).maxCoeff(&predicted);
            int actual = 0;
            label_batch.row(i).maxCoeff(&actual);
            if(predicted == actual) {
                correct++;
            }
        }
    }

    return (double)correct / (num_batches * batch_size) * 100.0;
}

int main() {
    try {
        // Load training and test data
        DataLoader trainData("/home/abhigyandoshi/CNNHandwriting/data/train-images.idx3-ubyte",
                            "/home/abhigyandoshi/CNNHandwriting/data/train-labels.idx1-ubyte");
        
        // Initialize DataLoader for test data
        DataLoader testData("/home/abhigyandoshi/CNNHandwriting/data/t10k-images.idx3-ubyte",
                           "/home/abhigyandoshi/CNNHandwriting/data/t10k-labels.idx1-ubyte");
        std::cout << "Loaded " << trainData.getNumSamples() << " training samples." << std::endl;
        std::cout << "Loaded " << testData.getNumSamples() << " test samples." << std::endl;

        // Initialize the network
        NeuralNetwork network;
        
        // Add layers
        network.addLayer(std::make_shared<ConvolutionalLayer>(1, 8, 3, 1, 1)); // Example parameters
        network.addLayer(std::make_shared<ReLU>());
        network.addLayer(std::make_shared<PoolingLayer>(2, 2));
        network.addLayer(std::make_shared<FullyConnectedLayer>(8 * 14 * 14, 10)); // Adjust based on pooling
        network.addLayer(std::make_shared<SoftmaxLayer>());

        // Initialize loss function and optimizer
        CrossEntropyLoss loss_fn;
        SGD optimizer(0.01); // Learning rate

        // Training parameters
        int epochs = 10;
        int batch_size = 64;
        int num_batches = trainData.getNumSamples() / batch_size;

        for(int epoch = 0; epoch < epochs; ++epoch) {
            double epoch_loss = 0.0;
            for(int batch_idx = 0; batch_idx < num_batches; ++batch_idx) {
                // Get batch
                Batch batch = trainData.getBatch(batch_idx * batch_size, batch_size);
                
                // Convert batch inputs and labels to matrices
                Eigen::MatrixXd input_batch(batch_size, batch.inputs[0].size());
                Eigen::MatrixXd label_batch(batch_size, batch.labels[0].size());
                for(int i = 0; i < batch_size; ++i) {
                    input_batch.row(i) = batch.inputs[i];
                    label_batch.row(i) = batch.labels[i];
                }

                // Forward pass
                Eigen::MatrixXd predictions = network.forward(input_batch);

                // Compute loss
                double loss = loss_fn.compute(predictions, label_batch);
                epoch_loss += loss;

                // Backward pass
                Eigen::MatrixXd grad_loss = loss_fn.derivative(predictions, label_batch);
                network.backward(grad_loss);

                // Update parameters
                optimizer.update(network.getLayers());
            }

            // Calculate average loss
            double average_loss = epoch_loss / num_batches;

            // Evaluate on test data
            double accuracy = evaluate(network, testData, batch_size);

            std::cout << "Epoch " << epoch + 1 << "/" << epochs 
                      << " - Loss: " << average_loss 
                      << " - Accuracy: " << accuracy << "%" << std::endl;
        }

        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }
    
    return 0;
}
