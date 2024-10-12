
#include "include/nn.h"
#include "include/matrix.h"
#include "include/image.h"
#include <time.h>

// The training process is too slow I'm not even sure if this is working properly


int main(int argc, char *argv[]) {
    if (argc < 2) {
        printf("Usage: %s <dataset_directory>\n", argv[0]);
        return 1;
    }

    srand(time(NULL));

    // Process the dataset
    Dataset *dataset = process_directory_with_labels(argv[1]);
    if (dataset == NULL || dataset->count == 0) {
        fprintf(stderr, "No images found or failed to load images.\n");
        return 1;
    }
    printf("Loaded %d images across %d classes.\n", dataset->count, dataset->num_classes);
    
    
    // Define network architecture
    size_t input_size = dataset->image_sizes[0]; // Number of input neurons
    size_t hidden_size = 128; // Hidden layer size
    size_t num_classes = dataset->num_classes;

    size_t architecture[] = {input_size, hidden_size, num_classes};
    size_t architecture_count = sizeof(architecture) / sizeof(architecture[0]);

    // Load or initialize the neural
    NN neural_network = nn_load("nn_configuration.txt");
    printf("Neural network loaded!\n");

    //NN neural_network = nn_alloc(architecture, architecture_count);
    //nn_rand(neural_network, -0.5f, 0.5f); // Initialize weights and biases randomly
    //printf("Neural network initialized!\n");

    // Training parameters
    int epochs = 100;
    float learning_rate = 0.1f;

    // Training loop
    for (int epoch = 0; epoch < epochs; epoch++) {
        float total_cost = 0.0f;
        
        for (int i = 0; i < dataset->count; i++) {
            // Convert image to matrix
            Mat input = mat_alloc(1, dataset->image_sizes[i]);
            for (int j = 0; j < dataset->image_sizes[i]; j++) {
                MAT_AT(input, 0, j) = dataset->images[i][j];
            }

            // One-hot encode the label
            Mat target = one_hot_encode(dataset->labels[i], num_classes);

            // Copy input to the network's input layer
            mat_copy(neural_network.as[0], input);

            // Forward pass
            nn_forward(neural_network);

            // Compute cost
            double cost = 0.0f;
            for (size_t j = 0; j < num_classes; j++) {
                double diff = MAT_AT(neural_network.as[neural_network.size], 0, j) - MAT_AT(target, 0, j);
                cost += diff * diff;
            }

            total_cost += cost;

            // Backpropagation
            nn_backprop(neural_network, input, target, learning_rate);

            // Free temporary matrices
            mat_free(input);
            mat_free(target);
        }

        // Compute average cost for the epoch
        float average_cost = total_cost / dataset->count;
    
        // Print progress every 100 epochs
        //if ((epoch + 1) % 100 == 0 || epoch == 0) {
            printf("Epoch %d/%d, Cost: %.4f\n", epoch + 1, epochs, average_cost);
        //}
    }

    // Save trained model
    nn_save(neural_network, "nn_configuration.txt");


    // Evaluation on the training set (for demonstration)
    int correct = 0;
    for (int i = 0; i < dataset->count; i++) {
        // Convert image to matrix
        Mat input = mat_alloc(1, dataset->image_sizes[i]);
        for (int j = 0; j < dataset->image_sizes[i]; j++) {
            MAT_AT(input, 0, j) = dataset->images[i][j];
        }

        int predicted = nn_predict(neural_network, input);

        // Check if prediction is correct
        if (predicted == dataset->labels[i]) {
            correct++;
        }

        // Free temporary matrices
        mat_free(input);
    }



    float accuracy = (float)correct / dataset->count * 100.0f;
    printf("Training Accuracy: %.2f%% (%d/%d)\n", accuracy, correct, dataset->count);

    // Free the neural network
    nn_free(neural_network);

    // Free the dataset
    for (int i = 0; i < dataset->count; i++) {
        free(dataset->images[i]);
    }
    free(dataset->images);
    free(dataset->image_sizes);
    free(dataset->labels);
    free(dataset);
    return 0;
}


