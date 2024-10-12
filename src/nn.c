#include "nn.h"

// Allocate memory for your neural network
NN nn_alloc(size_t *arch, size_t arch_count){

    assert(arch_count > 0);

    NN nn;
    nn.size = arch_count - 1; // Number of layers excluding input layer

    nn.ws = calloc(nn.size, sizeof(*nn.ws));
    assert(nn.ws != NULL);

    nn.bs = calloc(nn.size, sizeof(*nn.bs));
    assert(nn.bs != NULL);

    nn.as = calloc(nn.size + 1, sizeof(*nn.as));
    assert(nn.as != NULL);

    // Input
    nn.as[0] = mat_alloc(1, arch[0]);

    // Allocate all layers progressively 1 to nn.size
    for(size_t i = 1; i < nn.size + 1; i++){
        nn.ws[i-1] = mat_alloc(nn.as[i-1].cols, arch[i]);
        nn.bs[i-1] = mat_alloc(1, arch[i]);
        nn.as[i]   = mat_alloc(1, arch[i]);
    }

    return nn;
}

// Randomize the weights and biases
void nn_rand(NN nn, float low, float high){
    
    for(size_t i = 0; i < nn.size; i++){
        mat_rand(nn.ws[i], low, high);
        mat_rand(nn.bs[i], low, high);
    }
}

// Input to Output
void nn_forward(NN nn){
    for(size_t i = 0; i < nn.size; i++){
        mat_dot(nn.as[i+1], nn.as[i], nn.ws[i]);
        mat_sum(nn.as[i+1], nn.bs[i]);
        mat_sig(nn.as[i+1]);
    }
}

// compute the cost, mse
float nn_cost(NN nn, Mat training_input, Mat training_output){
    assert(training_input.rows == training_output.rows);
    assert(training_output.cols == nn.as[nn.size].cols);

    float cost = 0;
    for(size_t i = 0; i < training_input.rows; i++){
        Mat x = mat_row(training_input, i);
        Mat y = mat_row(training_output, i);

        mat_copy(nn.as[0], x);
        nn_forward(nn);

        for(size_t j = 0; j < training_output.cols; j++){
            float d = MAT_AT(nn.as[nn.size], 0, j) - MAT_AT(y, 0, j);
            cost += d*d;
        }
    }
    return (cost/training_input.rows);
}

// Magik
void nn_backprop(NN nn, Mat training_input, Mat training_output, float learning_rate) {
    assert(training_input.cols == nn.as[0].cols);
    assert(training_output.cols == nn.as[nn.size].cols);

    // Forward pass already done before calling backprop
    // Initialize delta for the output layer
    Mat delta = mat_alloc(1, nn.as[nn.size].cols);

    // Compute delta for the output layer: (a_L - y) * sigmoid'(z_L)
    for (size_t j = 0; j < nn.as[nn.size].cols; ++j) {
        float a = MAT_AT(nn.as[nn.size], 0, j);
        float y = MAT_AT(training_output, 0, j);
        float dsig = a * (1.0f - a); // Since sigmoid'(z) = a * (1 - a)
        MAT_AT(delta, 0, j) = (a - y) * dsig;
    }

    // Iterate backward through the layers
    for (size_t l = nn.size; l > 0; --l) {
        // Compute gradients for weights and biases
        // Gradient w.r. to weights: a_(l-1)^T * delta_l
        Mat a_prev = nn.as[l - 1];
        Mat delta_l = delta;

        // Allocate gradient matrices
        Mat grad_w = mat_alloc(a_prev.cols, nn.ws[l - 1].cols);
        Mat grad_b = mat_alloc(1, nn.bs[l - 1].cols);

        // Compute grad_w
        // Transpose a_prev and multiply by delta_l
        // Since a_prev is 1 x n, delta_l is 1 x m, grad_w will be n x m
        for (size_t i = 0; i < a_prev.cols; i++) {
            for (size_t j = 0; j < delta_l.cols; j++) {
                MAT_AT(grad_w, i, j) = MAT_AT(a_prev, 0, i) * MAT_AT(delta_l, 0, j);
            }
        }

        // Compute grad_b (same as delta_l)
        mat_copy(grad_b, delta_l);

        // Update weights and biases: W = W - learning_rate * grad_w
        //                           b = b - learning_rate * grad_b
        for (size_t i = 0; i < nn.ws[l - 1].rows; i++) {
            for (size_t j = 0; j < nn.ws[l - 1].cols; j++) {
                MAT_AT(nn.ws[l - 1], i, j) -= learning_rate * MAT_AT(grad_w, i, j);
            }
        }

        for (size_t j = 0; j < nn.bs[l - 1].cols; j++) {
            MAT_AT(nn.bs[l - 1], 0, j) -= learning_rate * MAT_AT(grad_b, 0, j);
        }

        // Compute delta for the previous layer if not at the input layer
        if (l > 1) {
            // delta_prev = (delta_l * W_l^T) .* sigmoid'(z_(l-1))
            Mat w_transpose = mat_alloc(nn.ws[l - 1].cols, nn.ws[l - 1].rows);
            // Transpose weights
            for (size_t i = 0; i < nn.ws[l - 1].rows; i++) {
                for (size_t j = 0; j < nn.ws[l - 1].cols; j++) {
                    MAT_AT(w_transpose, j, i) = MAT_AT(nn.ws[l - 1], i, j);
                }
            }

            // delta_prev = delta_l * W_l^T
            Mat delta_prev = mat_alloc(1, nn.ws[l - 1].rows);
            for (size_t i = 0; i < delta_prev.cols; i++) {
                MAT_AT(delta_prev, 0, i) = 0.0f;
                for (size_t j = 0; j < delta_l.cols; j++) {
                    MAT_AT(delta_prev, 0, i) += MAT_AT(delta_l, 0, j) * MAT_AT(w_transpose, j, i);
                }
            }

            // Apply sigmoid derivative
            mat_dsig(nn.as[l - 1]);
            for (size_t i = 0; i < delta_prev.cols; i++) {
                MAT_AT(delta_prev, 0, i) *= MAT_AT(nn.as[l - 1], 0, i);
            }

            // Free previous delta and set new delta
            mat_free(delta);
            mat_free(w_transpose);
            delta = delta_prev;
        }

        // Free gradients
        mat_free(grad_w);
        mat_free(grad_b);
    }

    // Free the final delta
    mat_free(delta);
}

// Free the neural network memory
void nn_free(NN nn) {
    for (size_t i = 0; i < nn.size; ++i) {
        mat_free(nn.ws[i]);
        mat_free(nn.bs[i]);
        mat_free(nn.as[i + 1]);
    }
    free(nn.ws);
    free(nn.bs);
    free(nn.as);
}

// Save the configuration of the neural network
void nn_save(NN nn, const char *filename) {
    FILE *file = fopen(filename, "wb");
    if (!file) {
        fprintf(stderr, "Failed to open file for saving model.\n");
        return;
    }

    // Save network size
    fwrite(&nn.size, sizeof(size_t), 1, file);

    // Save weights and biases
    for (size_t i = 0; i < nn.size; i++) {
        fwrite(&nn.ws[i].rows, sizeof(size_t), 1, file);
        fwrite(&nn.ws[i].cols, sizeof(size_t), 1, file);
        fwrite(nn.ws[i].es, sizeof(float), nn.ws[i].rows * nn.ws[i].cols, file);

        fwrite(&nn.bs[i].rows, sizeof(size_t), 1, file);
        fwrite(&nn.bs[i].cols, sizeof(size_t), 1, file);
        fwrite(nn.bs[i].es, sizeof(float), nn.bs[i].rows * nn.bs[i].cols, file);
    }

    fclose(file);
}

// Load the configuration currently saved
NN nn_load(const char *filename) {
    FILE *file = fopen(filename, "rb");
    if (!file) {
        fprintf(stderr, "Failed to open file for loading model.\n");
        exit(1);
    }

    NN nn;

    // Load network size
    if (fread(&nn.size, sizeof(size_t), 1, file) != 1) {
        fprintf(stderr, "Failed to read network size.\n");
        fclose(file);
        exit(1);
    }

    // Allocate memory for weights, biases, and activations
    nn.ws = calloc(nn.size, sizeof(*nn.ws));
    if (!nn.ws) {
        fprintf(stderr, "Failed to allocate memory for weights.\n");
        fclose(file);
        exit(1);
    }

    nn.bs = calloc(nn.size, sizeof(*nn.bs));
    if (!nn.bs) {
        fprintf(stderr, "Failed to allocate memory for biases.\n");
        free(nn.ws);
        fclose(file);
        exit(1);
    }

    nn.as = calloc(nn.size + 1, sizeof(*nn.as));
    if (!nn.as) {
        fprintf(stderr, "Failed to allocate memory for activations.\n");
        free(nn.ws);
        free(nn.bs);
        fclose(file);
        exit(1);
    }

    // Load weights and biases
    for (size_t i = 0; i < nn.size; i++) {
        size_t rows, cols;

        // Load weights dimensions
        if (fread(&rows, sizeof(size_t), 1, file) != 1 ||
            fread(&cols, sizeof(size_t), 1, file) != 1) {
            fprintf(stderr, "Failed to read weight dimensions for layer %zu.\n", i);
            // Free previously allocated memory before exiting
            for (size_t j = 0; j < i; j++) {
                mat_free(nn.ws[j]);
                mat_free(nn.bs[j]);
            }
            free(nn.ws);
            free(nn.bs);
            free(nn.as);
            fclose(file);
            exit(1);
        }

        // Allocate and load weights
        nn.ws[i] = mat_alloc(rows, cols);
        if (fread(nn.ws[i].es, sizeof(float), rows * cols, file) != rows * cols) {
            fprintf(stderr, "Failed to read weight data for layer %zu.\n", i);
            // Free previously allocated memory before exiting
            for (size_t j = 0; j <= i; j++) {
                mat_free(nn.ws[j]);
                if (j < i) {
                    mat_free(nn.bs[j]);
                }
            }
            free(nn.ws);
            free(nn.bs);
            free(nn.as);
            fclose(file);
            exit(1);
        }

        // Load biases dimensions
        if (fread(&rows, sizeof(size_t), 1, file) != 1 ||
            fread(&cols, sizeof(size_t), 1, file) != 1) {
            fprintf(stderr, "Failed to read bias dimensions for layer %zu.\n", i);
            // Free previously allocated memory before exiting
            for (size_t j = 0; j <= i; j++) {
                mat_free(nn.ws[j]);
                if (j < i) {
                    mat_free(nn.bs[j]);
                }
            }
            free(nn.ws);
            free(nn.bs);
            free(nn.as);
            fclose(file);
            exit(1);
        }

        // Allocate and load biases
        nn.bs[i] = mat_alloc(rows, cols);
        if (fread(nn.bs[i].es, sizeof(float), rows * cols, file) != rows * cols) {
            fprintf(stderr, "Failed to read bias data for layer %zu.\n", i);
            // Free previously allocated memory before exiting
            for (size_t j = 0; j <= i; j++) {
                mat_free(nn.ws[j]);
                mat_free(nn.bs[j]);
            }
            free(nn.ws);
            free(nn.bs);
            free(nn.as);
            fclose(file);
            exit(1);
        }
    }

    // Initialize activations
    for (size_t i = 0; i < nn.size + 1; i++) {
        size_t cols;
        if (i == 0) {
            cols = nn.ws[0].rows; // Input layer size
        } else {
            cols = nn.ws[i - 1].cols; // Current layer size
        }
        nn.as[i] = mat_alloc(1, cols);
    }

    fclose(file);
    return nn;
}

// Predict the output for a given input
int nn_predict(NN nn, Mat input) {
    mat_copy(nn.as[0], input);
    nn_forward(nn);

    // Find the index with the highest activation
    float max_val = MAT_AT(nn.as[nn.size], 0, 0);
    int predicted = 0;
    for (size_t j = 1; j < nn.as[nn.size].cols; j++) {
        if (MAT_AT(nn.as[nn.size], 0, j) > max_val) {
            max_val = MAT_AT(nn.as[nn.size], 0, j);
            predicted = j;
        }
    }
    return predicted;
}