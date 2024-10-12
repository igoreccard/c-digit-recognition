#include "matrix.h"

// Return a random float
float rand_float(void){
    return (float)rand()/(float)RAND_MAX;
}

// Allocate memory for your matrix
Mat mat_alloc(size_t rows, size_t cols){
    Mat m;
    m.rows = rows;
    m.cols = cols;
    m.stride = cols;
    m.es = calloc(rows*cols, sizeof(*m.es));
    assert(m.es != NULL);
    return m;
}

// Randomize your matrix indicies values
void mat_rand(Mat m, float low, float high) {
    for (size_t j = 0; j < m.rows; j++) {
        for (size_t i = 0; i < m.cols; i++) {
            MAT_AT(m, j, i) = low + rand_float() * (high - low);
        }
    }
}

// Return a matrix 1xm.rows
Mat mat_row(Mat m, size_t row){
    return (Mat){
        .rows = 1,
        .cols = m.cols,
        .es = &MAT_AT(m, row, 0)
    };
}

// Copy src matrix into dst matrix
void mat_copy(Mat dst, Mat src) {
    assert(dst.rows == src.rows);
    assert(dst.cols == src.cols);
    
    for(size_t j = 0; j < src.rows; j++) {
        for(size_t i = 0; i < src.cols; i++) {
            MAT_AT(dst, j, i) = MAT_AT(src, j, i);
        }
    }
}


// Multiply two matrices
void mat_dot(Mat dst, Mat a, Mat b){
    assert(a.cols == b.rows);

    assert(dst.cols == b.cols);
    assert(dst.rows == a.rows);

    for(size_t i = 0; i < dst.rows; i++){
        for(size_t j = 0; j < dst.cols; j++){
            MAT_AT(dst, i, j) = 0;
            for(size_t k = 0; k < a.cols; k++){
                MAT_AT(dst, i, j) += MAT_AT(a, i, k)*MAT_AT(b, k, j); // a[0 0]* b[0 0] + a[0 1]*b[1 0]
            }
        }
    }
}

// Add 2 matrices
void mat_sum(Mat dst, Mat a)
{
    assert(dst.rows == a.rows);
    assert(dst.cols == a.cols);
    for (size_t i = 0; i < dst.rows; ++i) {
        for (size_t j = 0; j < dst.cols; ++j) {
            MAT_AT(dst, i, j) += MAT_AT(a, i, j);
        }
    }
}

// Subtract 2 matrices
void mat_subtract(Mat dst, Mat a){
    assert(dst.rows == a.rows);
    assert(dst.cols == a.cols);
    for(size_t i = 0; i < dst.rows; i++){
        for(size_t j = 0; j < dst.cols; j++){
            MAT_AT(dst,i,j) -= MAT_AT(a, i, j);
        }
    }
}

// Scale the matrix, X = A*X
void mat_scale(Mat dst, float a){
    for(size_t i = 0; i < dst.rows; i++){
        for(size_t j = 0; j < dst.cols; j++){
            MAT_AT(dst,i,j) *= a;
        }
    }
}

// Print all the indicies of a matrix
void mat_print(Mat m, const char *name) {
    // Get the length of the name to specify the blank space
    int name_length = strlen(name);
    
    printf("%*s = [\n", name_length, name);
    
    for (size_t i = 0; i < m.rows; i++) {
        for (size_t j = 0; j < m.cols; j++) {
            printf("%f ", MAT_AT(m, i, j));
        }
        printf("\n");
    }
    
    printf("%*s]\n", name_length, "");
}


// Function to print an array of matrices
void mat_array_print(Mat *mat_array, size_t num_matrices, const char *name) {
    for (size_t i = 0; i < num_matrices; i++) {
        // Create a unique name for each matrix in the array Matrix_0, Matrix_1 etc...
        char matrix_name[256];
        snprintf(matrix_name, sizeof(matrix_name), "%s_%zu", name, i);
        
        // Print the individual matrix
        mat_print(mat_array[i], matrix_name);
    }
}

// Sigmoid function
float sigmoidf(float x)
{
    return 1.f / (1.f + expf(-x));
}

// Derivative of sigmoid function
float dsigmoidf(float x) {
    float exp_neg_x = expf(-x);
    return exp_neg_x / powf(1 + exp_neg_x, 2);
}

// Applies sigmoid function to all indicies of a matrix
void mat_sig(Mat m)
{
    for (size_t i = 0; i < m.rows; ++i) {
        for (size_t j = 0; j < m.cols; ++j) {
            MAT_AT(m, i, j) = sigmoidf(MAT_AT(m, i, j));
        }
    }
}

// Applies the derivative of sigmoid function to all indicies of a matrix
void mat_dsig(Mat m) {
    for (size_t i = 0; i < m.rows; ++i) {
        for (size_t j = 0; j < m.cols; ++j) {
            float sig = sigmoidf(MAT_AT(m, i, j));
            MAT_AT(m, i, j) = sig * (1.0f - sig);
        }
    }
}

// Free the memory of a matrix
void mat_free(Mat m) {
    free(m.es);
    m.es = NULL;
}

// Red hot chili pepper ass name
Mat one_hot_encode(int label, int num_classes) {
    Mat encoded = mat_alloc(1, num_classes);
    for (int i = 0; i < num_classes; i++) {
        MAT_AT(encoded, 0, i) = (i == label) ? 1.0f : 0.0f;
    }
    return encoded;
}


float reluf(float x) {
    return x > 0 ? x : 0;
}

float drelu(float x) {
    return x > 0 ? 1 : 0;
}

void mat_relu(Mat m) {
    for (size_t i = 0; i < m.rows; ++i) {
        for (size_t j = 0; j < m.cols; ++j) {
            MAT_AT(m, i, j) = reluf(MAT_AT(m, i, j));
        }
    }
}

void mat_drelu(Mat m) {
    for (size_t i = 0; i < m.rows; ++i) {
        for (size_t j = 0; j < m.cols; ++j) {
            MAT_AT(m, i, j) = drelu(MAT_AT(m, i, j));
        }
    }
}
