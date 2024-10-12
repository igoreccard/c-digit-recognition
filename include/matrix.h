#ifndef MATRIX_H
#define MATRIX_H


#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <math.h>


#define MAT_AT(m, row, col) (m.es[(row) * (m.stride) + (col)])


typedef struct{
    size_t rows;
    size_t cols;
    size_t stride;
    float *es;
}Mat;


float rand_float(void);

Mat mat_alloc(size_t rows, size_t cols);

void mat_rand(Mat m, float low, float high);

Mat mat_row(Mat m, size_t row);

void mat_copy(Mat dst, Mat src);

void mat_dot(Mat dst, Mat a, Mat b);

void mat_sum(Mat dst, Mat a);

void mat_scale(Mat dst, float a);

void mat_subtract(Mat dst, Mat a);

void mat_print(Mat m, const char *name);

float sigmoidf(float x);

float dsigmoidf(float x);

void mat_sig(Mat m);

void mat_dsig(Mat m);

void mat_free(Mat m);

Mat* array_to_mat(float **images, int *image_sizes, int count);

void mat_array_print(Mat *mat_array, size_t num_matrices, const char *name);

Mat one_hot_encode(int label, int num_classes);

float reluf(float x);

float drelu(float x);

void mat_relu(Mat m);

void mat_drelu(Mat m);


#endif