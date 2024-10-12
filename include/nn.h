#ifndef NN_H_
#define NN_H_
#include "matrix.h"

#define NN_OUTPUT(nn) (nn).as[(nn).count]

typedef struct{
    size_t size;
    Mat *ws;
    Mat *bs;
    Mat *as;
}NN;


NN nn_alloc(size_t *arch, size_t arch_count);

void nn_print();

void nn_rand(NN nn, float low, float high);

void nn_learn();

void nn_forward(NN nn);

void nn_learn();

float nn_cost(NN nn, Mat training_input, Mat training_output);

void nn_backprop(NN nn, Mat training_input, Mat training_output, float learning_rate);

void nn_free(NN nn);

void nn_save(NN nn, const char *filename);

NN nn_load(const char *filename);

int nn_predict(NN nn, Mat input);

#endif