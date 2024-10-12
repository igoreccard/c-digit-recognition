#ifndef IMAGE_H
#define IMAGE_H


#include "stb_image.h"

#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include <sys/stat.h>
#include <dirent.h>


typedef struct {
    float **images;
    int *image_sizes;
    int *labels;
    int count;
    int num_classes;
} Dataset;

// Your directory should be organized like: DATASET/CLASS_NAMES/IMAGES
// For each subfolder in DATASET/ a diferent class name will be created
// Each image has an individual label, if the label of an image is 3, it means the expected output should be [0, 0, 0, 3, 0 ,0 ..., 0]
// coutn is the number of images in a directory


int count_images_in_directory(const char *dir_name);

float *load_image(const char *filename, int *width, int *height, int *channels);

float **process_directory(const char *dir_name, int *count, int **image_sizes);

void print_images_data(float **images, int *image_sizes, int count);

int get_class_label(const char *class_name, char class_names[][256], int *num_classes);

Dataset* process_directory_with_labels(const char *dir_name);

#endif
