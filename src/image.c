#define STB_IMAGE_IMPLEMENTATION
#include "image.h"

#define MAX_CLASSES 100


// Returns the number of images in a directory
int count_images_in_directory(const char *dir_name) {
    DIR *dir = opendir(dir_name);
    struct dirent *entry;
    int count = 0;

    if (!dir) {
        printf("Error: Could not open directory %s.\n", dir_name);
        return -1;
    }

    while ((entry = readdir(dir)) != NULL) {
        char path[1024];
        snprintf(path, sizeof(path), "%s/%s", dir_name, entry->d_name);

        struct stat statbuf;
        if (stat(path, &statbuf) == 0 && S_ISREG(statbuf.st_mode)) {
            const char *ext = strrchr(entry->d_name, '.');
            if (ext != NULL && (strcmp(ext, ".jpg") == 0)/*|| strcmp(ext, ".jpeg") == 0 ||
                                strcmp(ext, ".png") == 0 || strcmp(ext, ".bmp") == 0 ||
                                strcmp(ext, ".gif") == 0)*/) {
                count++;
            }
        }
    }
    closedir(dir);
    return count;
}

float *load_image(const char *filename, int *width, int *height, int *channels) {
    unsigned char *img = stbi_load(filename, width, height, channels, 3);
    if (img == NULL) {
        printf("Error: Could not load image %s.\n", filename);
        return NULL;
    }

    int img_size = (*width) * (*height) * 3;
    float *image_data = malloc(img_size * sizeof(float));
    if (!image_data) {
        fprintf(stderr, "Failed to allocate memory for image data.\n");
        stbi_image_free(img);
        return NULL;
    }

    for (int i = 0; i < img_size; i++) {
        image_data[i] = (float)img[i] / 255.0f; // Normalization
    }

    stbi_image_free(img);
    return image_data;
}

// Iterate through a directory and store the images loaded into an array of float pointers
float **process_directory(const char *dir_name, int *count, int **image_sizes) {
    *count = count_images_in_directory(dir_name);
    if (*count < 0) return NULL; // Return NULL on error

    float **images = malloc(*count * sizeof(float *));
    if (!images) {
        fprintf(stderr, "Failed to allocate memory for images array.\n");
        return NULL;
    }
    *image_sizes = malloc(*count * sizeof(int)); // Store the size of each image in an array
    if (!*image_sizes) {
        fprintf(stderr, "Failed to allocate memory for image_sizes array.\n");
        free(images);
        return NULL;
    }

    DIR *dir = opendir(dir_name);
    struct dirent *entry;

    if (!dir) {
        printf("Error: Could not open directory %s.\n", dir_name);
        free(images);
        free(*image_sizes);
        return NULL;
    }

    int i = 0;
    while ((entry = readdir(dir)) != NULL && i < *count) {
        char path[1024];
        snprintf(path, sizeof(path), "%s/%s", dir_name, entry->d_name);

        struct stat statbuf;
        if (stat(path, &statbuf) == 0 && S_ISREG(statbuf.st_mode)) {
            const char *ext = strrchr(entry->d_name, '.');
            if (ext != NULL && (strcmp(ext, ".jpg") == 0 || strcmp(ext, ".jpeg") == 0 ||
                                strcmp(ext, ".png") == 0 || strcmp(ext, ".bmp") == 0 ||
                                strcmp(ext, ".gif") == 0)) {
                int width, height, channels;
                float *image = load_image(path, &width, &height, &channels);
                if (image == NULL) {
                    fprintf(stderr, "Failed to load image: %s\n", path);
                    continue; // Skip this image and continue
                }
                images[i] = image;
                (*image_sizes)[i] = width * height * 3; // Store the size of the image (RGB)
                i++;
            }
        }
    }
    closedir(dir);

    // If some images failed to load, adjust the count
    *count = i;

    return images;
}

//  Print all images data
void print_images_data(float **images, int *image_sizes, int count) {
    for (int i = 0; i < count; i++) {
        printf("Image %d data (Size: %d pixels):\n", i + 1, image_sizes[i] / 3);

        for (int j = 0; j < image_sizes[i]; j += 3) {
            printf("Pixel %d: R=%.2f, G=%.2f, B=%.2f\n", j / 3, images[i][j], images[i][j + 1], images[i][j + 2]);
        }

        printf("\n");
    }
}

// Map class names to integer labels
int get_class_label(const char *class_name, char class_names[][256], int *num_classes) {
    for (int i = 0; i < *num_classes; i++) {
        if (strcmp(class_name, class_names[i]) == 0) {
            return i;
        }
    }
    // If not found, add to class_names
    strcpy(class_names[*num_classes], class_name);
    (*num_classes)++;
    return (*num_classes - 1);
}

// 
Dataset* process_directory_with_labels(const char *dir_name) {
    Dataset *dataset = malloc(sizeof(Dataset));
    if (!dataset) {
        fprintf(stderr, "Failed to allocate memory for dataset.\n");
        exit(1);
    }

    dataset->images = NULL;
    dataset->image_sizes = NULL;
    dataset->labels = NULL;
    dataset->count = 0;
    dataset->num_classes = 0;

    char class_names[MAX_CLASSES][256]; // Store unique class names

    DIR *main_dir = opendir(dir_name);
    struct dirent *class_entry;

    if (!main_dir) {
        printf("Error: Could not open main directory %s.\n", dir_name);
        free(dataset);
        return NULL;
    }

    while ((class_entry = readdir(main_dir)) != NULL) {
        if (class_entry->d_type == DT_DIR && strcmp(class_entry->d_name, ".") != 0 && strcmp(class_entry->d_name, "..") != 0) {
            // For each subdirectory / class
            char class_path[1024];
            snprintf(class_path, sizeof(class_path), "%s/%s", dir_name, class_entry->d_name);

            // Get the class subfolder name / label
            int class_label = get_class_label(class_entry->d_name, class_names, &dataset->num_classes);

            // Process the images in the subfolder
            DIR *img_dir = opendir(class_path);
            struct dirent *img_entry;

            while ((img_entry = readdir(img_dir)) != NULL) {
                if (img_entry->d_type == DT_REG) {
                    // If it's a regular file
                    char img_path[1024];
                    snprintf(img_path, sizeof(img_path), "%s/%s", class_path, img_entry->d_name);

                    // Load the image
                    int width, height, channels;
                    float *image = load_image(img_path, &width, &height, &channels);
                    if (image == NULL) {
                        fprintf(stderr, "Failed to load image: %s\n", img_path);
                        continue;
                    }

                    // Reallocate memory to store this image
                    dataset->images = realloc(dataset->images, (dataset->count + 1) * sizeof(float *));
                    dataset->image_sizes = realloc(dataset->image_sizes, (dataset->count + 1) * sizeof(int));
                    dataset->labels = realloc(dataset->labels, (dataset->count + 1) * sizeof(int));

                    // Store the image and its size
                    dataset->images[dataset->count] = image;
                    dataset->image_sizes[dataset->count] = width * height * 3; // RGB channels
                    dataset->labels[dataset->count] = class_label;

                    dataset->count++;
                }
            }

            closedir(img_dir); // Close image directory after processing
        }
    }

    closedir(main_dir); // Close the main dataset directory

    return dataset;
}
