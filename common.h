#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <float.h>

void validate_result(float* c_result, const float* torch_ref, const char* name, int num_elements, float tolerance=1e-4) {
    int nfaults = 0;

    for (int i = 0; i < num_elements; i++) {
        // Check for NaNs
        if(isnan(c_result[i]) || isnan(torch_ref[i])) {
            printf("NaN detected in %s at %d: Yours: %10f  vs  Golden: %10f\n", name, i, c_result[i], torch_ref[i]);
            nfaults ++;
            if (nfaults >= 10) {
                break;
            }
            continue;
        }
        // Check for infinities
        if(isinf(c_result[i]) || isinf(torch_ref[i])) {
            printf("Infinity detected in %s at %d: Yours: %10f  vs  Golden: %10f\n", name, i, c_result[i], torch_ref[i]);
            nfaults ++;
            if (nfaults >= 10) {
                break;
            }
            continue;
        }

        // effective tolerance is based on expected rounding error (FLT_EPSILON),
        // plus any specified additional tolerance
        float t_eff = tolerance + fabs(torch_ref[i]) * FLT_EPSILON;
        // ensure correctness for all elements.
        if (fabs(c_result[i] - torch_ref[i]) > t_eff) {
            printf("Mismatch of %s at %d: Yours: %10f  vs  Golden: %10f\n", name, i, c_result[i], torch_ref[i]);
            nfaults ++;
            if (nfaults >= 10) {
                break;
            }
        }
    }

    if (nfaults > 0) {
        printf("Validation of %s failed with %d errors!\n", name, nfaults);
        exit(EXIT_FAILURE);
    }

    printf("Validation of %s passed!\n", name);
}

void load_txt(const char *filename, float *arr, int num_elements) {
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        perror("Error opening txt file");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < num_elements; i++) {
        if (fscanf(file, "%f", &arr[i]) != 1) {
            fprintf(stderr, "Failed to read float from file\n");
            break;
        }
    }

    fclose(file);
}

void save_txt(const char *filename, float *arr, int num_elements) {
    FILE *file = fopen(filename, "w");
    for (int i = 0; i < num_elements; i++) {
        fprintf(file, "%f\n", arr[i]);
    }
    fclose(file);
}

void save_bin(const char *filename, float *arr, int num_elements) {
    FILE *file = fopen(filename, "wb");
    fwrite(arr, sizeof(float), num_elements, file);
    fclose(file);
}

void load_bin(const char *filename, float *arr, int num_elements) {
    FILE *file = fopen(filename, "rb");
    if (file == NULL) {
        perror("Error opening bin file");
        exit(EXIT_FAILURE);
    }

    fread(arr, sizeof(float), num_elements, file);
    fclose(file);
}
