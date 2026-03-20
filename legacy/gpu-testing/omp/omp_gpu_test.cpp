#include <cstdio>
#include <omp.h>

int main() {
    int is_gpu = 0;

#pragma omp target map(from:is_gpu)
    {
        is_gpu = omp_is_initial_device() ? 0 : 1;
    }

    printf("Running on: %s\n", is_gpu ? "GPU" : "CPU");
}

