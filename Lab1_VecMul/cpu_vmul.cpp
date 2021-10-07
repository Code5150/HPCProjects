//
// Created by Vladislav on 07.10.2021.
//
#include "cpu_vmul.h"

float cpuArrayReduce(const float* a, int size) {
    float result = 0.0f;
    double start = omp_get_wtime();
#pragma omp parallel for simd schedule(static) shared(size, a) reduction(+:result) default(none)
    for(int i = 0; i < size; i += 4) {
        result += a[i] + a[i+1] + a[i+2] + a[i+3];
    }
    double end = omp_get_wtime();
    printf("Время выполнения на CPU: %.6f с\n", end - start);
    return result;
}

