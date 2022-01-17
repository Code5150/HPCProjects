//
// Created by Vladislav on 17.01.2022.
//
#include "CpuSubstring.h"
#include "omp.h"
#include <vector>
#include <cstdint>
#include <algorithm>

std::vector<uint32_t> cpuSearchSubstrings(const SubstringSymbolPos* positions, uint32_t positionsSize, const char* stringBuffer,
                                          uint32_t stringBufferSize, uint32_t* resultMatrix, uint32_t matrixSize) {
#pragma omp declare reduction (merge : std::vector<uint32_t> : omp_out.insert(omp_out.end(), omp_in.begin(), omp_in.end()))

    double start = omp_get_wtime();
#pragma omp parallel for collapse(2) default(none) shared(matrixSize, resultMatrix, stringBufferSize, positions, positionsSize, stringBuffer)
    for (uint32_t i = 0; i < stringBufferSize; ++i) {
        for (uint32_t j = 0; j < positionsSize; ++j) {
            if (stringBuffer[i] == positions[j].symbol) {
                auto resultIndex = positions[j].substringNum * stringBufferSize + i - positions[j].positionInSubstring;
#pragma omp atomic
                resultMatrix[resultIndex]--;
            }
        }
    }

    std::vector<uint32_t> result;
#pragma omp parallel for default(none) shared(matrixSize, resultMatrix) reduction(merge: result)
    for (uint32_t i = 0; i < matrixSize; ++i) {
        if (resultMatrix[i] == 0) result.push_back(i);
    }
    double end = omp_get_wtime();
    printf("Время выполнения на CPU: %.6f с\n", end - start);

    std::sort(result.begin(), result.end());
    return result;
}