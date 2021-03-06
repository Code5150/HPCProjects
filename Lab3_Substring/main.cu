#include <iostream>
#include "cuda_runtime.h"
#include <cstdint>
#include <vector>
#include <string>
#include <thrust/random.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/device_malloc.h>
#include <thrust/random/linear_congruential_engine.h>
#include <thrust/random/uniform_int_distribution.h>
#include <thrust/for_each.h>
#include <thrust/tuple.h>
#include <thrust/iterator/counting_iterator.h>
#include "CpuSubstring.h"
#include "SubstringSymbolPos.hpp"

#define BYTECAST(X) static_cast<uint8_t>(X)

//#define LAB3_DBG

using namespace thrust::placeholders;

enum SearchType {
    ANY, ALL_POS
};

__global__ void searchSubstrings(const SubstringSymbolPos* positions, uint32_t positionsSize, const char* stringBuffer,
                                 uint32_t stringBufferSize, uint32_t* resultMatrix) {
    auto index = blockDim.x * blockIdx.x + threadIdx.x;
    for (uint32_t i = 0; i < positionsSize; ++i) {
        if (stringBuffer[index] == positions[i].symbol) {
            auto resultIndex = positions[i].substringNum * stringBufferSize + index - positions[i].positionInSubstring;
#ifdef LAB3_DBG
            printf("Substring %d, buffer symbol %c, symbol %c, index %d, decrementing pos %d, pos value %d\n",
                   i, stringBuffer[index], positions[i].symbol, index, resultIndex, resultMatrix[resultIndex]);
#endif
            atomicSub(&resultMatrix[resultIndex], static_cast<uint32_t>(1));
        }
    }
}

//Генерирует случайный символ
struct bytegen {
    __device__ char operator () (int idx)
    {
        thrust::default_random_engine randEng;
        //randEng.seed(8657863);
        thrust::uniform_int_distribution<char> uniDist(97, 122);
        randEng.discard(idx);
        return uniDist(randEng);
    }
};

bool checkResults(std::vector<uint32_t>& a, std::vector<uint32_t>& b) {
    bool result = false;
    if (a.size() == b.size()) {
        result = true;
        for (uint32_t i = 0; i < a.size(); ++i) {
            if (a[i] != b[i]) {
                result = false;
                break;
            }
        }
    }
    return result;
}

enum Options {
    NONE,
    EXIT = 0,
    SUBSTR = 1,
    DEVICE_INFO = 2
};

int main() {
    setlocale(LC_ALL, "ru_RU.UTF-8");

    uint32_t searchStringNum;
    std::cout << "Введите количество строк:";
    std::cin >> searchStringNum;
    auto searchStrings = std::vector<std::string>(searchStringNum);
    auto positions = std::vector<SubstringSymbolPos>();
        for (auto &s: searchStrings) {
            std::cout << "Введите строку:";
            std::cin >> s;
        }
#ifdef LAB3_DBG
        for (auto &s: searchStrings) {
            std::cout << s << std::endl;
        }
#endif
        for (uint32_t i = 0; i < searchStringNum; ++i) {
            uint32_t length = searchStrings[i].length();
            for (uint32_t j = 0; j < length; ++j) {
                positions.emplace_back(searchStrings[i][j], i, j);
            }
        }
#ifdef LAB3_DBG
        for (auto &p: positions) {
            auto[s, sn, pos] = p;
            std::cout << "symbol " << s << ", substringNum " << sn << ", positionInSubstring " << pos << std::endl;
        }
#endif
    uint32_t stringBufferSize;
    uint32_t blockSize;

    std::cout << "Введите размер буфера:";
    std::cin >> stringBufferSize;

    std::cout << "Введите размер блока:";
    std::cin >> blockSize;

    //Заполняем буфер случайными символами
    thrust::device_vector<char> stringBuffer(stringBufferSize);
    thrust::transform(thrust::device,
            thrust::make_counting_iterator((uint32_t)0),
            thrust::make_counting_iterator(stringBufferSize),
            stringBuffer.begin(),
            bytegen());

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    thrust::device_vector<SubstringSymbolPos> devicePositions(positions);
    //Создаём рабочую матрицу
    thrust::device_vector<uint32_t> resultMatrix(searchStringNum * stringBufferSize);
    uint32_t startPos, length;
    for (uint32_t i = 0; i < searchStringNum; ++i) {
        startPos = i * stringBufferSize;
        length = searchStrings[i].length();
        thrust::fill_n(thrust::device, resultMatrix.begin() + startPos, stringBufferSize, length);
    }
#ifdef LAB3_DBG
    for (uint32_t i = 0; i < searchStringNum; ++i) {
        std::cout << "[" << i << ", " << 0 << "]: " << resultMatrix[i * stringBufferSize] << std::endl;
    }
    std::cout << "last: " << resultMatrix[(searchStringNum*stringBufferSize) - 1] << std::endl;
#endif
    //Получаем сырой указатель на буфер
    auto rawBuffer = static_cast<const char*>(thrust::raw_pointer_cast(&stringBuffer[0]));
    auto rawPos = static_cast<SubstringSymbolPos*>(thrust::raw_pointer_cast(&devicePositions[0]));
    auto rawMatrix = static_cast<uint32_t*>(thrust::raw_pointer_cast(&resultMatrix[0]));
    searchSubstrings<<<stringBufferSize/blockSize, blockSize>>>(rawPos, positions.size(), rawBuffer, stringBufferSize, rawMatrix);

    int foundCount = thrust::count(resultMatrix.begin(), resultMatrix.end(), 0);
    thrust::device_vector<uint32_t> foundPos(foundCount);

    thrust::copy_if(
            thrust::device,
            thrust::make_counting_iterator((uint32_t)0),
            thrust::make_counting_iterator(stringBufferSize * searchStringNum),
            resultMatrix.begin(),
            foundPos.begin(),
            _1 == 0
    );
    thrust::sort(thrust::device, foundPos.begin(), foundPos.end());

    auto resultStlVector = std::vector<uint32_t>(foundCount);
    thrust::copy(foundPos.begin(), foundPos.end(), resultStlVector.begin());

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime , start, stop);
    printf("Время выполнения на GPU: %.6f мс\n", elapsedTime);
#ifdef LAB3_DBG
    for (auto& i: resultStlVector) {
        std::cout << "Строка " << i / stringBufferSize << " с позиции " << i % stringBufferSize << std::endl;
    }
#endif
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // copy a device_vector into an STL vector
    auto str = std::vector<char>(stringBufferSize);
    thrust::copy(stringBuffer.begin(), stringBuffer.end(), str.begin());
#ifdef LAB3_DBG
    std::cout << std::string(str.begin(), str.end()) << std::endl;
#endif
    for (uint32_t i = 0; i < searchStringNum; ++i) {
        startPos = i * stringBufferSize;
        length = searchStrings[i].length();
        thrust::fill_n(thrust::device, resultMatrix.begin() + startPos, stringBufferSize, length);
    }

    auto cpuResultMatrix = std::vector<uint32_t>(stringBufferSize*searchStringNum);
    thrust::host_vector<uint32_t> hostResultMatrix = resultMatrix;
    thrust::copy(hostResultMatrix.begin(), hostResultMatrix.end(), cpuResultMatrix.begin());
    auto cpuResult = cpuSearchSubstrings(positions.data(), positions.size(), str.data(), stringBufferSize, cpuResultMatrix.data(), cpuResultMatrix.size());

    if (checkResults(resultStlVector, cpuResult)) {
        std::cout << "Результат на CPU и GPU не расходится" << std::endl;
    } else {
        std::cerr << "ВНИМАНИЕ!!! Результат на CPU и GPU расходится!" << std::endl;
    }

    return 0;
}
