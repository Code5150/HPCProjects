#include <iostream>
#include "cuda_runtime.h"
#include "curand.h"
#include "cpu_vmul.h"

#define ULLCAST(X) static_cast<unsigned long long>(X)
constexpr const int BLOCK_SIZE = 256;
/*__global__ void cuda_vmul(const float* a, const float* b, float* c, int size) {
    int idx =
    c[]
}*/

void verifyResult(const float* gpuRes, int gpuResSize, float cpuRes) {
    float gpuResSum = 0.0f;
    for(int i = 0; i < gpuResSize; ++i) {
        gpuResSum += gpuRes[i];
    }
    if (gpuResSum - cpuRes > 1e-3) printf("Результат некорректный");
    else printf("Результат корректный");
    printf("\n");
}

__global__ void cudaArrayReduce(const float *a, float *result) {
    extern __shared__ float aShared[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    aShared[tid] = a[i];
    __syncthreads();

    for(unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            aShared[tid] += aShared[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) result[blockIdx.x] = aShared[0];
}

cudaDeviceProp showProperties() {
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);//определение параметров GPU с номером 0

    printf("Device name : %s\n", deviceProp.name);
    printf("Total global memory : %llu MB\n",
           deviceProp.totalGlobalMem / 1024 / 1024);
    printf("Shared memory per block : %zu\n",
           deviceProp.sharedMemPerBlock);
    printf("Registers per block : %d\n",
           deviceProp.regsPerBlock);
    printf("Warp size : %d\n", deviceProp.warpSize);
    printf("Memory pitch : %zu\n", deviceProp.memPitch);
    printf("Max threads per block : %d\n",
           deviceProp.maxThreadsPerBlock);
    printf("Max threads dimensions : x = %d, y = %d, z = %d\n",
           deviceProp.maxThreadsDim[0],
           deviceProp.maxThreadsDim[1],
           deviceProp.maxThreadsDim[2]);
    printf("Max grid size: x = %d, y = %d, z = %d\n",
           deviceProp.maxGridSize[0], deviceProp.maxGridSize[1],
           deviceProp.maxGridSize[2]);
    printf("Clock rate: %d\n", deviceProp.clockRate);
    printf("Total constant memory: %zu\n",
           deviceProp.totalConstMem);
    printf("Compute capability: %d.%d\n",
           deviceProp.major, deviceProp.minor);
    printf("Texture alignment: %zu\n",
           deviceProp.textureAlignment);
    printf("Device overlap: %d\n",
           deviceProp.deviceOverlap);
    printf("Multiprocessor count: %d\n",
           deviceProp.multiProcessorCount);
    printf("Kernel execution timeout enabled: %s\n",
           deviceProp.kernelExecTimeoutEnabled ? "true" :
           "false");
    printf("Can map host memory: %s\n",
           deviceProp.canMapHostMemory ? "true" :
           "false");
    printf("Device has Compute Capability %d.%d\n",
           deviceProp.major, deviceProp.minor);

    return deviceProp;
}

void scan_vec_size(int* size) {
    printf("Введите размер вектора (от 1000 до 1000000): ");
    scanf_s("%d", size);
    while (*size < 1000 || *size > 1048576) {
        printf("Неверный размер вектора. %d не входит в интервал [1000;1000000]\n", *size);
        printf("Введите размер вектора (от 1000 до 1000000): ");
        scanf_s("%d", size);
    }
}

enum Options {
    NONE,
    EXIT = 0,
    RED = 1,
    DEVICE_INFO = 2
};

int main() {
    setlocale(LC_ALL, "ru_RU.UTF-8");

    curandGenerator_t gen;

    float *a, *result;
    float *cpuA, *cpuResult;
    cpuResult = new float(0.0f);

    curandCreateGeneratorHost(&gen, CURAND_RNG_PSEUDO_MTGP32);
    curandSetPseudoRandomGeneratorSeed(gen, ULLCAST(clock()));

    int vec_size = 0;
    Options option = Options::NONE;
    bool main_cycle = true;
    while(main_cycle) {
        printf("Список действий:\n");
        printf("0 - выход из программы\n");
        printf("1 - сумма вектора\n");
        printf("2 - информация о GPU\n");
        printf("Выберите действие:");
        scanf_s("%d", &option);
        switch (option) {
            case Options::RED: {
                scan_vec_size(&vec_size);
                cpuA = new float[vec_size] {};
                cpuResult = new float[vec_size/BLOCK_SIZE] {};

                curandGenerateUniform(gen, cpuA, vec_size);

                cudaMalloc((void**)&a, vec_size * sizeof(float));
                cudaMalloc((void**)&result, (vec_size/BLOCK_SIZE) * sizeof(float));

                cudaMemcpy(a, cpuA, vec_size * sizeof(float), cudaMemcpyHostToDevice);
                cudaMemcpy(result, cpuResult, (vec_size/BLOCK_SIZE) * sizeof(float), cudaMemcpyHostToDevice);

                // Initialize events
                cudaEvent_t start, stop;
                float elapsedTime;

                // Create events
                cudaEventCreate(&start);
                cudaEventCreate(&stop);

                // Record events
                cudaEventRecord(start, nullptr);

                cudaArrayReduce<<<vec_size/BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE*sizeof(float)>>>(a, result);

                cudaEventRecord(stop, nullptr);

                // Waiting to kernel finish
                cudaEventSynchronize(stop);
                cudaEventElapsedTime(&elapsedTime, start, stop);
                printf("Время выполнения на GPU: %.6f мс\n", elapsedTime);

                cudaMemcpy(cpuResult, result, (vec_size/BLOCK_SIZE) * sizeof(float), cudaMemcpyDeviceToHost);
                //printf("Результат: %.6f\n", *cpuResult);

                float r = cpuArrayReduce(cpuA, vec_size);

                printf("Результат: %.6f\n", r);

                verifyResult(cpuResult, vec_size/BLOCK_SIZE, r);

                cudaFree(a);
                cudaFree(result);

                delete[] cpuA;

                break;
            }
            case Options::DEVICE_INFO: {
                showProperties();
                break;
            }
            case Options::EXIT:{
                main_cycle = false;
                break;
            }
            default: {
                printf("Данной опции не существует. Попробуйте ещё раз.\n");
                break;
            }
        }
    }
    return 0;
}
