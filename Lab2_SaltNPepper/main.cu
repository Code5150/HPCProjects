#include "cuda_runtime.h"
#include "EBMP/EasyBMP.h"
#include <iostream>
#include <ctime>
#include <algorithm>

texture<float, cudaTextureType2D, cudaReadModeElementType> texRef;

#define F32CAST(X) static_cast<float>(X)

constexpr const char *INPUT_PATH = R"(C:\Users\Vladislav\Desktop\HPCProjects\Lab2_SaltNPepper\lab_img\whiskers-wallpaper-480x320.bmp)";
constexpr const char *GPU_OUT_PATH = R"(C:\Users\Vladislav\Desktop\HPCProjects\Lab2_SaltNPepper\lab_img\GPUoutCat.bmp)";
constexpr const char *CPU_OUT_PATH = R"(C:\Users\Vladislav\Desktop\HPCProjects\Lab2_SaltNPepper\lab_img\CPUoutCat.bmp)";
constexpr const char *NOISE_OUT_PATH = R"(C:\Users\Vladislav\Desktop\HPCProjects\Lab2_SaltNPepper\lab_img\NoiseCat.bmp)";

void saveImage(float *image, int height, int width, bool gpuAlgorithm) {
    BMP output;
    output.SetSize(width, height);

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            RGBApixel pixel;
            pixel.Red = image[i * width + j];
            pixel.Green = image[i * width + j];
            pixel.Blue = image[i * width + j];
            output.SetPixel(j, i, pixel);
        }
    }
    output.WriteToFile(gpuAlgorithm ? GPU_OUT_PATH : CPU_OUT_PATH);
}

void noiseImg(float *image, int height, int width, int per) {
    BMP output;
    output.SetSize(width, height);

    int pixelCount = int(height * width / 100 * per);

    while (pixelCount > 0) {
        int i = rand() % height;
        int j = rand() % width;
        int c = rand() % 2;

        if (c == 1)
            image[i * width + j] = 255;
        else
            image[i * width + j] = 0;
        pixelCount--;
    }

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            RGBApixel pixel;
            pixel.Red = image[i * width + j];
            pixel.Green = image[i * width + j];
            pixel.Blue = image[i * width + j];
            output.SetPixel(j, i, pixel);
        }
    }
    output.WriteToFile(NOISE_OUT_PATH);
}

void medianFilterCPU(const float *image, float *result, int height, int width) {
    //mask3x3
    int m = 3;
    int n = 3;
    int mean = m * n / 2;
    int pad = m / 2;

    float *expandImageArray = (float *) calloc((height + 2 * pad) * (width + 2 * pad), sizeof(float));

    for (int j = 0; j < height; j++) {
        for (int i = 0; i < width; i++) {
            expandImageArray[(j + pad) * (width + 2 * pad) + i + pad] = image[j * width + i];
        }
    }

    for (int j = 0; j < height; j++) {
        for (int i = 0; i < width; i++) {
            float *window = (float *) calloc(m * n, sizeof(float));

            for (int k = 0; k < m; k++) {
                for (int t = 0; t < n; t++) {
                    window[k * n + t] = expandImageArray[j * (width + 2 * pad) + i + k * (width + 2 * pad) + t];
                }
            }

            bool swapped = true;
            int t = 0;
            int tmp;

            while (swapped) {
                swapped = false;
                t++;
                for (int i = 0; i < m * n - t; i++) {
                    if (window[i] > window[i + 1]) {
                        tmp = window[i];
                        window[i] = window[i + 1];
                        window[i + 1] = tmp;
                        swapped = true;
                    }
                }
            }
            result[j * width + i] = window[mean];
        }
    }
}

__global__ void myFilter(float *output, int imageWidth, int imageHeight) {

    auto col = blockIdx.x * blockDim.x + threadIdx.x;
    auto row = blockIdx.y * blockDim.y + threadIdx.y;

    // mask 3x3
    float window[9];
    int m = 3;
    int n = 3;
    int mean = m * n / 2;
    int pad = m / 2;

    for (int i = -pad; i <= pad; i++) {
        for (int j = -pad; j <= pad; j++) {
            window[(i + pad) * n + j + pad] = tex2D(texRef, F32CAST(col + j), F32CAST(row + i));
        }
    }

    bool swapped = true;
    int t = 0;
    int tmp;

    while (swapped) {
        swapped = false;
        t++;
        for (int i = 0; i < m * n - t; i++) {
            if (window[i] > window[i + 1]) {
                tmp = window[i];
                window[i] = window[i + 1];
                window[i + 1] = tmp;
                swapped = true;
            }
        }
    }
    output[row * imageWidth + col] = window[mean];
}

int main() {
    setlocale(LC_ALL, "ru_RU.UTF-8");

    int nIter = 100;
    BMP Image;
    Image.ReadFromFile(INPUT_PATH);
    int height = Image.TellHeight();
    int width = Image.TellWidth();

    auto *imageArray = new float[height * width];
    auto *outputCPU = new float[height * width];
    auto *outputGPU = new float[height * width];
    float *outputDevice;


    for (int j = 0; j < Image.TellHeight(); j++) {
        for (int i = 0; i < Image.TellWidth(); i++) {
            imageArray[j * width + i] = Image(i, j)->Red;
        }
    }

    noiseImg(imageArray, height, width, 8);

    unsigned int start_time = clock();

    for (int j = 0; j < nIter; j++) {
        medianFilterCPU(imageArray, outputCPU, height, width);
    }

    unsigned int elapsedTime = clock() - start_time;
    float msecPerMatrixMulCpu = (elapsedTime / CLOCKS_PER_SEC) / nIter;
    printf("Время выполнения на CPU: %.6f с\n", msecPerMatrixMulCpu);

    // Allocate CUDA array in device memory

    //Returns a channel descriptor with format f and number of bits of each component x, y, z, and w
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
    cudaArray *cu_arr;

    cudaMallocArray(&cu_arr, &channelDesc, width, height);
    cudaMemcpy(cu_arr, imageArray, height * width * sizeof(float), cudaMemcpyHostToDevice);

    // set texture parameters
    texRef.addressMode[0] = cudaAddressModeClamp;
    texRef.addressMode[1] = cudaAddressModeClamp;
    texRef.filterMode = cudaFilterModePoint;

    // Bind the array to the texture
    cudaBindTextureToArray(texRef, cu_arr, channelDesc);

    cudaMalloc(&outputDevice, height * width * sizeof(float));

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // start record
    cudaEventRecord(start, 0);

    for (int j = 0; j < nIter; j++) {
        myFilter <<<blocksPerGrid, threadsPerBlock >>>(outputDevice, width, height);
    }

    // stop record
    cudaEventRecord(stop, 0);

    // wait end of event
    cudaEventSynchronize(stop);

    float msecTotal = 0.0f;
    cudaEventElapsedTime(&msecTotal, start, stop);
    float msecPerMatrixMul = msecTotal / nIter;

    printf("Время выполнения на GPU: %.6f мс\n", msecPerMatrixMul);

    cudaDeviceSynchronize();

    cudaMemcpy(outputGPU, outputDevice, height * width * sizeof(float), cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();

    saveImage(outputGPU, height, width, true);
    saveImage(outputCPU, height, width, false);

    cudaFreeArray(cu_arr);
    cudaFree(outputDevice);
    return 0;
}
