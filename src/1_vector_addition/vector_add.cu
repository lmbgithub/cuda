#include <stdio.h>
#include <cuda_runtime.h>

// CPU implementation of vector addition
void vectorAddCPU(const float* A, const float* B, float* C, int n) {
    for (int i = 0; i < n; i++) {
        C[i] = A[i] + B[i];
    }
}

// CUDA kernel for vector addition
__global__ void vectorAddGPU(const float* A, const float* B, float* C, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        C[i] = A[i] + B[i];
    }
}

// Function to check CUDA errors
void checkCudaError(cudaError_t error, const char* msg) {
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s: %s\n", msg, cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
}

int main() {
    // Vector size
    const int n = 10e7;
    size_t size = n * sizeof(float);

    // Allocate host memory
    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C = (float*)malloc(size);
    float *h_C_CPU = (float*)malloc(size);

    // Initialize input vectors
    for (int i = 0; i < n; i++) {
        h_A[i] = rand() / (float)RAND_MAX;
        h_B[i] = rand() / (float)RAND_MAX;
    }

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    checkCudaError(cudaMalloc(&d_A, size), "cudaMalloc A");
    checkCudaError(cudaMalloc(&d_B, size), "cudaMalloc B");
    checkCudaError(cudaMalloc(&d_C, size), "cudaMalloc C");

    // Copy data from host to device
    checkCudaError(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice), "cudaMemcpy A");
    checkCudaError(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice), "cudaMemcpy B");

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Record start time
    cudaEventRecord(start);

    vectorAddGPU<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, n);

    // Record stop time
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Calculate GPU execution time
    float gpuTime = 0;
    cudaEventElapsedTime(&gpuTime, start, stop);

    // Copy result back to host
    checkCudaError(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost), "cudaMemcpy C");

    // CPU implementation timing
    clock_t cpuStart = clock();
    vectorAddCPU(h_A, h_B, h_C_CPU, n);
    clock_t cpuEnd = clock();
    float cpuTime = ((float)(cpuEnd - cpuStart)) / CLOCKS_PER_SEC * 1000.0f;

    // Verify results
    bool correct = true;
    for (int i = 0; i < n; i++) {
        if (fabs(h_C[i] - h_C_CPU[i]) > 1e-5) {
            correct = false;
            break;
        }
    }

    // Print results
    printf("Vector size: %d\n", n);
    printf("GPU Time: %.3f ms\n", gpuTime);
    printf("CPU Time: %.3f ms\n", cpuTime);
    printf("Speedup: %.2fx\n", cpuTime / gpuTime);
    printf("Results are %s\n", correct ? "correct" : "incorrect");

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C_CPU);

    return 0;
}
