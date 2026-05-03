#include <random>
#include <iostream>
#include <chrono>

// I assume this initialization logic should be kept on the CPU
void init(int32_t numElements, int32_t *vec_a, int32_t *vec_b, int32_t *mat)
{
    std::mt19937 prng(2024);
    std::uniform_int_distribution<int32_t> distrib(-16, 16);

    for (auto i = 0; i < numElements; i++)
    {
        vec_a[i] = distrib(prng);
        vec_b[i] = distrib(prng);
    }

    for (auto i = 0; i < numElements * numElements; i++)
        mat[i] = distrib(prng);
}

// from lecture example
void check_cuda(cudaError_t err, std::string msg)
{
    if (err == cudaSuccess)
    {
        return;
    }
    std::cerr << msg << " (error code " << cudaGetErrorString(err) << ")\n";
    exit(EXIT_FAILURE);
}

__global__ void vectorAdd(int32_t *vec_a, int32_t *vec_b, int32_t *out, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= numElements)
    {
        return;
    }

    out[i] = vec_a[i] + vec_b[i];
}

__global__ void vecMatMult(int32_t *vec, int32_t *mat, int32_t *out, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= numElements)
    {
        return;
    }

    // Note that I originally wanted to skip this loop by creating a thread per
    // matrix entry (instead of per vector entry), but launching 32768x32768 threads is probably not a good idea :)

    int32_t sum = 0;
    for (int j = 0; j < numElements; j++)
    {
        sum += vec[j] * mat[i * numElements + j];
    }
    out[i] = sum;
}

int main(int argc, char **argv)
{
    cudaError_t err = cudaSuccess;

    int32_t numElements = 32768;
    size_t vec_size = sizeof(int32_t) * numElements;
    size_t mat_size = sizeof(int32_t) * numElements * numElements;

    // 1. init host vectors and matrix
    auto h_vec_a = (int32_t *)malloc(vec_size);
    auto h_vec_b = (int32_t *)malloc(vec_size);
    auto h_mat = (int32_t *)malloc(mat_size);
    auto h_out = (int32_t *)malloc(vec_size);

    init(numElements, h_vec_a, h_vec_b, h_mat);

    // 2. malloc and copy to device
    int32_t *d_vec_a = NULL;
    err = cudaMalloc((void **)&d_vec_a, vec_size);
    check_cuda(err, "Failed to allocate device vector a");
    err = cudaMemcpy(d_vec_a, h_vec_a, vec_size, cudaMemcpyHostToDevice);
    check_cuda(err, "Failed to copy vector a from host to device");

    int32_t *d_vec_b = NULL;
    err = cudaMalloc((void **)&d_vec_b, vec_size);
    check_cuda(err, "Failed to allocate device vector b");
    err = cudaMemcpy(d_vec_b, h_vec_b, vec_size, cudaMemcpyHostToDevice);
    check_cuda(err, "Failed to copy vector b from host to device");

    int32_t *d_mat = NULL;
    err = cudaMalloc((void **)&d_mat, mat_size);
    check_cuda(err, "Failed to allocate device matrix");
    err = cudaMemcpy(d_mat, h_mat, mat_size, cudaMemcpyHostToDevice);
    check_cuda(err, "Failed to copy matrix from host to device");

    int32_t *d_vec_tmp = NULL;
    err = cudaMalloc((void **)&d_vec_tmp, vec_size);
    check_cuda(err, "Failed to allocate device vector c");

    int32_t *d_out = NULL;
    err = cudaMalloc((void **)&d_out, vec_size);
    check_cuda(err, "Failed to allocate device vector c");

    // 3. launch kernel

    // 3.1. read launch config
    int threadsPerBlock = argc > 1 ? std::atoi(argv[1]) : 256;
    int blocksPerGrid = argc > 2 ? std::atoi(argv[2]) : (numElements + threadsPerBlock - 1) / threadsPerBlock;

    if (threadsPerBlock <= 0 || blocksPerGrid <= 0 ||
        threadsPerBlock * blocksPerGrid < numElements)
    {
        std::cerr << "Invalid launch config!\n";
        exit(EXIT_FAILURE);
    }

    std::cout << "Launching with " << threadsPerBlock << " threads per block and "
                << blocksPerGrid << " blocks per grid.\n";

    cudaDeviceSynchronize(); // make sure gpu is ready
    auto start = std::chrono::system_clock::now();

    // 3.2. addition
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_vec_a, d_vec_b, d_vec_tmp, numElements);
    check_cuda(cudaGetLastError(), "Failed to launch vectorAdd kernel");

    // 3.3 multiplication
    vecMatMult<<<blocksPerGrid, threadsPerBlock>>>(d_vec_tmp, d_mat, d_out, numElements);
    check_cuda(cudaGetLastError(), "Failed to launch vecMatMult kernel");

    cudaDeviceSynchronize(); // wait for gpu to finish
    auto end = std::chrono::system_clock::now();

    // 4. copy result from device to host
    err = cudaMemcpy(h_out, d_out, vec_size, cudaMemcpyDeviceToHost);
    check_cuda(err, "Failed to copy vector out from device to host");

    // 5. Print and clean-up
    std::cout << "First 3 entries of Out Vec:" << std::endl;
    for (int32_t i = 0; i < 3; i++)
        std::cout << h_out[i] << std::endl;

    std::chrono::duration<double> elapsed_seconds = end - start;
    std::cout << "Elapsed time: " << elapsed_seconds.count() << "s" << std::endl;

    err = cudaFree(d_vec_a);
    check_cuda(err, "Failed to free device vector a");
    err = cudaFree(d_vec_b);
    check_cuda(err, "Failed to free device vector b");
    err = cudaFree(d_mat);
    check_cuda(err, "Failed to free device matrix");
    err = cudaFree(d_vec_tmp);
    check_cuda(err, "Failed to free device vector tmp");
    err = cudaFree(d_out);
    check_cuda(err, "Failed to free device vector out");

    free(h_vec_a);
    free(h_vec_b);
    free(h_mat);
    free(h_out);

    return 0;
}