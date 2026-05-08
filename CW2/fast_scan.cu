
#include "helper.cu"

// helper function for accessing number's real part on device
__device__ int real_id(int t_id)
{
    return t_id * 2;
}

// helper function for accessing number's imaginary part on device
__device__ int imag_id(int t_id)
{
    return t_id * 2 + 1;
}

// todo: this probably currently just works for the first block, as we don't have any logic across blocks yet
__global__ void parallel_scan(float *in, float *out, size_t vec_length)
{
    __shared__ int in_s[blockDim.x];
    int t_id = threadIdx.x; // id within block
    int t_global_id = t_id + blockDim.x * blockIdx.x;
    in_s[real_id(t_id)] = in[real_id(t_global_id)];
    in_s[imag_id(t_id)] = in[imag_id(t_global_id)];

    if (t_id >= vec_length / 2) // each thread is responsible for one number, so two vector elements (real + imaginary)
    {
        return;
    }

    // note that we might need to change this for shared memory

    __syncthreads(); // wait until copy finished

    // ref slide 24 and image in task.md.
    // stride is used to reduce threads! first just thread 0, then 0-1, then 0-3, then 0-7, etc.
    for (auto stride = 1; stride < blockDim.x; stride *= 2)
    {
        float tmp_real = 0.0f;
        float tmp_imag = 0.0f;

        // we have to separate read and write operations to make sure every thread read before any writes.

        // accumulate in global memory (for now) using the output to store intermediate results
        if (t_id >= stride)
        {
            tmp_real = in_s[real_id(t_id) - stride * 2];
            tmp_imag = in_s[imag_id(t_id) - stride * 2];
        }

        __syncthreads();

        if (t_id >= stride) {
            in_s[real_id(t_id)] += tmp_real;
            in_s[imag_id(t_id)] += tmp_imag;
        }

        __syncthreads();
    }

    // todo: If not first block, add last numbers of all previous blocks
}

int main()
{
    size_t num_count = 33554432;
    size_t vec_length = num_count * 2;
    float *in_d, *in_h, *out_d, *out_h;

    // 1. allocate vectors
    in_h = (float *)calloc(vec_length, sizeof(float));
    CHECK_ALLOC(in_h);
    out_h = (float *)calloc(vec_length, sizeof(float));
    CHECK_ALLOC(out_h);

    size_t vec_size = vec_length * sizeof(float);

    CUDA_CALL(cudaMalloc((void **)&in_d, vec_size));
    CUDA_CALL(cudaMalloc((void **)&out_d, vec_size));

    // 2. initialize input on host and device
    int e = random_init(vec_length, in_d, in_h);
    if (e == EXIT_FAILURE)
        return EXIT_FAILURE;

    // 3. launch kernel
    // we currently want to launch half as many threads as vec elements (so one for each number)
    int threadsPerBlock = 512;
    int blocksPerGrid = (num_count + threadsPerBlock - 1) / threadsPerBlock;

    cudaDeviceSynchronize();
    auto start = std::chrono::system_clock::now();

    parallel_scan<<<blocksPerGrid, threadsPerBlock>>>(in_d, out_d, vec_length);

    cudaDeviceSynchronize();
    auto end = std::chrono::system_clock::now();

    // 4. copy result from device to host
    CUDA_CALL(cudaMemcpy(out_h, out_d, vec_size, cudaMemcpyDeviceToHost));

    // 5. print and clean-up
    std::cout << "First 5 entries of input:" << std::endl;
    for (int32_t i = 0; i < 5 * 2; i += 2)
        std::cout << in_h[i] << " + " << in_h[i + 1] << "i" << std::endl;
    std::cout << "First 5 entries of output:" << std::endl;
    for (int32_t i = 0; i < 5 * 2; i += 2)
        std::cout << out_h[i] << " + " << out_h[i + 1] << "i" << std::endl;

    std::chrono::duration<double> elapsed_seconds = end - start;
    std::cout << "Elapsed time: " << elapsed_seconds.count() << "s" << std::endl;

    CUDA_CALL(cudaFree(in_d));
    CUDA_CALL(cudaFree(out_d));
    free(in_h);
    free(out_h);
    return EXIT_SUCCESS;
}