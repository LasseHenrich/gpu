#include <format>
#include <iostream>
#include <random>
#include <algorithm>
#include <vector>

void check(cudaError_t err, std::string msg) {
  if (err != cudaSuccess) {
    std::cerr << std::format("{} (error code %s)\n", msg, cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}

__global__ void vectorAdd(const float *A, const float *B,
                          float *C, int numElements) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;

  if (i < numElements) {
    C[i] = A[i] + B[i];
  }
}

int main() {
  // Error code to check return values for CUDA calls
  cudaError_t err = cudaSuccess;

  int numElements = 50000; //65536;
  std::cout << std::format("[Vector addition of {} elements]\n", numElements);

  auto h_A = std::vector<float>(numElements);
  auto h_B = std::vector<float>(numElements);
  auto h_C = std::vector<float>(numElements);

  std::random_device rd;
  std::mt19937 engine {rd()};
  std::uniform_real_distribution<> dist {1.0, 2.0};
  auto rd_gen = [&](){ return dist(engine); };

  std::generate(h_A.begin(), h_A.end(), rd_gen);
  std::generate(h_B.begin(), h_B.end(), rd_gen);

  // Allocate device memory
  size_t size = numElements * sizeof(float);

  float *d_A = NULL;
  err = cudaMalloc((void **)&d_A, size);
  check(err, "Failed to allocate device vector A");

  float *d_B = NULL;
  err = cudaMalloc((void **)&d_B, size);
  check(err, "Failed to allocate device vector B");

  float *d_C = NULL;
  err = cudaMalloc((void **)&d_C, size);
  check(err, "Failed to allocate device vector C");

  // Copy the host input vectors A and B in host memory to the device input
  // vectors in device memory
  std::cout << "Copy input data from the host memory to the CUDA device\n";
  err = cudaMemcpy(d_A, h_A.data(), size, cudaMemcpyHostToDevice);
  check(err, "Failed to copy vector A from host to device");

  err = cudaMemcpy(d_B, h_B.data(), size, cudaMemcpyHostToDevice);
  check(err, "Failed to copy vector B from host to device");

  // Launch the Vector Add CUDA Kernel
  int threadsPerBlock = 256;
  int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
  printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid,
         threadsPerBlock);
  vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);
  check(cudaGetLastError(), "Failed to launch vectorAdd kernel");

  // Copy the device result vector in device memory to the host result vector
  // in host memory.
  std::cout << "Copy output data from the CUDA device to the host memory\n";
  err = cudaMemcpy(h_C.data(), d_C, size, cudaMemcpyDeviceToHost);
  check(err, "Failed to copy vector C from device to host");

  // Verify that the result vector is correct
  for (int i = 0; i < numElements; ++i) {
    if (std::abs(h_A[i] + h_B[i] - h_C[i]) > 1e-5) {
      fprintf(stderr, "Result verification failed at element %d!\n", i);
      exit(EXIT_FAILURE);
    }
  }

  std::cout << "Test PASSED\n";

  // Free device global memory
  err = cudaFree(d_A);
  check(err, "Failed to free device vector A");
  err = cudaFree(d_B);
  check(err, "Failed to free device vector B");
  err = cudaFree(d_C);
  check(err, "Failed to free device vector C");


  std::cout << "Done\n";
  return 0;
}
