

__global__ void vecAdd(const float* A, const float* B, float* C, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) C[i] = A[i] + B[i];                  
}
int main() {
    // Launch: <<< gridSize, blockSize >>>  ->  4096 blocks x 256 threads
    vecAdd<<<4096, 256>>>(d_A, d_B, d_C, n);
}
