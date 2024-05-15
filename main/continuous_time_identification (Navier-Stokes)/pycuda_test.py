import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import numpy as np

def test_cuda():
    # Create a simple CUDA kernel that squares each element of an array
    mod = SourceModule("""
    __global__ void square_array(float *a, int N)
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx < N) a[idx] = a[idx] * a[idx];
    }
    """)

    # Generate input data: an array of 256 floats
    a = np.random.randn(256).astype(np.float32)
    original_a = a.copy()  # Keep a copy to verify against

    # Allocate memory on the device
    a_gpu = cuda.mem_alloc(a.nbytes)
    cuda.memcpy_htod(a_gpu, a)

    # Configure and launch the kernel
    func = mod.get_function("square_array")
    func(a_gpu, np.int32(a.size), block=(256,1,1), grid=(1,1))

    # Copy the result back to host
    cuda.memcpy_dtoh(a, a_gpu)

    # Print the original and the result to see the change
    print("Original array:", original_a[:10])  # Print first 10 elements
    print("Squared array:", a[:10])  # Print first 10 squared elements

if __name__ == "__main__":
    test_cuda()
