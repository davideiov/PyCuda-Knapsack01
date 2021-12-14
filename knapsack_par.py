import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy
import sys
import time


def knapsack():

    ###only for prints
    numpy.set_printoptions(threshold=sys.maxsize)

    W = numpy.int32(500)  #total weight
    val = numpy.array([60, 100, 120, 80, 90, 110, 70, 50, 130, 40]).astype(numpy.int32)  #array value
    wei = numpy.array([50, 70, 80, 80, 90, 50, 80, 10, 25, 80]).astype(numpy.int32)  #weight array
    result = numpy.empty_like(numpy.random.randn((len(val) + 1) * (W + 1)).astype(numpy.int32))

    val_d = cuda.mem_alloc(val.nbytes)
    wei_d = cuda.mem_alloc(wei.nbytes)
    result_d = cuda.mem_alloc(result.nbytes)

    cuda.memcpy_htod(val_d, val)
    cuda.memcpy_htod(wei_d, wei)
    cuda.memcpy_htod(result_d, result)

    mod = SourceModule("""
    __device__ int maxi(int a, int b) { 
        return (a > b)? a : b; 
    }
  
    __global__ void knapsack(int *wt, int *val, int *output, int i,int W) {
      int w = threadIdx.x + blockIdx.x*blockDim.x;;
  
      if (i == 0 || W == 0)
        output[(i*W)+w] = 0;
      else if (wt[i-1] <= w)
        output[(i*W)+w] = maxi(val[i-1] + output[((i-1)*W)+(w-wt[i-1])],  output[((i-1)*W)+w]);
      else
        output[(i*W)+w] = output[((i-1)*W)+w];
      __syncthreads();
      
    }
    """)

    func = mod.get_function("knapsack")
    numThreads = 128
    numBlocks = int((W+1) / numThreads) + (0 if (W+1) % numThreads == 0 else 1)
    #print(numBlocks)

    start = time.time()
    for i in range(len(val)+1):
        i_d = numpy.int32(i)
        func(wei_d, val_d, result_d, i_d, W, block=(numThreads, 1, 1), grid=(numBlocks, 1, 1))
    end = time.time()

    cuda.memcpy_dtoh(result, result_d)
    print('Execution time: ' + str(end - start) + 's')
    print('Max value of knapsack with capacity ' + str(W) + ' is ' + str(result[(len(val)+1) * W]))

    #for i in range(len(val)+1):
    #  for j in range(W+1):
    #    print(result[i*W + j], end=',')
    #  print('')


if __name__ == '__main__':
    knapsack()
