//////////////////////////////////////////////////////////////////////////////
// CUDA exercise 1: Basics
//////////////////////////////////////////////////////////////////////////////

// includes
#include <stdio.h>

#include <helper_timer.h>
#include <helper_error.h>

//////////////////////////////////////////////////////////////////////////////
// Kernel (for part 2 of the exercise)
//////////////////////////////////////////////////////////////////////////////

__global__ void multiplyKernel(int *d_a) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	d_a[idx] = d_a[idx] * 2;
}

//////////////////////////////////////////////////////////////////////////////
// Main function
//////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv) {
	//////// Declarations ////////////////////////////////////////////

	// pointers to host memory
	int *h_a_cpu; // pointer to CPU data
	int *h_a_gpu; // pointer to GPU data on host

	// pointer to device memory
	int *d_a; // pointer to GPU data on device

	// define grid and block sizes
	int numBlocks = 16;
	int numThreadsPerBlock = 16;

	// number of elements and memory size
	int nElem = numBlocks * numThreadsPerBlock;
	size_t memSize = nElem * sizeof(int);

	//////// Allocate Memory ///////////////////////////////////////

	// Allocate host memory
	h_a_cpu = (int *) malloc(memSize);
	h_a_gpu = (int *) malloc(memSize);

	// Allocate device memory
	CUDA_CHECK(cudaMalloc( &d_a, memSize));

	//////// Initialize host memory ////////////////////////////////
	for (int i = 0; i < nElem; i = i + 1) {
		h_a_cpu[i] = 5 * i + 17;
	}

	//////// Copy Data From Host To Device /////////////////////////
	cudaMemcpy( d_a, h_a_cpu, memSize, cudaMemcpyHostToDevice );

	//////// Calculation (Host) ////////////////////////////////////
	for (int i = 0; i < nElem; i++) {
		h_a_cpu[i] = 2 * h_a_cpu[i];
	}

	//////// Calculation (Device) //////////////////////////////////
	dim3 dimGrid(numBlocks) ;
	dim3 dimBlock(numThreadsPerBlock);

	// block until the device has completed
	CUDA_CHECK_KERNEL( multiplyKernel<<<dimGrid, dimBlock>>>(d_a)); 

	// End of region for part 2 of the exercise ----------------
	// 

	//////// Copy Data From Device To Host /////////////////////////
	cudaMemcpy( h_a_gpu, d_a, memSize, cudaMemcpyDeviceToHost );

	//////// Compare the results ///////////////////////////////////
	for (int i = 0; i < nElem; i++) {
		if (h_a_gpu[i] != h_a_cpu[i]) {
			printf("Error: i=%d h_a_gpu[i]=%d h_a_cpu[i]=%d\n", i,
					h_a_gpu[i], h_a_cpu[i]);
			return 1;
		}
	}

	//////// Free Memory ///////////////////////////////////////////

	// free device memory
	cudaFree(d_a) ; 

	// free host memory
	free(h_a_cpu);
	free(h_a_gpu);

	//////// End ///////////////////////////////////////////////////
	printf("Success\n");

	return 0;
}
