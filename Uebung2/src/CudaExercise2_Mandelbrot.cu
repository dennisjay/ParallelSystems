//////////////////////////////////////////////////////////////////////////////
// CUDA exercise 2: Mandelbrot set
//////////////////////////////////////////////////////////////////////////////

// includes
#include <stdio.h>
#include <stdint.h>

#include <helper_timer.h>
#include <helper_error.h>
#include <helper_image.h>

__global__ void mandelbrotKernel( uint32_t *d_a, uint32_t niter, float xmin, float xmax, float ymin, float ymax ) {
	int i = blockIdx.x * blockDim.x + threadIdx.x ;
	int j = blockIdx.y * blockDim.y + threadIdx.y ;

	int nElemX = blockDim.x * gridDim.x ;
	int nElemY = blockDim.y * gridDim.y ;

	float xc = xmin + (xmax - xmin) / (nElemX - 1) * i; //xc=real(c)

	__syncthreads();

	float yc = ymin + (ymax - ymin) / (nElemY - 1) * j; //yc=imag(c)
	
	__syncthreads();

	float x = 0.0; //x=real(z_k)
	float y = 0.0; //y=imag(z_k)
	for (int k = 0; k < niter; k = k + 1) { //iteration loop
		float tempx = x * x - y * y + xc; //z_{n+1}=(z_n)^2+c;
		y = 2 * x * y + yc;
		x = tempx;
		float r2 = x * x + y * y; //r2=|z_k|^2
		if ((r2 > 4) || k == niter - 1) { //divergence condition
			d_a[i + j * nElemX] = k;
			break;
		}
	}
	


}

//////////////////////////////////////////////////////////////////////////////
// Program main
//////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv) {
	//////// Declarations //////////////////////////////////////////

	// pointers to host memory
	uint32_t *h_a_cpu; //pointer to CPU results
	uint32_t *h_a_gpu; //pointer to GPU results on host

	// pointer to device memory
	uint32_t *d_a; //pointer to GPU results on device

	// define grid and block sizes
	int numBlocksX = 256;
	int numBlocksY = 256;
	int numThreadsPerBlockX = 8;
	int numThreadsPerBlockY = 8;

	// number of elements and memory size
	int nElemX = numBlocksX * numThreadsPerBlockX;
	int nElemY = numBlocksY * numThreadsPerBlockY;
	size_t memSize = nElemX * nElemY * sizeof(uint32_t);

	// timers
	StopWatchInterface* timer_cpu = NULL;
	StopWatchInterface* timer_gpu = NULL;
	StopWatchInterface* timer_mt = NULL;

	// parameters for the mandelbrot set
	int niter = 10; // maximum number of iterations
	float xmin = -2; // limits for c=x+i*y
	float xmax = 1;
	float ymin = -1.5;
	float ymax = 1.5;
	int64_t maxError = 1; // maximum difference between CPU and GPU solution (to account for rounding errors)

	//////// Allocate Memory ///////////////////////////////////////

	//Allocate host memory
	h_a_cpu = (uint32_t *) malloc(memSize);
	h_a_gpu = (uint32_t *) malloc(memSize);

	//Allocate device memory
	CUDA_CHECK(cudaMalloc( &d_a, memSize));

	// create timers
	sdkCreateTimer(&timer_cpu);
	sdkCreateTimer(&timer_gpu);
	sdkCreateTimer(&timer_mt);

	sdkStartTimer(&timer_cpu);
	//////// Calculation (Host) ////////////////////////////////////
	for (int i = 0; i < nElemX; i = i + 1) { //loop in the x-direction
		float xc = xmin + (xmax - xmin) / (nElemX - 1) * i; //xc=real(c)
		for (int j = 0; j < nElemY; j = j + 1) { //loop in the y-direction
			float yc = ymin + (ymax - ymin) / (nElemY - 1) * j; //yc=imag(c)
			float x = 0.0; //x=real(z_k)
			float y = 0.0; //y=imag(z_k)
			for (int k = 0; k < niter; k = k + 1) { //iteration loop
				float tempx = x * x - y * y + xc; //z_{n+1}=(z_n)^2+c;
				y = 2 * x * y + yc;
				x = tempx;
				float r2 = x * x + y * y; //r2=|z_k|^2
				if ((r2 > 4) || k == niter - 1) { //divergence condition
					h_a_cpu[i + j * nElemX] = k;
					break;
				}
			}
		}
	}
	sdkStopTimer(&timer_cpu);

	sdkStartTimer(&timer_gpu);
	//////// Calculation (Device) //////////////////////////////////
	dim3 dimGrid(numBlocksX, numBlocksY);
	dim3 dimBlock(numThreadsPerBlockX, numThreadsPerBlockY);
	mandelbrotKernel<<< dimGrid, dimBlock >>>( d_a, niter, xmin, xmax, ymin, ymax );

	// block until the device has completed
	cudaDeviceSynchronize();

	sdkStopTimer(&timer_gpu);

	sdkStartTimer(&timer_mt);
	//////// Copy Data From Device To Host /////////////////////////
	cudaMemcpy( h_a_gpu, d_a, memSize, cudaMemcpyDeviceToHost );

	sdkStopTimer(&timer_mt);

	//////// Store result images ///////////////////////////////////
	float* imageDataCpu = (float *) malloc (sizeof (float) * nElemX * nElemY);
	float* imageDataGpu = (float *) malloc (sizeof (float) * nElemX * nElemY);
	for (int i = 0; i < nElemX; i++) {
		for (int j = 0; j < nElemY; j++) {
			imageDataCpu[i + nElemX * j] = 1 - 1.0f * h_a_cpu[i + j * nElemX] / (niter - 1);
			imageDataGpu[i + nElemX * j] = 1 - 1.0f * h_a_gpu[i + j * nElemX] / (niter - 1);
		}
	}
	sdkSavePGM("output_cpu.pgm", imageDataCpu, nElemX, nElemY);
	sdkSavePGM("output_gpu.pgm", imageDataGpu, nElemX, nElemY);
	free (imageDataCpu);
	free (imageDataGpu);

	//////// Print elapsed time //////////////////////////////////////
	float t_cpu = sdkGetAverageTimerValue(&timer_cpu);
	float t_gpu = sdkGetAverageTimerValue(&timer_gpu); //TODO
	float t_mt =sdkGetAverageTimerValue(&timer_mt); //TODO
	printf("CUDA Mandelbrot niter=%d : GPU: % 4.1f ms MT: % 4.1f ms GPU+MT: % 4.1f ms CPU: % 4.1f ms speedup % 4.1f ( %4.1f )\n",
			niter, t_gpu, t_mt, t_gpu + t_mt, t_cpu, t_cpu / (t_gpu + t_mt),
			t_cpu / t_gpu);
	sdkResetTimer(&timer_cpu);
	sdkResetTimer(&timer_gpu);
	sdkResetTimer(&timer_mt);

	//////// Compare the results ///////////////////////////////////
	for (int i = 0; i < nElemX; i++) {
		for (int j = 0; j < nElemY; j++) {
			if (abs (((int64_t) h_a_gpu[i + j * nElemX]) - ((int64_t)h_a_cpu[i + j * nElemX])) > maxError) {
				printf("Error : i=%d j=%d h_a_gpu[i*nElemY+j]=%d h_a_cpu[i*nElemY+j]=%d \n",
						i, j, h_a_gpu[i + j * nElemX], h_a_cpu[i + j * nElemX]);
				return 1;
			}
		}
	}

	//////// Free Memory ///////////////////////////////////////////

	// free device memory
	cudaFree(d_a) ; 

	// free host memory
	free(h_a_cpu);
	free(h_a_gpu);

	// delete timers
	sdkDeleteTimer(&timer_cpu);
	sdkDeleteTimer(&timer_gpu);
	sdkDeleteTimer(&timer_mt);

	//////// End ///////////////////////////////////////////////////
	printf("Success\n");

	return 0;
}
