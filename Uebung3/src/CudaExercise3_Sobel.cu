//////////////////////////////////////////////////////////////////////////////
// CUDA exercise 3: Sobel filter
//////////////////////////////////////////////////////////////////////////////

// includes
#include <stdio.h>
#include <stdint.h>

#include <helper_timer.h>
#include <helper_error.h>
#include <helper_image.h>

#define BLOCKWIDTH 16
#define BLOCKHEIGHT 16
#define GRIDWIDTH  40
#define GRIDHEIGHT  30
#define WIDTH (BLOCKWIDTH*GRIDWIDTH)
#define HEIGHT (BLOCKHEIGHT*GRIDHEIGHT)

//////////////////////////////////////////////////////////////////////////////
// Helper functions (device / host)
//////////////////////////////////////////////////////////////////////////////

__host__ __device__ int getIndexGlobal(int i, int j) {
	return j * WIDTH + i;
}

// Read value from global array a, return 0 if outside image
__host__ __device__ float getValueGlobal(float* a, int i, int j) {
	if (i < 0 || i >= WIDTH || j < 0 || j >= HEIGHT)
		return 0;
	else
		return a[getIndexGlobal(i, j)];
}

__device__ int getIndexShared(int tx, int ty) {
	//TODO
}

//////////////////////////////////////////////////////////////////////////////
// Host implementation
//////////////////////////////////////////////////////////////////////////////
void filterOnHost(float* h_a, float* h_res_cpu) {
	for (int i=0; i<WIDTH; i++) {
		for (int j=0; j<HEIGHT; j++) {
			float Gx = getValueGlobal(h_a, i-1, j-1)+2*getValueGlobal(h_a, i-1, j)+getValueGlobal(h_a, i-1, j+1)
					-getValueGlobal(h_a, i+1, j-1)-2*getValueGlobal(h_a, i+1, j)-getValueGlobal(h_a, i+1, j+1);
			float Gy = getValueGlobal(h_a, i-1, j-1)+2*getValueGlobal(h_a, i, j-1)+getValueGlobal(h_a, i+1, j-1)
					-getValueGlobal(h_a, i-1, j+1)-2*getValueGlobal(h_a, i, j+1)-getValueGlobal(h_a, i+1, j+1);
			h_res_cpu[getIndexGlobal(i, j)] = sqrt(Gx * Gx + Gy * Gy);
		}
	}
}

//////////////////////////////////////////////////////////////////////////////
// Device implementation (kernel)
//////////////////////////////////////////////////////////////////////////////
__global__ void filter_Kernel(float* d_a, float* d_res) {
	// 2D Thread ID
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	//TODO int i = ...; // global thread Ids
	//TODO int j = ...; // global thread Ids

	// This part is only for version 2 and 3 of the kernel
	/*
	// declaration of the array in shared memory
	__shared__ float s_a[BLOCKWIDTH * BLOCKHEIGHT];
	//
	// filling the array in shared memory
	// central part
	s_a[getIndexShared(...)] = d_a[...]; //TODO
	//
	// synchronize
	//TODO
	*/

	// Values of neighbouring pixels
	float a_xm1_ym1;
	float a_xm1_y;
	float a_xm1_yp1;
	float a_x_ym1;
	float a_x_yp1;
	float a_xp1_ym1;
	float a_xp1_y;
	float a_xp1_yp1;

	// This part is only for version 2 of the kernel
	/*
	if ((tx > 0) && (ty > 0))
		a_xm1_ym1 = s_a[...];
	else
		a_xm1_ym1 = getValueGlobal(...);
	//
	//TODO
	*/

	// calculate Gx and Gy
	//TODO float Gx = ...;
	//TODO float Gy = ...;
	//TODO d_res[...] = sqrt(Gx * Gx + Gy * Gy); // write the result to global memory
}

//////////////////////////////////////////////////////////////////////////////
// Device implementation (host function)
//////////////////////////////////////////////////////////////////////////////
void filterOnDevice(float* d_a, float* d_res) {
	//TODO: Kernel call
}

//////////////////////////////////////////////////////////////////////////////
// Main function
//////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv) {
	CUDA_CHECK(cudaSetDevice(0));
	int device;
	CUDA_CHECK(cudaGetDevice(&device));
	struct cudaDeviceProp properties;
	CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
	printf("Running on device %d (%s) (%d.%d)\n", device, properties.name, properties.major, properties.minor);

	//////// Declarations ////////////////////////////////////////////

	// pointers to host memory
	float* h_a; // pointer to input data on CPU
	float* h_res_cpu; // pointer to CPU result data
	float* h_res_gpu; // pointer to GPU result data on host

	// pointer to device memory
	float* d_a; // pointer to GPU input data on device
	float* d_res; // pointer to GPU result data on device

	// timers
	StopWatchInterface* timer_cpu = NULL;
	StopWatchInterface* timer_gpu = NULL;
	StopWatchInterface* timer_mt = NULL;

	//////// Allocate Memory ///////////////////////////////////////

	// Allocate host memory
	h_a = (float*) malloc (WIDTH * HEIGHT * sizeof (float));
	h_res_cpu = (float*) malloc (WIDTH * HEIGHT * sizeof (float));
	h_res_gpu = (float*) malloc (WIDTH * HEIGHT * sizeof (float));

	// Allocate Device Memory
	CUDA_CHECK(cudaMalloc( &d_a, (WIDTH * HEIGHT * sizeof (float))));
	CUDA_CHECK(cudaMalloc( &d_res, (WIDTH * HEIGHT * sizeof (float))));

	// create timers
	sdkCreateTimer(&timer_cpu);
	sdkCreateTimer(&timer_gpu);
	sdkCreateTimer(&timer_mt);

	//////// Initialize host memory ////////////////////////////////
	/*
	for (int i=0; i<WIDTH*HEIGHT; i++) {
		h_a[i]=(rand() % 100) / 5.0f - 10.0f;
		h_res_cpu[i]= 0.0f;
	}
	*/
	float* inputData = NULL;
	unsigned int inputWidth, inputHeight;
	if (!sdkLoadPGM("Valve.pgm", &inputData, &inputWidth, &inputHeight)) {
		printf("Error: Could not load input file\n");
		return 1;
	}
	for (int j = 0; j < HEIGHT; j++) {
		for (int i = 0; i < WIDTH; i++) {
			h_a[i + WIDTH * j] = inputData[(i % inputWidth) + inputWidth * (j % inputHeight)];
		}
	}
	free(inputData);

	//////// Copy Input Data From Host To Device ///////////////////
	sdkStartTimer(&timer_mt);	
	cudaMemcpy( d_a, h_a, (WIDTH * HEIGHT * sizeof (float)), cudaMemcpyHostToDevice );
	sdkStopTimer(&timer_mt);

	//////// Calculation (Host) ////////////////////////////////////
	sdkStartTimer(&timer_cpu);
	filterOnHost(h_a, h_res_cpu);
	sdkStopTimer(&timer_cpu);
	
	//////// Calculation (Device) //////////////////////////////////
	sdkStartTimer(&timer_gpu);
	filterOnDevice(d_a, d_res);
	sdkStopTimer(&timer_gpu);

	//////// Copy Output Data From Device To Host //////////////////
	sdkStartTimer(&timer_mt);
	cudaMemcpy( h_res_gpu, d_res, (WIDTH * HEIGHT * sizeof (float)), cudaMemcpyDeviceToHost );
	sdkStopTimer(&timer_mt);

	//////// Store output images ///////////////////////////////////
	sdkSavePGM("output_cpu.pgm", h_res_cpu, WIDTH, HEIGHT);
	sdkSavePGM("output_gpu.pgm", h_res_gpu, WIDTH, HEIGHT);

	//////// Print elapsed time / speedup //////////////////////////
	float t_cpu = sdkGetAverageTimerValue(&timer_cpu);
	float t_gpu = sdkGetAverageTimerValue(&timer_gpu); //TODO
	float t_mt =sdkGetAverageTimerValue(&timer_mt); //TODO
	printf("CUDA Sobel: GPU: % 4.1f ms MT: % 4.1f ms GPU+MT: % 4.1f ms CPU: % 4.1f ms speedup % 4.1f ( %4.1f )\n",
			t_gpu, t_mt, t_gpu + t_mt, t_cpu, t_cpu / (t_gpu + t_mt),
			t_cpu / t_gpu);
	sdkResetTimer(&timer_cpu);
	sdkResetTimer(&timer_gpu);
	sdkResetTimer(&timer_mt);

	//////// Compare the results ///////////////////////////////////
	int maxErrors = 50;
	int errorCount = 0;
	for (int k = 0; k < WIDTH*HEIGHT; k++) {
		float diff = abs (h_res_cpu[k] - h_res_gpu[k]);
		if (!(diff < 0.0001)) {
			printf ("k: %d i: %d  j: %d CPU: %f GPU: %f Diff: %f\n",k, k%WIDTH, k/WIDTH, h_res_cpu[k], h_res_gpu[k], h_res_cpu[k]-h_res_gpu[k]);
			errorCount++;
			if (errorCount > maxErrors) {
				printf("More than %d errors\n", maxErrors);
				return 1;
			}
		}
	}
	if (errorCount > 0) {
		printf("%d errors\n", errorCount);
		return 1;
	}

	//////// Free Memory ///////////////////////////////////////////

	// free host memory
	free(h_a);
	free(h_res_cpu);
	free(h_res_gpu);

	// free device memory
	cudaFree(d_a) ; 
	cudaFree(d_res) ; 

	// delete timers
	sdkDeleteTimer(&timer_cpu);
	sdkDeleteTimer(&timer_gpu);
	sdkDeleteTimer(&timer_mt);

	//////// End ///////////////////////////////////////////////////
	printf("Success\n");

	return 0;
}
