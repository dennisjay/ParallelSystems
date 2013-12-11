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
	return ty * BLOCKWIDTH + tx;
}

__device__ int getIndexSharedBorder(int tx, int ty) {
	if (tx < -1 || tx >= BLOCKWIDTH+1 || ty < -1 || ty >= BLOCKHEIGHT+1) {
		printf( "wrong Idx" ) ;
		return 0;
	}
	return (ty+1) * (BLOCKWIDTH+2) + (tx+1);
}

__device__ float getValueSharedBorder(float* a, int tx, int ty) {
	return a[getIndexSharedBorder(tx,ty)] ;
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
__global__ void filter_Kernel_Border(float* d_a, float* d_res) {	
	// 2D Thread ID
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int i = blockIdx.x * blockDim.x + threadIdx.x; // global thread Ids
	int j = blockIdx.y * blockDim.y + threadIdx.y; // global thread Ids

	// This part is only for version 2 and 3 of the kernel
	// declaration of the array in shared memory
	__shared__ float s_a[(BLOCKWIDTH+2) * (BLOCKHEIGHT+2)];
	//
	// filling the array in shared memory
	// central part
	s_a[getIndexSharedBorder(tx, ty)] = d_a[getIndexGlobal(i, j)]; 
	
	// Idea: Threads at the border load the neighboring part
	// left part
	if (tx <= 0) {
		s_a[getIndexSharedBorder(tx-1,ty)] = getValueGlobal(d_a, i-1, j); 
	}
	// right part
	if (tx >= BLOCKWIDTH-1) {
		s_a[getIndexSharedBorder(tx+1,ty)] = getValueGlobal(d_a, i+1, j); 
	}
	// downer part
	if (ty <= 0) {
		s_a[getIndexSharedBorder(tx,ty-1)] = getValueGlobal(d_a, i, j-1); 
	}	
	// upper part
	if (ty >= BLOCKHEIGHT-1) {
		s_a[getIndexSharedBorder(tx,ty+1)] = getValueGlobal(d_a, i, j+1); 
	}
	//edges
	if (tx <= 0 && ty <= 0) {
		s_a[getIndexSharedBorder(tx-1,ty-1)] = getValueGlobal(d_a, i-1, j-1); 
	}
	if (tx >= BLOCKWIDTH-1 && ty <= 0) {
		s_a[getIndexSharedBorder(tx+1,ty-1)] = getValueGlobal(d_a, i+1, j-1); 
	}
	if (tx <= 0 && ty >= BLOCKHEIGHT-1) {
		s_a[getIndexSharedBorder(tx-1,ty+1)] = getValueGlobal(d_a, i-1, j+1); 
	}
	if (tx >= BLOCKWIDTH-1 && ty >= BLOCKHEIGHT-1) {
		s_a[getIndexSharedBorder(tx+1,ty+1)] = getValueGlobal(d_a, i+1, j+1); 
	}



	// synchronize
	__syncthreads() ;

	float Gx = getValueSharedBorder(s_a, tx-1, ty-1)+2*getValueSharedBorder(s_a, tx-1, ty)+getValueSharedBorder(s_a, tx-1, ty+1)
					-getValueSharedBorder(s_a, tx+1, ty-1)-2*getValueSharedBorder(s_a, tx+1, ty)-getValueSharedBorder(s_a, tx+1, ty+1);

	float Gy = getValueSharedBorder(s_a, tx-1, ty-1)+2*getValueSharedBorder(s_a, tx, ty-1)+getValueSharedBorder(s_a, tx+1, ty-1)
					-getValueSharedBorder(s_a, tx-1, ty+1)-2*getValueSharedBorder(s_a, tx, ty+1)-getValueSharedBorder(s_a, tx+1, ty+1);
	
	d_res[getIndexGlobal(i, j)] = sqrt(Gx * Gx + Gy * Gy);
}

__global__ void filter_Kernel(float* d_a, float* d_res) {
	// 2D Thread ID
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int i = blockIdx.x * blockDim.x + threadIdx.x; // global thread Ids
	int j = blockIdx.y * blockDim.y + threadIdx.y; // global thread Ids

	// This part is only for version 2 and 3 of the kernel
	// declaration of the array in shared memory
	__shared__ float s_a[BLOCKWIDTH * BLOCKHEIGHT];
	//
	// filling the array in shared memory
	// central part
	s_a[getIndexShared(tx, ty)] = d_a[getIndexGlobal(i, j)]; 
	//
	// synchronize
	__syncthreads() ;

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
	if ((tx > 0) && (ty > 0))
		a_xm1_ym1 = s_a[getIndexShared(tx-1, ty-1)];
	else
		a_xm1_ym1 = getValueGlobal(d_a, i-1, j-1);

	if ((tx > 0))
		a_xm1_y = s_a[getIndexShared(tx-1, ty)];
	else
		a_xm1_y = getValueGlobal(d_a, i-1, j);

	if ((tx > 0) && (ty < BLOCKHEIGHT-1))
		a_xm1_yp1 = s_a[getIndexShared(tx-1, ty+1)];
	else
		a_xm1_yp1 = getValueGlobal(d_a, i-1, j+1);

	if (ty > 0)
		a_x_ym1 = s_a[getIndexShared(tx, ty-1)];
	else
		a_x_ym1 = getValueGlobal(d_a, i, j-1);

	if (ty < BLOCKHEIGHT-1)
		a_x_yp1 = s_a[getIndexShared(tx, ty+1)];
	else
		a_x_yp1 = getValueGlobal(d_a, i, j+1);

	if ((tx < BLOCKWIDTH-1) && (ty>0))
		a_xp1_ym1 = s_a[getIndexShared(tx+1, ty-1)];
	else
		a_xp1_ym1 = getValueGlobal(d_a, i+1, j-1);

	if ((tx < BLOCKWIDTH-1))
		a_xp1_y = s_a[getIndexShared(tx+1, ty)];
	else
		a_xp1_y = getValueGlobal(d_a, i+1, j);

	if ((tx < BLOCKWIDTH-1) && (ty < BLOCKHEIGHT-1))
		a_xp1_yp1 = s_a[getIndexShared(tx+1, ty+1)];
	else
		a_xp1_yp1 = getValueGlobal(d_a, i+1, j+1);


	float Gx = a_xm1_ym1+2*a_xm1_y+a_xm1_yp1-a_xp1_ym1-2*a_xp1_y-a_xp1_yp1;
	float Gy = a_xm1_ym1+2*a_x_ym1+a_xp1_ym1-a_xm1_yp1-2*a_x_yp1-a_xp1_yp1;
	
	d_res[getIndexGlobal(i, j)] = sqrt(Gx * Gx + Gy * Gy);
}

//////////////////////////////////////////////////////////////////////////////
// Device implementation (host function)
//////////////////////////////////////////////////////////////////////////////
void filterOnDevice(float* d_a, float* d_res) {
	dim3 dimGrid(GRIDWIDTH, GRIDHEIGHT);
	dim3 dimBlock(BLOCKWIDTH, BLOCKHEIGHT);
	CUDA_CHECK_KERNEL(filter_Kernel_Border<<< dimGrid, dimBlock >>>( d_a, d_res ));
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
