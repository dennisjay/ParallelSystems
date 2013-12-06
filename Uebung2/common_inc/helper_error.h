/**
 * This macro checks return value of the CUDA runtime call and exits
 * the application if the call failed.
 */
#define CUDA_CHECK(value) do {													\
	cudaError_t _m_cudaStat = value;											\
	if (_m_cudaStat != cudaSuccess) {											\
		fprintf(stderr, "Error %s at line %d in file %s in `%s'\n",				\
				cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__, #value);  	\
		exit(1);																\
	} } while (0)

/**
 * This macro checks return value of a kernel call and exits
 * the application if the call failed.
 */
#define CUDA_CHECK_KERNEL(...) do {												\
	cudaGetLastError();	/* Clear last error */										\
	__VA_ARGS__;		/* Call kernel      */										\
	cudaError_t _m_cudaStat = cudaGetLastError();									\
	if (_m_cudaStat != cudaSuccess) {												\
		fprintf(stderr, "Error %s at line %d in file %s in `%s'\n",					\
				cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__, #__VA_ARGS__);	\
		exit(1);																	\
	} } while (0)
