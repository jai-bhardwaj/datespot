#include <stdio.h>
#include <cuda_runtime.h>
#include "math.cuh"

/**
 * @brief Gets the maximum number of threads per multi-processor for a given CUDA device.
 * 
 * This function takes a pointer to a `deviceContextStruct` and returns the maximum number of threads per multi-processor for the CUDA device specified in the `deviceContextStruct`.
 * 
 * @param deviceContextStructP Pointer to the `deviceContextStruct` holding information about the CUDA device.
 * @return The maximum number of threads per multi-processor for the specified CUDA device.
 */
int getMaxThreadsPerMultiProcessor(  deviceContextStruct * deviceContextStructP ){
    /**
     * @brief Structure containing information about the CUDA device.
     */
    cudaDeviceProp deviceProp;

    /**
     * @brief Retrieve properties of the CUDA device.
     *
     * @param deviceProp Pointer to the structure to store the device properties.
     * @param deviceId ID of the device to retrieve properties for.
     */
    cudaGetDeviceProperties( &deviceProp, deviceContextStructP->deviceId );

    /**
     * @brief Return the maximum number of threads per multi-processor of the CUDA device.
     *
     * @return The maximum number of threads per multi-processor.
     */
    return deviceProp.maxThreadsPerMultiProcessor;

}

/**
 * @brief CUDA kernel to add a scalar value to an array of `double` values.
 * 
 * This CUDA kernel takes a pointer to an array of `double` values, a scalar value `alpha`, and the number of elements in the array, and adds `alpha` to each element of the array.
 * The computation is performed by each thread in parallel, where each thread is responsible for adding `alpha` to one element of the array.
 * 
 * @param deviceA Pointer to the array of `double` values to add `alpha` to.
 * @param alpha The scalar value to add to each element of the array.
 * @param elementNum The number of elements in the array.
 */
__global__ void arrayAddKernel( double *deviceA, double alpha, int elementNum ){
    /**
     * @brief Calculate the global index for the current thread.
     *
     * @param blockDim.x Number of threads per block in the x-dimension.
     * @param blockIdx.x Index of the current block in the x-dimension.
     * @param threadIdx.x Index of the current thread in the x-dimension.
     * @return The global index for the current thread.
     */
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if( i < elementNum) {
        /**
         * @brief Add the scalar value alpha to the element at index i in deviceA and store the result back in deviceA.
         *
         * @param deviceA Pointer to the device array A.
         * @param i Index of the element in the array.
         * @param alpha The scalar value to add.
         */
        deviceA[i] = deviceA[i] + alpha;
    }
}

/**
 * @brief Adds a scalar value to an array of `double` values using CUDA.
 * 
 * This function takes a pointer to a `deviceContextStruct`, a pointer to an array of `double` values, the number of elements in the array, and a scalar value `alpha`.
 * It allocates memory on the CUDA device, copies the input array to the device, runs the `arrayAddKernel` CUDA kernel to add `alpha` to each element of the array, and then copies the result back to the host.
 * 
 * @param deviceContextStructP Pointer to the `deviceContextStruct` holding information about the CUDA device.
 * @param hostAP Pointer to the array of `double` values to add `alpha` to.
 * @param elementNum The number of elements in the array.
 * @param alpha The scalar value to add to each element of the array.
 */
void arrayAdd( deviceContextStruct * deviceContextStructP, double * hostAP, int elementNum, double alpha ){
    /**
     * @brief Calculate the size of memory needed for array A on the device.
     *
     * @param elementNum The number of elements in the array.
     * @return The size of memory required in bytes.
     */
    int sizeA = elementNum * sizeof(double);

    double * deviceA;
    /**
     * @brief Allocate memory on the device for array deviceA.
     *
     * @param deviceA Pointer to the device array A.
     * @param sizeA Size of the memory to be allocated in bytes.
     */
    cudaMalloc( (void **) &deviceA, sizeA );

    /**
     * @brief Copy the host array hostAP to device array A.
     *
     * @param deviceA Pointer to the device array A.
     * @param hostAP Pointer to the host array to be copied.
     * @param sizeA Size of the memory to be copied in bytes.
     */
    cudaMemcpy( deviceA, hostAP, sizeA, cudaMemcpyHostToDevice );

    int threadsPerBlock = getMaxThreadsPerMultiProcessor( deviceContextStructP );
    int blocksPerGrid = ( elementNum + threadsPerBlock - 1 ) / threadsPerBlock;

    /**
     * @brief Launch the arrayAddKernel on the device.
     *
     * @param blocksPerGrid Number of blocks in the grid.
     * @param threadsPerBlock Number of threads per block.
     * @param deviceA Pointer to the device array A.
     * @param alpha The scalar value to add to the array elements.
     * @param elementNum The number of elements in the array.
     */
    arrayAddKernel<<< blocksPerGrid, threadsPerBlock >>>( deviceA, alpha, elementNum );

    /**
     * @brief Copy the result from device array A to the host array hostAP.
     *
     * @param hostAP Pointer to the host array where the result will be copied.
     * @param deviceA Pointer to the device array A.
     * @param sizeA Size of the memory to be copied in bytes.
     */
    cudaMemcpy( hostAP, deviceA, sizeA, cudaMemcpyDeviceToHost );

    /**
     * @brief Free the allocated memory on the device.
     *
     * @param deviceA Pointer to the device array A.
     */
    cudaFree(deviceA);

    
}

/**
 * @brief CUDA kernel to subtract an array of `double` values from a scalar value.
 * 
 * This CUDA kernel takes a scalar value `alpha`, a pointer to an array of `double` values, and the number of elements in the array, and subtracts each element of the array from `alpha`.
 * The computation is performed by each thread in parallel, where each thread is responsible for subtracting one element of the array from `alpha`.
 * 
 * @param alpha The scalar value to subtract the elements of the array from.
 * @param deviceA Pointer to the array of `double` values to subtract from `alpha`.
 * @param elementNum The number of elements in the array.
 */
__global__ void subtractArrayKernel(  double alpha, double *deviceA, int elementNum ){
    /**
     * @brief Calculate the global index for the current thread.
     *
     * @param blockDim.x Number of threads per block in the x-dimension.
     * @param blockIdx.x Index of the current block in the x-dimension.
     * @param threadIdx.x Index of the current thread in the x-dimension.
     * @return The global index for the current thread.
     */
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if( i < elementNum) {
        /**
         * @brief Subtract the element at index i in deviceA from the scalar value alpha and store the result back in deviceA.
         *
         * @param deviceA Pointer to the device array A.
         * @param i Index of the element in the array.
         * @param alpha The scalar value to subtract from.
         */
        deviceA[i] = alpha - deviceA[i];
    }
}

/**
 * @brief Subtracts an array of `double` values from a scalar value using CUDA.
 * 
 * This function takes a pointer to a `deviceContextStruct`, a scalar value `alpha`, a pointer to an array of `double` values, and the number of elements in the array.
 * It allocates memory on the CUDA device, copies the input array to the device, runs the `subtractArrayKernel` CUDA kernel to subtract each element of the array from `alpha`, and then copies the result back to the host.
 * 
 * @param deviceContextStructP Pointer to the `deviceContextStruct` holding information about the CUDA device.
 * @param alpha The scalar value to subtract the elements of the array from.
 * @param hostAP Pointer to the array of `double` values to subtract from `alpha`.
 * @param elementNum The number of elements in the array.
 */
void subtractArray( deviceContextStruct * deviceContextStructP,  double alpha, double * hostAP, int elementNum ){
    /**
     * @brief Calculate the size of memory needed for array A on the device.
     *
     * @param elementNum The number of elements in the array.
     * @return The size of memory required in bytes.
     */
    int sizeA = elementNum * sizeof(double);

    double * deviceA;
    /**
     * @brief Allocate memory on the device for array deviceA.
     *
     * @param deviceA Pointer to the device array A.
     * @param sizeA Size of the memory to be allocated in bytes.
     */
    cudaMalloc( (void **) &deviceA, sizeA );

    /**
     * @brief Copy the host array hostAP to device array A.
     *
     * @param deviceA Pointer to the device array A.
     * @param hostAP Pointer to the host array to be copied.
     * @param sizeA Size of the memory to be copied in bytes.
     */
    cudaMemcpy( deviceA, hostAP, sizeA, cudaMemcpyHostToDevice );

    int threadsPerBlock = getMaxThreadsPerMultiProcessor( deviceContextStructP );
    int blocksPerGrid = ( elementNum + threadsPerBlock - 1 ) / threadsPerBlock;

    /**
     * @brief Launch the subtractArrayKernel on the device.
     *
     * @param blocksPerGrid Number of blocks in the grid.
     * @param threadsPerBlock Number of threads per block.
     * @param alpha The scalar value to subtract from the array elements.
     * @param deviceA Pointer to the device array A.
     * @param elementNum The number of elements in the array.
     */
    subtractArrayKernel<<< blocksPerGrid, threadsPerBlock >>>( alpha, deviceA, elementNum );

    /**
     * @brief Copy the result from device array A to the host array hostAP.
     *
     * @param hostAP Pointer to the host array where the result will be copied.
     * @param deviceA Pointer to the device array A.
     * @param sizeA Size of the memory to be copied in bytes.
     */
    cudaMemcpy( hostAP, deviceA, sizeA, cudaMemcpyDeviceToHost );

    /**
     * @brief Free the allocated memory on the device.
     *
     * @param deviceA Pointer to the device array A.
     */
    cudaFree(deviceA);
    
}

/**
 * @brief CUDA kernel to multiply an array of `double` values by a scalar value.
 * 
 * This CUDA kernel takes a pointer to an array of `double` values, a scalar value `alpha`, and the number of elements in the array, and multiplies each element of the array by `alpha`.
 * The computation is performed by each thread in parallel, where each thread is responsible for multiplying one element of the array by `alpha`.
 * 
 * @param deviceA Pointer to the array of `double` values to multiply by `alpha`.
 * @param alpha The scalar value to multiply each element of the array by.
 * @param elementNum The number of elements in the array.
 */
__global__ void arrayMultiplyKernel( double *deviceA, double alpha, int elementNum ){
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if( i < elementNum) {
        deviceA[i] = deviceA[i] * alpha;
    }
}

/**
 * @brief Multiplies an array of `double` values by a scalar value using CUDA.
 * 
 * This function takes a pointer to a `deviceContextStruct`, a pointer to an array of `double` values, the number of elements in the array, and a scalar value `alpha`.
 * It allocates memory on the CUDA device, copies the input array to the device, runs the `arrayMultiplyKernel` CUDA kernel to multiply each element of the array by `alpha`, and then copies the result back to the host.
 * 
 * @param deviceContextStructP Pointer to the `deviceContextStruct` holding information about the CUDA device.
 * @param hostAP Pointer to the array of `double` values to multiply by `alpha`.
 * @param elementNum The number of elements in the array.
 * @param alpha The scalar value to multiply each element of the array by.
 */
void arrayMultiply( deviceContextStruct * deviceContextStructP, double * hostAP, int elementNum, double alpha ){
    /**
     * @brief Calculate the size of memory needed for array A on the device.
     *
     * @param elementNum The number of elements in the array.
     * @return The size of memory required in bytes.
     */
    int sizeA = elementNum * sizeof(double);

    double * deviceA;
    /**
     * @brief Allocate memory on the device for array deviceA.
     *
     * @param deviceA Pointer to the device array A.
     * @param sizeA Size of the memory to be allocated in bytes.
     */
    cudaMalloc( (void **) &deviceA, sizeA );

    /**
     * @brief Copy the host array hostAP to device array A.
     *
     * @param deviceA Pointer to the device array A.
     * @param hostAP Pointer to the host array to be copied.
     * @param sizeA Size of the memory to be copied in bytes.
     */
    cudaMemcpy( deviceA, hostAP, sizeA, cudaMemcpyHostToDevice );

    int threadsPerBlock = getMaxThreadsPerMultiProcessor( deviceContextStructP );
    int blocksPerGrid = ( elementNum + threadsPerBlock - 1 ) / threadsPerBlock;

    /**
     * @brief Launch the arrayMultiplyKernel on the device.
     *
     * @param blocksPerGrid Number of blocks in the grid.
     * @param threadsPerBlock Number of threads per block.
     * @param deviceA Pointer to the device array A.
     * @param alpha The scalar value to multiply the array elements by.
     * @param elementNum The number of elements in the array.
     */
    arrayMultiplyKernel<<< blocksPerGrid, threadsPerBlock >>>( deviceA, alpha, elementNum );

    /**
     * @brief Copy the result from device array A to the host array hostAP.
     *
     * @param hostAP Pointer to the host array where the result will be copied.
     * @param deviceA Pointer to the device array A.
     * @param sizeA Size of the memory to be copied in bytes.
     */
    cudaMemcpy( hostAP, deviceA, sizeA, cudaMemcpyDeviceToHost );

    /**
     * @brief Free the allocated memory on the device.
     *
     * @param deviceA Pointer to the device array A.
     */
    cudaFree(deviceA);
    
}


/**
 * @brief CUDA kernel to divide a scalar value by an array of `double` values.
 * 
 * This CUDA kernel takes a scalar value `alpha`, a pointer to an array of `double` values, and the number of elements in the array, and divides `alpha` by each element of the array.
 * The computation is performed by each thread in parallel, where each thread is responsible for dividing `alpha` by one element of the array.
 * 
 * @param alpha The scalar value to divide by each element of the array.
 * @param deviceA Pointer to the array of `double` values to divide `alpha` by.
 * @param elementNum The number of elements in the array.
 */
__global__ void divideArrayKernel(  double alpha, double *deviceA, int elementNum ){
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if( i < elementNum) {
        deviceA[i] = alpha / deviceA[i];
    }
}

/**
 * @brief Divides a scalar value by an array of `double` values using CUDA.
 * 
 * This function takes a pointer to a `deviceContextStruct`, a scalar value `alpha`, a pointer to an array of `double` values, and the number of elements in the array.
 * It allocates memory on the CUDA device, copies the input array to the device, runs the `divideArrayKernel` CUDA kernel to divide `alpha` by each element of the array, and then copies the result back to the host.
 * 
 * @param deviceContextStructP Pointer to the `deviceContextStruct` holding information about the CUDA device.
 * @param alpha The scalar value to divide by each element of the array.
 * @param hostAP Pointer to the array of `double` values to divide `alpha` by.
 * @param elementNum The number of elements in the array.
 */
void divideArray( deviceContextStruct * deviceContextStructP,  double alpha, double * hostAP, int elementNum ){
    /**
     * @brief Calculate the size of memory needed for array A on the device.
     *
     * @param elementNum The number of elements in the array.
     * @return The size of memory required in bytes.
     */
    int sizeA = elementNum * sizeof(double);

    double * deviceA;
    /**
     * @brief Allocate memory on the device for array deviceA.
     *
     * @param deviceA Pointer to the device array A.
     * @param sizeA Size of the memory to be allocated in bytes.
     */
    cudaMalloc( (void **) &deviceA, sizeA );

    /**
     * @brief Copy the host array hostAP to device array A.
     *
     * @param deviceA Pointer to the device array A.
     * @param hostAP Pointer to the host array to be copied.
     * @param sizeA Size of the memory to be copied in bytes.
     */
    cudaMemcpy( deviceA, hostAP, sizeA, cudaMemcpyHostToDevice );

    int threadsPerBlock = getMaxThreadsPerMultiProcessor( deviceContextStructP );
    int blocksPerGrid = ( elementNum + threadsPerBlock - 1 ) / threadsPerBlock;

    /**
     * @brief Launch the divideArrayKernel on the device.
     *
     * @param blocksPerGrid Number of blocks in the grid.
     * @param threadsPerBlock Number of threads per block.
     * @param alpha The value to divide the array elements by.
     * @param deviceA Pointer to the device array A.
     * @param elementNum The number of elements in the array.
     */
    divideArrayKernel<<< blocksPerGrid, threadsPerBlock >>>( alpha, deviceA, elementNum );

    /**
     * @brief Copy the result from device array A to the host array hostAP.
     *
     * @param hostAP Pointer to the host array where the result will be copied.
     * @param deviceA Pointer to the device array A.
     * @param sizeA Size of the memory to be copied in bytes.
     */
    cudaMemcpy( hostAP, deviceA, sizeA, cudaMemcpyDeviceToHost );

    /**
     * @brief Free the allocated memory on the device.
     *
     * @param deviceA Pointer to the device array A.
     */
    cudaFree(deviceA);
    
}


/**
 * @brief CUDA kernel to raise an array of `double` values to a power.
 * 
 * This CUDA kernel takes a scalar value `alpha`, a pointer to an array of `double` values, and the number of elements in the array, and raises each element of the array to the power `alpha`.
 * The computation is performed by each thread in parallel, where each thread is responsible for raising one element of the array to the power `alpha`.
 * 
 * @param deviceA Pointer to the array of `double` values to raise to the power `alpha`.
 * @param alpha The scalar value to use as the power.
 * @param elementNum The number of elements in the array.
 */
__global__ void arrayPowerKernel( double *deviceA, double alpha, int elementNum ){
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if( i < elementNum) {
        deviceA[i] =  pow( deviceA[i], alpha );
    }
}

/**
 * @brief Raises an array of `double` values to a power using CUDA.
 * 
 * This function takes a pointer to a `deviceContextStruct`, a pointer to an array of `double` values, the number of elements in the array, and a scalar value `alpha`.
 * It allocates memory on the CUDA device, copies the input array to the device, runs the `arrayPowerKernel` CUDA kernel to raise each element of the array to the power `alpha`, and then copies the result back to the host.
 * 
 * @param deviceContextStructP Pointer to the `deviceContextStruct` holding information about the CUDA device.
 * @param hostAP Pointer to the array of `double` values to raise to the power `alpha`.
 * @param elementNum The number of elements in the array.
 * @param alpha The scalar value to use as the power.
 */
void arrayPower( deviceContextStruct * deviceContextStructP, double * hostAP, int elementNum, double alpha ){
    /**
     * @brief Calculate the size of memory needed for array A on the device.
     *
     * @param elementNum The number of elements in the array.
     * @return The size of memory required in bytes.
     */
    int sizeA = elementNum * sizeof(double);

    double * deviceA;
    /**
     * @brief Allocate memory on the device for array deviceA.
     *
     * @param deviceA Pointer to the device array A.
     * @param sizeA Size of the memory to be allocated in bytes.
     */
    cudaMalloc( (void **) &deviceA, sizeA );

    /**
     * @brief Copy the host array hostAP to device array A.
     *
     * @param deviceA Pointer to the device array A.
     * @param hostAP Pointer to the host array to be copied.
     * @param sizeA Size of the memory to be copied in bytes.
     */
    cudaMemcpy( deviceA, hostAP, sizeA, cudaMemcpyHostToDevice );

    int threadsPerBlock = getMaxThreadsPerMultiProcessor( deviceContextStructP );
    int blocksPerGrid = ( elementNum + threadsPerBlock - 1 ) / threadsPerBlock;

    /**
     * @brief Launch the arrayPowerKernel on the device.
     *
     * @param blocksPerGrid Number of blocks in the grid.
     * @param threadsPerBlock Number of threads per block.
     * @param deviceA Pointer to the device array A.
     * @param alpha The exponent to raise the array elements to.
     * @param elementNum The number of elements in the array.
     */
    arrayPowerKernel<<< blocksPerGrid, threadsPerBlock >>>( deviceA, alpha, elementNum );

    /**
     * @brief Copy the result from device array A to the host array hostAP.
     *
     * @param hostAP Pointer to the host array where the result will be copied.
     * @param deviceA Pointer to the device array A.
     * @param sizeA Size of the memory to be copied in bytes.
     */
    cudaMemcpy( hostAP, deviceA, sizeA, cudaMemcpyDeviceToHost );

    /**
     * @brief Free the allocated memory on the device.
     *
     * @param deviceA Pointer to the device array A.
     */
    cudaFree(deviceA);
    
}

/**
 * @brief CUDA kernel function to calculate the square root of an array of elements
 * 
 * @param deviceA pointer to the array of elements on the device
 * @param elementNum number of elements in the array
 */
__global__ void arraySquareRootKernel( double *deviceA, int elementNum ){
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if( i < elementNum) {
        deviceA[i] =  sqrt( deviceA[i] );
    }
}

/**
 * @brief Calculates the square root of elements in an array
 * 
 * @param deviceContextStructP pointer to the device context structure
 * @param hostAP pointer to the array of elements on the host
 * @param elementNum number of elements in the array
 */
void arraySquareRoot( deviceContextStruct * deviceContextStructP, double * hostAP, int elementNum ){
    /**
     * @brief Calculate the size of memory needed for array A on the device.
     *
     * @param elementNum The number of elements in the array.
     * @return The size of memory required in bytes.
     */
    int sizeA = elementNum * sizeof(double);

    double * deviceA;
    /**
     * @brief Allocate memory on the device for array deviceA.
     *
     * @param deviceA Pointer to the device array A.
     * @param sizeA Size of the memory to be allocated in bytes.
     */
    cudaMalloc( (void **) &deviceA, sizeA );

    /**
     * @brief Copy the host array hostAP to device array A.
     *
     * @param deviceA Pointer to the device array A.
     * @param hostAP Pointer to the host array to be copied.
     * @param sizeA Size of the memory to be copied in bytes.
     */
    cudaMemcpy( deviceA, hostAP, sizeA, cudaMemcpyHostToDevice );

    int threadsPerBlock = getMaxThreadsPerMultiProcessor( deviceContextStructP );
    int blocksPerGrid = ( elementNum + threadsPerBlock - 1 ) / threadsPerBlock;

    /**
     * @brief Launch the arraySquareRootKernel on the device.
     *
     * @param blocksPerGrid Number of blocks in the grid.
     * @param threadsPerBlock Number of threads per block.
     * @param deviceA Pointer to the device array A.
     * @param elementNum The number of elements in the array.
     */
    arraySquareRootKernel<<< blocksPerGrid, threadsPerBlock >>>( deviceA, elementNum );

    /**
     * @brief Copy the result from device array A to the host array hostAP.
     *
     * @param hostAP Pointer to the host array where the result will be copied.
     * @param deviceA Pointer to the device array A.
     * @param sizeA Size of the memory to be copied in bytes.
     */
    cudaMemcpy( hostAP, deviceA, sizeA, cudaMemcpyDeviceToHost );

    /**
     * @brief Free the allocated memory on the device.
     *
     * @param deviceA Pointer to the device array A.
     */
    cudaFree(deviceA);
}

/**
 * @brief CUDA kernel function to calculate the cube root of an array of elements
 * 
 * @param deviceA pointer to the array of elements on the device
 * @param elementNum number of elements in the array
 */
__global__ void arrayCubeRootKernel( double *deviceA, int elementNum ){
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if( i < elementNum) {
        deviceA[i] =  cbrt( deviceA[i] );
    }
}

/**
 * @brief Calculates the cube root of elements in an array
 * 
 * @param deviceContextStructP pointer to the device context structure
 * @param hostAP pointer to the array of elements on the host
 * @param elementNum number of elements in the array
 */
void arrayCubeRoot( deviceContextStruct * deviceContextStructP, double * hostAP, int elementNum ){
    /**
     * @brief Calculate the size of memory needed for array A on the device.
     *
     * @param elementNum The number of elements in the array.
     * @return The size of memory required in bytes.
     */
    int sizeA = elementNum * sizeof(double);

    double * deviceA;
    /**
     * @brief Allocate memory on the device for array deviceA.
     *
     * @param deviceA Pointer to the device array A.
     * @param sizeA Size of the memory to be allocated in bytes.
     */
    cudaMalloc( (void **) &deviceA, sizeA );

    /**
     * @brief Copy the host array hostAP to device array A.
     *
     * @param deviceA Pointer to the device array A.
     * @param hostAP Pointer to the host array to be copied.
     * @param sizeA Size of the memory to be copied in bytes.
     */
    cudaMemcpy( deviceA, hostAP, sizeA, cudaMemcpyHostToDevice );

    int threadsPerBlock = getMaxThreadsPerMultiProcessor( deviceContextStructP );
    int blocksPerGrid = ( elementNum + threadsPerBlock - 1 ) / threadsPerBlock;

    /**
     * @brief Launch the arrayCubeRootKernel on the device.
     *
     * @param blocksPerGrid Number of blocks in the grid.
     * @param threadsPerBlock Number of threads per block.
     * @param deviceA Pointer to the device array A.
     * @param elementNum The number of elements in the array.
     */
    arrayCubeRootKernel<<< blocksPerGrid, threadsPerBlock >>>( deviceA, elementNum );

    /**
     * @brief Copy the result from device array A to the host array hostAP.
     *
     * @param hostAP Pointer to the host array where the result will be copied.
     * @param deviceA Pointer to the device array A.
     * @param sizeA Size of the memory to be copied in bytes.
     */
    cudaMemcpy( hostAP, deviceA, sizeA, cudaMemcpyDeviceToHost );

    /**
     * @brief Free the allocated memory on the device.
     *
     * @param deviceA Pointer to the device array A.
     */
    cudaFree(deviceA);
}

/**
 * @brief CUDA kernel function to calculate the natural logarithm (base e) of an array of elements
 * 
 * @param deviceA pointer to the array of elements on the device
 * @param elementNum number of elements in the array
 */
__global__ void logEArrayKernel( double *deviceA, int elementNum ){
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if( i < elementNum) {
        deviceA[i] =  log( deviceA[i] );
    }
}

/**
 * @brief Calculates the natural logarithm (base e) of elements in an array
 * 
 * @param deviceContextStructP pointer to the device context structure
 * @param hostAP pointer to the array of elements on the host
 * @param elementNum number of elements in the array
 */
void logEArray( deviceContextStruct * deviceContextStructP, double * hostAP, int elementNum ){
    /**
     * @brief Calculate the size of memory needed for array A on the device.
     *
     * @param elementNum The number of elements in the array.
     * @return The size of memory required in bytes.
     */
    int sizeA = elementNum * sizeof(double);

    double * deviceA;
    /**
     * @brief Allocate memory on the device for array deviceA.
     *
     * @param deviceA Pointer to the device array A.
     * @param sizeA Size of the memory to be allocated in bytes.
     */
    cudaMalloc( (void **) &deviceA, sizeA );

    /**
     * @brief Copy the host array hostAP to device array A.
     *
     * @param deviceA Pointer to the device array A.
     * @param hostAP Pointer to the host array to be copied.
     * @param sizeA Size of the memory to be copied in bytes.
     */
    cudaMemcpy( deviceA, hostAP, sizeA, cudaMemcpyHostToDevice );

    int threadsPerBlock = getMaxThreadsPerMultiProcessor( deviceContextStructP );
    int blocksPerGrid = ( elementNum + threadsPerBlock - 1 ) / threadsPerBlock;

    /**
     * @brief Launch the logEArray kernel on the device.
     *
     * @param blocksPerGrid Number of blocks in the grid.
     * @param threadsPerBlock Number of threads per block.
     * @param deviceA Pointer to the device array A.
     * @param elementNum The number of elements in the array.
     */
    logEArrayKernel<<< blocksPerGrid, threadsPerBlock >>>( deviceA, elementNum );

    /**
     * @brief Copy the result from device array A to the host array hostAP.
     *
     * @param hostAP Pointer to the host array where the result will be copied.
     * @param deviceA Pointer to the device array A.
     * @param sizeA Size of the memory to be copied in bytes.
     */
    cudaMemcpy( hostAP, deviceA, sizeA, cudaMemcpyDeviceToHost );

    /**
     * @brief Free the allocated memory on the device.
     *
     * @param deviceA Pointer to the device array A.
     */
    cudaFree(deviceA);
}

/**
 * @brief CUDA kernel function to calculate the logarithm (base 2) of an array of elements
 * 
 * @param deviceA pointer to the array of elements on the device
 * @param elementNum number of elements in the array
 */
__global__ void log2ArrayKernel( double *deviceA, int elementNum ){
    /**
     * @brief Calculate the global index for the current thread.
     *
     * @param blockDim.x Number of threads per block in the x-dimension.
     * @param blockIdx.x Index of the current block in the x-dimension.
     * @param threadIdx.x Index of the current thread in the x-dimension.
     * @return The global index for the current thread.
     */
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if( i < elementNum) {
        /**
         * @brief Compute the base-2 logarithm of the element at index i in deviceA and store the result back in deviceA.
         *
         * @param deviceA Pointer to the device array A.
         * @param i Index of the element in the array.
         */
        deviceA[i] =  log2( deviceA[i] );
    } 
}

/**
 * @brief Calculates the logarithm (base 2) of elements in an array
 * 
 * @param deviceContextStructP pointer to the device context structure
 * @param hostAP pointer to the array of elements on the host
 * @param elementNum number of elements in the array
 */
void log2Array( deviceContextStruct * deviceContextStructP, double * hostAP, int elementNum ){
    /**
     * @brief Calculate the size of memory needed for array A on the device.
     *
     * @param elementNum The number of elements in the array.
     * @return The size of memory required in bytes.
     */
    int sizeA = elementNum * sizeof(double);

    double * deviceA;
    /**
     * @brief Allocate memory on the device for array deviceA.
     *
     * @param deviceA Pointer to the device array A.
     * @param sizeA Size of the memory to be allocated in bytes.
     */
    cudaMalloc( (void **) &deviceA, sizeA );

    /**
     * @brief Copy the host array hostAP to device array A.
     *
     * @param deviceA Pointer to the device array A.
     * @param hostAP Pointer to the host array to be copied.
     * @param sizeA Size of the memory to be copied in bytes.
     */
    cudaMemcpy( deviceA, hostAP, sizeA, cudaMemcpyHostToDevice );

    int threadsPerBlock = getMaxThreadsPerMultiProcessor( deviceContextStructP );
    int blocksPerGrid = ( elementNum + threadsPerBlock - 1 ) / threadsPerBlock;

    /**
     * @brief Launch the log2Array kernel on the device.
     *
     * @param blocksPerGrid Number of blocks in the grid.
     * @param threadsPerBlock Number of threads per block.
     * @param deviceA Pointer to the device array A.
     * @param elementNum The number of elements in the array.
     */
    log2ArrayKernel<<< blocksPerGrid, threadsPerBlock >>>( deviceA, elementNum );

    /**
     * @brief Copy the result from device array A to the host array hostAP.
     *
     * @param hostAP Pointer to the host array where the result will be copied.
     * @param deviceA Pointer to the device array A.
     * @param sizeA Size of the memory to be copied in bytes.
     */
    cudaMemcpy( hostAP, deviceA, sizeA, cudaMemcpyDeviceToHost );

    /**
     * @brief Free the allocated memory on the device.
     *
     * @param deviceA Pointer to the device array A.
     */
    cudaFree(deviceA);
}

/**
 * @brief CUDA kernel function to calculate the logarithm (base 10) of an array of elements
 * 
 * @param deviceA pointer to the array of elements on the device
 * @param elementNum number of elements in the array
 */
__global__ void log10ArrayKernel( double *deviceA, int elementNum ){
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if( i < elementNum) {
        deviceA[i] =  log10( deviceA[i] );
    }
}

/**
 * @brief Calculates the logarithm (base 10) of elements in an array
 * 
 * @param deviceContextStructP pointer to the device context structure
 * @param hostAP pointer to the array of elements on the host
 * @param elementNum number of elements in the array
 */
void log10Array( deviceContextStruct * deviceContextStructP, double * hostAP, int elementNum ){
    /**
     * @brief Calculate the size of memory needed for array A on the device.
     *
     * @param elementNum The number of elements in the array.
     * @return The size of memory required in bytes.
     */
    int sizeA = elementNum * sizeof(double);

    double * deviceA;
    /**
     * @brief Allocate memory on the device for array deviceA.
     *
     * @param deviceA Pointer to the device array A.
     * @param sizeA Size of the memory to be allocated in bytes.
     */
    cudaMalloc( (void **) &deviceA, sizeA );

    /**
     * @brief Copy the host array hostAP to device array A.
     *
     * @param deviceA Pointer to the device array A.
     * @param hostAP Pointer to the host array to be copied.
     * @param sizeA Size of the memory to be copied in bytes.
     */
    cudaMemcpy( deviceA, hostAP, sizeA, cudaMemcpyHostToDevice );

    int threadsPerBlock = getMaxThreadsPerMultiProcessor( deviceContextStructP );
    int blocksPerGrid = ( elementNum + threadsPerBlock - 1 ) / threadsPerBlock;

    /**
     * @brief Launch the log10Array kernel on the device.
     *
     * @param blocksPerGrid Number of blocks in the grid.
     * @param threadsPerBlock Number of threads per block.
     * @param deviceA Pointer to the device array A.
     * @param elementNum The number of elements in the array.
     */
    log10ArrayKernel<<< blocksPerGrid, threadsPerBlock >>>( deviceA, elementNum );

    /**
     * @brief Copy the result from device array A to the host array hostAP.
     *
     * @param hostAP Pointer to the host array where the result will be copied.
     * @param deviceA Pointer to the device array A.
     * @param sizeA Size of the memory to be copied in bytes.
     */
    cudaMemcpy( hostAP, deviceA, sizeA, cudaMemcpyDeviceToHost );

    /**
     * @brief Free the allocated memory on the device.
     *
     * @param deviceA Pointer to the device array A.
     */
    cudaFree(deviceA);
}

/**
 * @brief CUDA kernel function to calculate the Hadamard product of two arrays
 * 
 * @param deviceA pointer to the first array on the device
 * @param deviceB pointer to the second array on the device
 * @param elementNum number of elements in the arrays
 */
__global__ void hadamardProductKernel( double * deviceA, double * deviceB, int elementNum ){
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    
    if( i < elementNum ){
        deviceA[i] = deviceA[i] * deviceB[i];
    }
}

/**
 * @brief Calculates the Hadamard product of two arrays
 * 
 * @param deviceContextStructP pointer to the device context structure
 * @param hostAP pointer to the first array on the host
 * @param hostBP pointer to the second array on the host
 * @param elementNum number of elements in the arrays
 */
void hadamardProduct( deviceContextStruct * deviceContextStructP, double * hostAP, double * hostBP, int elementNum ){
    /**
     * @brief Calculate the size of the memory needed for arrays A and B on the device.
     *
     * @param elementNum The number of elements in the arrays.
     * @return The size of memory required in bytes.
     */
    int sizeA = elementNum * sizeof(double);

    double * deviceA, * deviceB;
    /**
     * @brief Allocate memory on the device for arrays deviceA and deviceB.
     *
     * @param deviceA Pointer to the device array A.
     * @param deviceB Pointer to the device array B.
     * @param sizeA Size of the memory to be allocated in bytes.
     */
    cudaMalloc( (void **) &deviceA, sizeA );
    cudaMalloc( (void **) &deviceB, sizeA );

    /**
     * @brief Copy the host array hostAP to device array A.
     *
     * @param deviceA Pointer to the device array A.
     * @param hostAP Pointer to the host array to be copied.
     * @param sizeA Size of the memory to be copied in bytes.
     */
    cudaMemcpy( deviceA, hostAP, sizeA, cudaMemcpyHostToDevice );
    /**
     * @brief Copy the host array hostBP to device array B.
     *
     * @param deviceB Pointer to the device array B.
     * @param hostBP Pointer to the host array to be copied.
     * @param sizeA Size of the memory to be copied in bytes.
     */
    cudaMemcpy( deviceB, hostBP, sizeA, cudaMemcpyHostToDevice );

    int threadsPerBlock = getMaxThreadsPerMultiProcessor( deviceContextStructP );
    int blocksPerGrid = ( elementNum + threadsPerBlock - 1 ) / threadsPerBlock;

    /**
     * @brief Launch the Hadamard product kernel on the device.
     *
     * @param blocksPerGrid Number of blocks in the grid.
     * @param threadsPerBlock Number of threads per block.
     * @param deviceA Pointer to the device array A.
     * @param deviceB Pointer to the device array B.
     * @param elementNum The number of elements in the arrays.
     */
    hadamardProductKernel<<< blocksPerGrid, threadsPerBlock >>>( deviceA, deviceB, elementNum );

    /**
     * @brief Copy the result from device array A to the host array hostAP.
     *
     * @param hostAP Pointer to the host array where the result will be copied.
     * @param deviceA Pointer to the device array A.
     * @param sizeA Size of the memory to be copied in bytes.
     */
    cudaMemcpy( hostAP, deviceA, sizeA, cudaMemcpyDeviceToHost );

    /**
     * @brief Free the allocated memory on the device.
     *
     * @param deviceA Pointer to the device array A.
     */
    cudaFree(deviceA);
}

/**
 * @brief CUDA kernel function to transpose a matrix
 * 
 * @param deviceA pointer to the input matrix on the device
 * @param elementNum number of elements in the matrix
 * @param heightA height of the input matrix
 * @param widthA width of the input matrix
 * @param deviceB pointer to the transposed matrix on the device
 */
__global__ void transposeKernel( double *deviceA, int elementNum, int heightA, int widthA, double *deviceB ){
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if( i < elementNum ) {
        int bI = i / heightA;
        int bJ = i % heightA;
        deviceB[i] =  deviceA[ bJ * widthA + bI ];
    }
}

/**
 * @brief Transposes a matrix
 * 
 * @param deviceContextStructP pointer to the device context structure
 * @param hostAP pointer to the input matrix on the host
 * @param elementNum number of elements in the matrix
 * @param heightA height of the input matrix
 * @param wigthA width of the input matrix
 */
void transpose( deviceContextStruct * deviceContextStructP, double * hostAP, int elementNum, int heightA, int wigthA ){
    /**
     * @brief Allocate memory on the device for arrays deviceA and deviceB.
     *
     * @param elementNum The number of elements in the arrays.
     * @param hostAP Pointer to the host array to be copied to deviceA.
     * @param deviceA Pointer to the device array A.
     * @param deviceB Pointer to the device array B.
     */
    int sizeA = elementNum * sizeof(double);

    double * deviceA, * deviceB;
    cudaMalloc( (void **) &deviceA, sizeA );
    cudaMalloc( (void **) &deviceB, sizeA );

    /**
     * @brief Copy the host array to device array A.
     *
     * @param deviceA Pointer to the device array A.
     * @param hostAP Pointer to the host array to be copied.
     * @param sizeA Size of the memory to be copied in bytes.
     */
    cudaMemcpy( deviceA, hostAP, sizeA, cudaMemcpyHostToDevice );

    int threadsPerBlock = getMaxThreadsPerMultiProcessor( deviceContextStructP );
    int blocksPerGrid = ( elementNum + threadsPerBlock - 1 ) / threadsPerBlock;

    /**
     * @brief Launch the transpose kernel on the device.
     *
     * @param blocksPerGrid Number of blocks in the grid.
     * @param threadsPerBlock Number of threads per block.
     * @param deviceA Pointer to the device array A.
     * @param elementNum The number of elements in the arrays.
     * @param heightA Height of the matrix A.
     * @param widthA Width of the matrix A.
     * @param deviceB Pointer to the device array B.
     */
    transposeKernel<<< blocksPerGrid, threadsPerBlock >>>( deviceA, elementNum, heightA, wigthA, deviceB );

    /**
     * @brief Copy the result from device array B to the host array.
     *
     * @param hostAP Pointer to the host array where the result will be copied.
     * @param deviceB Pointer to the device array B.
     * @param sizeA Size of the memory to be copied in bytes.
     */
    cudaMemcpy( hostAP, deviceB, sizeA, cudaMemcpyDeviceToHost );

    /**
     * @brief Free the allocated memory on the device.
     *
     * @param deviceA Pointer to the device array A.
     * @param deviceB Pointer to the device array B.
     */
    cudaFree(deviceA);
    cudaFree(deviceB);
}
