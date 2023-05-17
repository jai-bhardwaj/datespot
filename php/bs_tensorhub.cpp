#include <stdio.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#ifdef HAVE_CONFIG_H
# include "config.h"
#endif

#include "php.h"
#include <zend_exceptions.h>
#include "ext/standard/info.h"
#include "dev_util_c.h"
#include "dev_util_p.h"
#include "php_bs_tensorhub.h"
#include "php_bs_util.h"
#include "php_bs_math.h"

#ifndef ZEND_PARSE_PARAMETERS_NONE
/**
 * \brief Macro to parse parameters with zero arguments.
 */
#define ZEND_PARSE_PARAMETERS_NONE() \
    ZEND_PARSE_PARAMETERS_START(0, 0) \
    ZEND_PARSE_PARAMETERS_END()
#endif

/**
 * \brief Structure to hold a number.
 */
typedef struct {
    /**
     * @var int num
     * The number.
     */
    int num;
} numStruct;

/**
 * \brief Structure to hold a cublas handle.
 */
typedef struct {
    /**
     * @var cublasHandle_t handle
     * The cublas handle.
     */
    cublasHandle_t handle;
} cublasHandleStruct;

/**
 * \brief The class entry for the BLAS class.
 */
zend_class_entry* BLAS_ce;

/**
 * \brief The external class entry for the Util class.
 */
extern zend_class_entry* Util_ce;

/**
 * \brief The function entries for the Util class.
 */
extern zend_function_entry Util_functions[];

/**
 * \brief The external class entry for the Math class.
 */
extern zend_class_entry* Math_ce;

/**
 * \brief The function entries for the Math class.
 */
extern zend_function_entry Math_functions[];

/**
 * \brief The resource number for the handle resource.
 */
static int handleResourceNum;

/**
 * \brief Prints a double array in the device.
 *
 * @param C The double array.
 * @param height The height of the array.
 * @param width The width of the array.
 */
void dev_printtensorhub(double* C, unsigned int height, unsigned int width) {
    /**
     * Prints a separator line using php_printf.
     */
    php_printf("---------------------------\n");

    /**
     * Prints the contents of a matrix C row by row using php_printf.
     *
     * @param height The number of rows in the matrix C.
     * @param width The number of columns in the matrix C.
     * @param C The matrix to be printed.
     */
    for (unsigned int i = 0; i < height; ++i) {
        for (unsigned int j = 0; j < width; ++j) {
            php_printf("%f  ", C[i * width + j]);
        }
        php_printf("\n");
    }
}

/**
 * \brief Destructor for the handle resource.
 *
 * @param rsrc The handle resource.
 */
static void handleResourceDescontructor(zend_resource* rsrc) {
    /**
     * Pointer to a `cublasHandleStruct` object obtained from the resource pointer.
     */
    cublasHandleStruct* pointer = (cublasHandleStruct*)rsrc->ptr;

    /**
     * Checks if the pointer is not NULL before destroying the associated cublasHandle.
     * Frees the memory allocated for the pointer and sets the resource pointer to NULL.
     */
    if (pointer) {
        cublasDestroy(pointer->handle);
        efree(pointer);
        rsrc->ptr = NULL;
    }
}

/**
 * @brief Information about the arguments for the BLAS::__construct method.
 *
 * This macro defines the argument information for the BLAS::__construct method.
 */
ZEND_BEGIN_ARG_INFO_EX(BLAS_construct_ArgInfo, 0, 0, 0)
ZEND_END_ARG_INFO()

/**
 * @brief BLAS::__construct method.
 *
 * Constructor method for the BLAS class.
 *
 * @throws An exception with code 1001 if no CUDA devices are available.
 */
PHP_METHOD(BLAS, __construct) {
    /**
     * Variable to store the number of CUDA devices available on the system.
     */
    int deviceCount = 0;

    /**
     * Retrieves the number of CUDA devices available on the system and stores it in `deviceCount`.
     */
    cudaGetDeviceCount(&deviceCount);

    /**
     * Checks if there are no CUDA devices available.
     * Throws a custom exception with error code 1001 and an error message retrieved using `duc_getErrorMsg`.
     */
    if (deviceCount == 0) {
        zend_throw_exception_ex(nullptr, 1001, duc_getErrorMsg(1001), nullptr);
    }

    /**
     * Pointer to a zend_resource struct to store the registered cublasHandle resource.
     */
    zend_resource *cublasHandleResourceP;

    /**
     * Allocates memory for a cublasHandleStruct object and initializes it to zero.
     */
    cublasHandleStruct *cublasHandleStructP = (cublasHandleStruct *)ecalloc(1, sizeof(cublasHandleStruct));

    /**
     * Variable to store the cublasHandle.
     */
    cublasHandle_t handle;

    /**
     * Creates a cublasHandle and stores it in `handle`.
     */
    checkCudaResult(cublasCreate(&handle));

    /**
     * Assigns the created cublasHandle to the cublasHandleStruct.
     */
    cublasHandleStructP->handle = handle;

    /**
     * Registers the cublasHandleStruct pointer as a resource and assigns the resource number to `cublasHandleResourceP`.
     */
    cublasHandleResourceP = zend_register_resource(cublasHandleStructP, handleResourceNum);

    /**
     * Variable to store the cublasHandle resource as a zval.
     */
    zval cudaHandle;

    /**
     * Sets the zval as a resource type with the registered cublasHandle resource.
     */
    ZVAL_RES(&cudaHandle, cublasHandleResourceP);

    /**
     * Updates the "cublasHandle" property of the current object with the cublasHandle zval.
     */
    zend_update_property(BLAS_ce, getThis(), "cublasHandle", sizeof("cublasHandle") - 1, &cudaHandle);
}

/**
 * @brief Information about the arguments for the BLAS::setHandle method.
 *
 * This macro defines the argument information for the BLAS::setHandle method.
 *
 * @param cublasHandleP The cublasHandle to be set.
 */
ZEND_BEGIN_ARG_INFO_EX(BLAS_setHandle_ArgInfo, 0, 0, 1)
    ZEND_ARG_INFO(0, cublasHandleP)
ZEND_END_ARG_INFO()

/**
 * @brief BLAS::setHandle method.
 *
 * Sets the cublasHandle property of the object.
 *
 * @param cublasHandleP The cublasHandle to be set.
 */
PHP_METHOD(BLAS, setHandle) {
    zval *cublasHandleP = nullptr;

    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_RESOURCE(cublasHandleP)
    ZEND_PARSE_PARAMETERS_END();

    zend_update_property(BLAS_ce, getThis(), "cublasHandle", sizeof("cublasHandle") - 1, cublasHandleP);
}

/**
 * @brief Information about the arguments for the BLAS::getHandle method.
 *
 * This macro defines the argument information for the BLAS::getHandle method.
 */
ZEND_BEGIN_ARG_INFO_EX(BLAS_getHandle_ArgInfo, 0, 0, 0)
ZEND_END_ARG_INFO()

/**
 * @brief BLAS::getHandle method.
 *
 * Retrieves the cublasHandle property of the object.
 *
 * @return Returns the cublasHandle property as a zval.
 */
PHP_METHOD(BLAS, getHandle) {
    zval *obj = getThis();
    zval *tempZVal;

    zval *cublasHandleP = zend_read_property(BLAS_ce, obj, "cublasHandle", sizeof("cublasHandle") - 1, 1, tempZVal TSRMLS_CC);
    cublasHandleStruct *temp = (cublasHandleStruct *)zend_fetch_resource(Z_RES_P(cublasHandleP), "handleResourceName", handleResourceNum);

    RETURN_ZVAL(cublasHandleP, 1, 0);
}

ZEND_BEGIN_ARG_INFO_EX( BLAS_multiply_ArgInfo, 0, 0, 2)
    ZEND_ARG_ARRAY_INFO( 1, tensorhubAP, 0 )
    ZEND_ARG_ARRAY_INFO( 1, tensorhubBP, 0 )
    ZEND_ARG_ARRAY_INFO( 1, tensorhubCP, 1 )
    ZEND_ARG_INFO( 0, alpha )
    ZEND_ARG_INFO( 0, beta )
ZEND_END_ARG_INFO()

PHP_METHOD(BLAS, multiply){
    zval * tensorhubAP = NULL;
    zval * tensorhubBP = NULL;
    zval * tensorhubCP = NULL;
    double alpha = 1.0;
    double beta = 0.0;

    ZEND_PARSE_PARAMETERS_START(2, 5)
        Z_PARAM_ARRAY_EX( tensorhubAP, 0, 1 )
        Z_PARAM_ARRAY_EX( tensorhubBP, 0, 1 )
        Z_PARAM_OPTIONAL
        Z_PARAM_ARRAY_EX( tensorhubCP, 0, 1 )
        Z_PARAM_DOUBLE(alpha)
        Z_PARAM_DOUBLE(beta)
    ZEND_PARSE_PARAMETERS_END();

    /**
     * Pointer to the HashTable of tensorhubAP.
     */
    HashTable * hashTableAP = Z_ARRVAL_P(tensorhubAP);

    /**
     * Checks if the value stored in hashTableAP is not an array.
     * Throws a custom exception with error code 2001 and an error message retrieved using `duc_getErrorMsg`.
     */
    if (Z_TYPE((hashTableAP->arData)->val) != IS_ARRAY) {
        zend_throw_exception_ex(NULL, 2001, duc_getErrorMsg(2001), NULL);
    }

    /**
     * Number of elements in hashTableAP, representing the height of matrix A.
     */
    int heightA = zend_hash_num_elements(hashTableAP);

    /**
     * Number of elements in the nested array of hashTableAP, representing the width of matrix A.
     */
    int widthA = zend_hash_num_elements(Z_ARRVAL((hashTableAP->arData)->val));

    /**
     * Pointer to the HashTable of tensorhubBP.
     */
    HashTable * hashTableBP = Z_ARRVAL_P(tensorhubBP);

    /**
     * Checks if the value stored in hashTableBP is not an array.
     * Throws a custom exception with error code 2002 and an error message retrieved using `duc_getErrorMsg`.
     */
    if (Z_TYPE((hashTableBP->arData)->val) != IS_ARRAY) {
        zend_throw_exception_ex(NULL, 2002, duc_getErrorMsg(2002), NULL);
    }

    /**
     * Number of elements in hashTableBP, representing the height of matrix B.
     */
    int heightB = zend_hash_num_elements(hashTableBP);

    /**
     * Number of elements in the nested array of hashTableBP, representing the width of matrix B.
     */
    int widthB = zend_hash_num_elements(Z_ARRVAL((hashTableBP->arData)->val));

    /**
     * Pointer to the HashTable of tensorhubCP.
     */
    HashTable * hashTableCP = NULL;

    /**
     * Number of elements in hashTableCP, representing the height of matrix C.
     */
    int heightC = 0;

    /**
     * Number of elements in the nested array of hashTableCP, representing the width of matrix C.
     */
    int widthC = 0;

    /**
     * Checks if tensorhubCP is not NULL.
     * If not NULL, assigns the HashTable of tensorhubCP to hashTableCP.
     * Checks if the value stored in hashTableCP is not an array.
     * Throws a custom exception with error code 2003 and an error message retrieved using `duc_getErrorMsg`.
     */
    if (tensorhubCP != NULL) {
        hashTableCP = Z_ARRVAL_P(tensorhubCP);
        if (Z_TYPE((hashTableCP->arData)->val) != IS_ARRAY) {
            zend_throw_exception_ex(NULL, 2003, duc_getErrorMsg(2003), NULL);
        }

        /**
         * Number of elements in hashTableCP, representing the height of matrix C.
         */
        heightC = zend_hash_num_elements(hashTableCP);

        /**
         * Number of elements in the nested array of hashTableCP, representing the width of matrix C.
         */
        widthC = zend_hash_num_elements(Z_ARRVAL((hashTableCP->arData)->val));

        /**
         * Checks if the height of matrix C is not equal to the height of matrix A.
         * Throws a custom exception with error code 2004 and an error message retrieved using `duc_getErrorMsg`.
         */
        if (heightC != heightA) {
            zend_throw_exception_ex(NULL, 2004, duc_getErrorMsg(2004), heightA, widthA, heightC, widthC);
        }

        /**
         * Checks if the width of matrix C is not equal to the width of matrix B.
         * Throws a custom exception with error code 2006 and an error message retrieved using `duc_getErrorMsg`.
         *
         * @param heightB The height of matrix B.
         * @param widthB The width of matrix B.
         * @param heightC The height of matrix C.
         * @param widthC The width of matrix C.
         */
        if (widthC != widthB) {
            zend_throw_exception_ex(NULL, 2006, duc_getErrorMsg(2006), heightB, widthB, heightC, widthC);
        }

    }

    if( widthA !=  heightB ){
        zend_throw_exception_ex(NULL, 2000, duc_getErrorMsg(2000), heightA, widthA, heightB, widthB );
    }

    double * hostAP = ( double * )malloc( heightA * widthA * sizeof(double) );
    double * hostBP = ( double * )malloc( heightB * widthB * sizeof(double) );
    double * hostCP = ( double * )calloc( heightA * widthB, sizeof(double) );

    dup_HashTableTo1DArr( hashTableAP, hostAP );
    dup_HashTableTo1DArr( hashTableBP, hostBP );
    if( hashTableCP != NULL ){
        dup_HashTableTo1DArr( hashTableCP, hostCP );
    }


    double * deviceAP, * deviceBP, * deviceCP;
    cudaMalloc( (void**)&deviceAP, heightA * widthA * sizeof(double) );
    cudaMalloc( (void**)&deviceBP, heightB * widthB * sizeof(double) );
    cudaMalloc( (void**)&deviceCP, heightA * widthB * sizeof(double) );

    cudaMemcpy( deviceAP, hostAP, heightA * widthA * sizeof(double), cudaMemcpyHostToDevice );
    cudaMemcpy( deviceBP, hostBP, heightB * widthB * sizeof(double), cudaMemcpyHostToDevice );
    if( hashTableCP != NULL ){
        cudaMemcpy( deviceCP, hostCP, heightC * widthC * sizeof(double), cudaMemcpyHostToDevice );
    }

    zval *obj = getThis();
    zval *tempZVal;
    zval *cublasHandleP = zend_read_property(BLAS_ce, obj, "cublasHandle", sizeof("cublasHandle") - 1, 1, tempZVal TSRMLS_CC);
    cublasHandleStruct * cudaHandleStructP = (cublasHandleStruct *)zend_fetch_resource(Z_RES_P(cublasHandleP), "handleResourceName", handleResourceNum);

    cudaEvent_t stop;
    cudaEventCreate(&stop);



    checkCudaResult(
            cublasDgemm(cudaHandleStructP->handle,
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    widthB, heightA, widthA,
                    &alpha,
                    deviceBP, widthB,
                    deviceAP, widthA,
                    &beta,
                    deviceCP, widthB
            )
    );

    cudaEventSynchronize(stop);

    cudaMemcpy( hostCP, deviceCP, heightA * widthB * sizeof(double), cudaMemcpyDeviceToHost );




    zval returnZval; array_init_size( &returnZval, heightA );

    for( int tempI = 0; tempI < heightA; tempI++ ){

        zval tempZval; array_init_size( &tempZval, widthB );
        for( int tempJ = 0; tempJ < widthB; tempJ++ ){

            add_next_index_double( &tempZval, hostCP[ tempI * widthB + tempJ ]);
        }

        add_next_index_zval( &returnZval, &tempZval );
    }

    RETVAL_ZVAL( &returnZval, 1, 1 );

    free(hostAP);
    free(hostBP);
    free(hostCP);
    cudaFree(deviceAP);
    cudaFree(deviceBP);
    cudaFree(deviceCP);

    return ;

}

ZEND_BEGIN_ARG_INFO_EX( BLAS_multiplyS_ArgInfo, 0, 0, 2)
    ZEND_ARG_ARRAY_INFO( 1, tensorhubAP, 0 )
    ZEND_ARG_ARRAY_INFO( 1, tensorhubBP, 0 )
    ZEND_ARG_ARRAY_INFO( 1, tensorhubCP, 1 )
    ZEND_ARG_INFO( 0, alpha )
    ZEND_ARG_INFO( 0, beta )
ZEND_END_ARG_INFO()

PHP_METHOD(BLAS, multiplyS){
    zval * tensorhubAP = NULL;
    zval * tensorhubBP = NULL;
    zval * tensorhubCP = NULL;
    float alpha; double alphaTemp = 1.0;
    float beta; double betaTemp = 0.0;

    ZEND_PARSE_PARAMETERS_START(2, 5)
        Z_PARAM_ARRAY_EX( tensorhubAP, 0, 1 )
        Z_PARAM_ARRAY_EX( tensorhubBP, 0, 1 )
        Z_PARAM_OPTIONAL
        Z_PARAM_ARRAY_EX( tensorhubCP, 0, 1 )
        Z_PARAM_DOUBLE(alphaTemp)
        Z_PARAM_DOUBLE(betaTemp)
    ZEND_PARSE_PARAMETERS_END();

    alpha = (float)alphaTemp;
    beta = (float)betaTemp;

    HashTable * hashTableAP = Z_ARRVAL_P(tensorhubAP);
    if( Z_TYPE( (hashTableAP->arData)->val ) != IS_ARRAY ){
        zend_throw_exception_ex(NULL, 2001, duc_getErrorMsg(2001), NULL );
    }
    int heightA = zend_hash_num_elements(hashTableAP);
    int widthA = zend_hash_num_elements( Z_ARRVAL( (hashTableAP->arData)->val ) );

    HashTable * hashTableBP = Z_ARRVAL_P(tensorhubBP);
    if( Z_TYPE( (hashTableBP->arData)->val ) != IS_ARRAY ){
        zend_throw_exception_ex(NULL, 2002,  duc_getErrorMsg(2002), NULL );
    }
    int heightB = zend_hash_num_elements(hashTableBP);
    int widthB = zend_hash_num_elements( Z_ARRVAL( (hashTableBP->arData)->val ) );

    HashTable * hashTableCP = NULL;
    int heightC = 0;
    int widthC = 0;
    if( tensorhubCP != NULL ){
        hashTableCP = Z_ARRVAL_P(tensorhubCP);
        if( Z_TYPE( (hashTableCP->arData)->val ) != IS_ARRAY ){
            zend_throw_exception_ex(NULL, 2003, duc_getErrorMsg(2003), NULL );
        }
        heightC = zend_hash_num_elements(hashTableCP);
        widthC = zend_hash_num_elements( Z_ARRVAL( (hashTableCP->arData)->val ) );

        if( heightC != heightA ){
            zend_throw_exception_ex(NULL, 2004,  duc_getErrorMsg(2004), heightA, widthA, heightC, widthC );
        }

        if( widthC != widthB ){
            zend_throw_exception_ex(NULL, 2005, duc_getErrorMsg(2005), heightB, widthB, heightC, widthC  );
        }

    }

    if( widthA !=  heightB ){
        zend_throw_exception_ex(NULL, 2000, duc_getErrorMsg(2000), heightA, widthA, heightB, widthB );
    }

    float * hostAP = ( float * )malloc( heightA * widthA * sizeof(float) );
    float * hostBP = ( float * )malloc( heightB * widthB * sizeof(float) );
    float * hostCP = ( float * )calloc( heightA * widthB, sizeof(float) );

    dup_HashTableTo1DArrS( hashTableAP, hostAP );
    dup_HashTableTo1DArrS( hashTableBP, hostBP );
    if( hashTableCP != NULL ){
        dup_HashTableTo1DArrS( hashTableCP, hostCP );
    }


    float * deviceAP, * deviceBP, * deviceCP;
    cudaMalloc( (void**)&deviceAP, heightA * widthA * sizeof(float) );
    cudaMalloc( (void**)&deviceBP, heightB * widthB * sizeof(float) );
    cudaMalloc( (void**)&deviceCP, heightA * widthB * sizeof(float) );

    cudaMemcpy( deviceAP, hostAP, heightA * widthA * sizeof(float), cudaMemcpyHostToDevice );
    cudaMemcpy( deviceBP, hostBP, heightB * widthB * sizeof(float), cudaMemcpyHostToDevice );
    if( hashTableCP != NULL ){
        cudaMemcpy( deviceCP, hostCP, heightC * widthC * sizeof(float), cudaMemcpyHostToDevice );
    }

    zval *obj = getThis();
    zval *tempZVal;
    zval *cublasHandleP = zend_read_property(BLAS_ce, obj, "cublasHandle", sizeof("cublasHandle") - 1, 1, tempZVal TSRMLS_CC);
    cublasHandleStruct * cudaHandleStructP = (cublasHandleStruct *)zend_fetch_resource(Z_RES_P(cublasHandleP), "handleResourceName", handleResourceNum);

    cudaEvent_t stop;
    cudaEventCreate(&stop);

    checkCudaResult(
            cublasSgemm(cudaHandleStructP->handle,
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    widthB, heightA, widthA,
                    &alpha,
                    deviceBP, widthB,
                    deviceAP, widthA,
                    &beta,
                    deviceCP, widthB
            )
    );

    cudaEventSynchronize(stop);

    cudaMemcpy( hostCP, deviceCP, heightA * widthB * sizeof(float), cudaMemcpyDeviceToHost );




    zval returnZval; array_init_size( &returnZval, heightA );

    for( int tempI = 0; tempI < heightA; tempI++ ){

        zval tempZval; array_init_size( &tempZval, widthB );
        for( int tempJ = 0; tempJ < widthB; tempJ++ ){

            add_next_index_double( &tempZval, hostCP[ tempI * widthB + tempJ ]);
        }

        add_next_index_zval( &returnZval, &tempZval );
    }

    RETVAL_ZVAL( &returnZval, 1, 1 );

    free(hostAP);
    free(hostBP);
    free(hostCP);
    cudaFree(deviceAP);
    cudaFree(deviceBP);
    cudaFree(deviceCP);

    return ;

}
/**
 * @brief Information about the arguments for the BLAS::dot method.
 *
 * This macro defines the argument information for the BLAS::dot method.
 *
 * @param oneDimensionArrAP The input array A.
 * @param oneDimensionArrBP The input array B.
 * @param strideA The stride value for array A (optional, default: 1).
 * @param strideB The stride value for array B (optional, default: 1).
 */
ZEND_BEGIN_ARG_INFO_EX( BLAS_dot_ArgInfo, 0, 0, 2)
    ZEND_ARG_ARRAY_INFO( 1, oneDimensionArrAP, 0 )
    ZEND_ARG_ARRAY_INFO( 1, oneDimensionArrBP, 0 )
    ZEND_ARG_INFO( 0, strideA )
    ZEND_ARG_INFO( 0, strideB )
ZEND_END_ARG_INFO()
/**
 * @brief BLAS::dot method.
 *
 * Performs the dot product of two one-dimensional arrays.
 *
 * @param oneDimensionArrAP The input array A.
 * @param oneDimensionArrBP The input array B.
 * @param strideA The stride value for array A (optional, default: 1).
 * @param strideB The stride value for array B (optional, default: 1).
 * @return The dot product of arrays A and B.
 */
PHP_METHOD(BLAS, dot){
    zval * oneDimensionArrAP = NULL;
    zval * oneDimensionArrBP = NULL;
    zend_long strideA = 1;
    zend_long strideB = 1;


    ZEND_PARSE_PARAMETERS_START(2, 4)
        Z_PARAM_ARRAY_EX( oneDimensionArrAP, 0, 1 )
        Z_PARAM_ARRAY_EX( oneDimensionArrBP, 0, 1 )
        Z_PARAM_OPTIONAL
        Z_PARAM_LONG(strideA)
        Z_PARAM_LONG(strideB)
    ZEND_PARSE_PARAMETERS_END();

    HashTable * hashTableAP = Z_ARRVAL_P(oneDimensionArrAP);
    int elementNumA = zend_hash_num_elements(hashTableAP);

    HashTable * hashTableBP = Z_ARRVAL_P(oneDimensionArrBP);
    int elementNumB = zend_hash_num_elements(hashTableBP);

    double * hostAP = ( double * )malloc( elementNumA * sizeof(double) );
    double * hostBP = ( double * )malloc( elementNumB * sizeof(double) );
    double * hostCP = ( double * )malloc( 1 * sizeof(double) );

    dup_HashTableTo1DArrOne( hashTableAP, hostAP );
    dup_HashTableTo1DArrOne( hashTableBP, hostBP );

    double * deviceAP, * deviceBP, * deviceCP;
    cudaMalloc( (void**)&deviceAP, elementNumA * sizeof(double) );
    cudaMalloc( (void**)&deviceBP, elementNumB * sizeof(double) );
    cudaMalloc( (void**)&deviceCP, 1 * sizeof(double) );

    cudaMemcpy( deviceAP, hostAP, elementNumA * sizeof(double), cudaMemcpyHostToDevice );
    cudaMemcpy( deviceBP, hostBP, elementNumB * sizeof(double), cudaMemcpyHostToDevice );

    zval *obj = getThis();
    zval *tempZVal;
    zval *cublasHandleP = zend_read_property(BLAS_ce, obj, "cublasHandle", sizeof("cublasHandle") - 1, 1, tempZVal TSRMLS_CC);
    cublasHandleStruct * cudaHandleStructP = (cublasHandleStruct *)zend_fetch_resource(Z_RES_P(cublasHandleP), "handleResourceName", handleResourceNum);

    cudaEvent_t stop;
    cudaEventCreate(&stop);


    checkCudaResult(
            cublasDdot(cudaHandleStructP->handle,
                    elementNumA,
                    deviceAP, strideA,
                    deviceBP, strideB,
                    deviceCP
            )
    );

    cudaEventSynchronize(stop);

    cudaMemcpy( hostCP, deviceCP, 1 * sizeof(double), cudaMemcpyDeviceToHost );


    RETVAL_DOUBLE( *hostCP );

    free(hostAP);
    free(hostBP);
    free(hostCP);
    cudaFree(deviceAP);
    cudaFree(deviceBP);
    cudaFree(deviceCP);

    return ;

}

ZEND_BEGIN_ARG_INFO_EX( BLAS_dotS_ArgInfo, 0, 0, 2)
    ZEND_ARG_ARRAY_INFO( 1, oneDimensionArrAP, 0 )
    ZEND_ARG_ARRAY_INFO( 1, oneDimensionArrBP, 0 )
    ZEND_ARG_INFO( 0, strideA )
    ZEND_ARG_INFO( 0, strideB )
ZEND_END_ARG_INFO()

PHP_METHOD(BLAS, dotS){
    zval * oneDimensionArrAP = NULL;
    zval * oneDimensionArrBP = NULL;
    zend_long strideA = 1;
    zend_long strideB = 1;


    ZEND_PARSE_PARAMETERS_START(2, 4)
        Z_PARAM_ARRAY_EX( oneDimensionArrAP, 0, 1 )
        Z_PARAM_ARRAY_EX( oneDimensionArrBP, 0, 1 )
        Z_PARAM_OPTIONAL
        Z_PARAM_LONG(strideA)
        Z_PARAM_LONG(strideB)
    ZEND_PARSE_PARAMETERS_END();

    /**
     * Pointer to the HashTable of oneDimensionArrAP.
     */
    HashTable * hashTableAP = Z_ARRVAL_P(oneDimensionArrAP);

    /**
     * Number of elements in oneDimensionArrAP.
     */
    int elementNumA = zend_hash_num_elements(hashTableAP);

    /**
     * Pointer to the HashTable of oneDimensionArrBP.
     */
    HashTable * hashTableBP = Z_ARRVAL_P(oneDimensionArrBP);

    /**
     * Number of elements in oneDimensionArrBP.
     */
    int elementNumB = zend_hash_num_elements(hashTableBP);

    /**
     * Pointer to the host memory for matrix AP.
     */
    float * hostAP = (float *)malloc(elementNumA * sizeof(float));

    /**
     * Pointer to the host memory for matrix BP.
     */
    float * hostBP = (float *)malloc(elementNumB * sizeof(float));

    /**
     * Pointer to the host memory for matrix CP.
     */
    float * hostCP = (float *)malloc(1 * sizeof(float));

    /**
     * Duplicates the content of hashTableAP into the hostAP array.
     *
     * @param hashTableAP Pointer to the HashTable of AP.
     * @param hostAP Pointer to the host memory for matrix AP.
     */
    dup_HashTableTo1DArrOneS(hashTableAP, hostAP);

    /**
     * Duplicates the content of hashTableBP into the hostBP array.
     *
     * @param hashTableBP Pointer to the HashTable of BP.
     * @param hostBP Pointer to the host memory for matrix BP.
     */
    dup_HashTableTo1DArrOneS(hashTableBP, hostBP);

    /**
     * Pointer to the device memory for matrix AP.
     */
    float * deviceAP;

    /**
     * Pointer to the device memory for matrix BP.
     */
    float * deviceBP;

    /**
     * Pointer to the device memory for matrix CP.
     */
    float * deviceCP;

    /**
     * Allocates device memory for matrix AP.
     *
     * @param deviceAP Pointer to the device memory for matrix AP.
     * @param elementNumA Number of elements in matrix AP.
     */
    cudaMalloc((void**)&deviceAP, elementNumA * sizeof(float));

    /**
     * Allocates device memory for matrix BP.
     *
     * @param deviceBP Pointer to the device memory for matrix BP.
     * @param elementNumB Number of elements in matrix BP.
     */
    cudaMalloc((void**)&deviceBP, elementNumB * sizeof(float));

    /**
     * Allocates device memory for matrix CP.
     *
     * @param deviceCP Pointer to the device memory for matrix CP.
     */
    cudaMalloc((void**)&deviceCP, 1 * sizeof(float));

    /**
     * Copies matrix AP from host to device.
     *
     * @param deviceAP Pointer to the device memory for matrix AP.
     * @param hostAP Pointer to the host memory for matrix AP.
     * @param elementNumA Number of elements in matrix AP.
     */
    cudaMemcpy(deviceAP, hostAP, elementNumA * sizeof(float), cudaMemcpyHostToDevice);

    /**
     * Copies matrix BP from host to device.
     *
     * @param deviceBP Pointer to the device memory for matrix BP.
     * @param hostBP Pointer to the host memory for matrix BP.
     * @param elementNumB Number of elements in matrix BP.
     */
    cudaMemcpy(deviceBP, hostBP, elementNumB * sizeof(float), cudaMemcpyHostToDevice);

    /**
     * Pointer to the current object.
     */
    zval *obj = getThis();

    /**
     * Temporary zval pointer.
     */
    zval *tempZVal;
    /**
     * Pointer to store the cublasHandle property.
     */
    zval *cublasHandleP = zend_read_property(BLAS_ce, obj, "cublasHandle", sizeof("cublasHandle") - 1, 1, tempZVal TSRMLS_CC);

    /**
     * Pointer to the cublasHandleStruct obtained from the cublasHandle resource.
     *
     * @param cublasHandleP Pointer to the cublasHandle resource.
     * @param "handleResourceName" Name of the resource type.
     * @param handleResourceNum Resource number of the handle resource.
     * @param cudaHandleStructP Pointer to store the cublasHandleStruct pointer.
     */
    cublasHandleStruct *cudaHandleStructP = (cublasHandleStruct *)zend_fetch_resource(Z_RES_P(cublasHandleP), "handleResourceName", handleResourceNum);

    /**
     * CUDA event for synchronization.
     */
    cudaEvent_t stop;

    /**
     * Creates a CUDA event.
     *
     * @param stop Pointer to the CUDA event.
     */
    cudaEventCreate(&stop);

    /**
     * Calculates the dot product of deviceAP and deviceBP and stores the result in deviceCP.
     *
     * @param cudaHandleStructP Pointer to the cublasHandleStruct.
     * @param elementNumA Number of elements in deviceAP and deviceBP.
     * @param deviceAP Pointer to device memory for matrix AP.
     * @param strideA Stride of deviceAP.
     * @param deviceBP Pointer to device memory for matrix BP.
     * @param strideB Stride of deviceBP.
     * @param deviceCP Pointer to device memory for matrix CP.
     */
    checkCudaResult(cublasSdot(cudaHandleStructP->handle, elementNumA, deviceAP, strideA, deviceBP, strideB, deviceCP));

    /**
     * Synchronizes the CUDA event.
     *
     * @param stop Pointer to the CUDA event.
     */
    cudaEventSynchronize(stop);

    /**
     * Copies the result from deviceCP to hostCP.
     *
     * @param hostCP Pointer to host memory for matrix CP.
     * @param deviceCP Pointer to device memory for matrix CP.
     */
    cudaMemcpy(hostCP, deviceCP, 1 * sizeof(float), cudaMemcpyDeviceToHost);

    /**
     * Sets the return value to the value in hostCP.
     *
     * @param hostCP Pointer to host memory for matrix CP.
     */
    RETVAL_DOUBLE(*hostCP);

    /**
     * Frees the host memory allocated for matrix AP.
     *
     * @param hostAP Pointer to host memory for matrix AP.
     */
    free(hostAP);

    /**
     * Frees the host memory allocated for matrix BP.
     *
     * @param hostBP Pointer to host memory for matrix BP.
     */
    free(hostBP);

    /**
     * Frees the host memory allocated for matrix CP.
     *
     * @param hostCP Pointer to host memory for matrix CP.
     */
    free(hostCP);

    /**
     * Frees the device memory allocated for matrix AP.
     *
     * @param deviceAP Pointer to device memory for matrix AP.
     */
    cudaFree(deviceAP);

    /**
     * Frees the device memory allocated for matrix BP.
     *
     * @param deviceBP Pointer to device memory for matrix BP.
     */
    cudaFree(deviceBP);

    /**
     * Frees the device memory allocated for matrix CP.
     *
     * @param deviceCP Pointer to device memory for matrix CP.
     */
    cudaFree(deviceCP);

    /**
     * Exits the function and returns to the calling code.
     */
    return;

}

/**
 * @brief Information about the arguments for the BLAS::scal method.
 *
 * This macro defines the argument information for the BLAS::scal method.
 *
 * @param alpha The scaling factor.
 * @param oneDimensionArrAP The input array A.
 * @param increase The stride value (optional, default: 1).
 */
ZEND_BEGIN_ARG_INFO_EX(BLAS_scal_ArgInfo, 0, 0, 2)
    ZEND_ARG_INFO(0, alpha)
    ZEND_ARG_ARRAY_INFO(1, oneDimensionArrAP, 0)
    ZEND_ARG_INFO(0, increase)
ZEND_END_ARG_INFO()

/**
 * @brief BLAS::scal method.
 *
 * Scales the elements of a one-dimensional array by a constant factor.
 *
 * @param alpha The scaling factor.
 * @param oneDimensionArrAP The input array A.
 * @param increase The stride value (optional, default: 1).
 * @return An array with scaled elements.
 */
PHP_METHOD(BLAS, scal) {
    /**
     * Pointer to device memory for matrix AP.
     */
    double *deviceAP;

    /**
     * Allocates device memory for matrix AP.
     *
     * @param deviceAP Pointer to device memory for matrix AP.
     * @param elementNumA Number of elements in matrix AP.
     */
    cudaMalloc((void**)&deviceAP, elementNumA * sizeof(double));

    /**
     * Copies matrix AP from host to device.
     *
     * @param deviceAP Pointer to device memory for matrix AP.
     * @param hostAP Pointer to host memory for matrix AP.
     * @param elementNumA Number of elements in matrix AP.
     */
    cudaMemcpy(deviceAP, hostAP, elementNumA * sizeof(double), cudaMemcpyHostToDevice);

    /**
     * Pointer to the current object.
     */
    zval *obj = getThis();

    /**
     * Temporary zval pointer.
     */
    zval *tempZVal;

    /**
     * Retrieves the cublasHandle property from the current object.
     *
     * @param obj Pointer to the current object.
     * @param cublasHandleP Pointer to store the cublasHandle property.
     */
    zval *cublasHandleP = zend_read_property(BLAS_ce, obj, "cublasHandle", sizeof("cublasHandle") - 1, 1, tempZVal TSRMLS_CC);

    /**
     * Retrieves the cublasHandleStruct pointer from the cublasHandle resource.
     *
     * @param cublasHandleP Pointer to the cublasHandle resource.
     * @param "handleResourceName" Name of the resource type.
     * @param handleResourceNum Resource number of the handle resource.
     * @param cudaHandleStructP Pointer to store the cublasHandleStruct pointer.
     */
    cublasHandleStruct *cudaHandleStructP = (cublasHandleStruct *)zend_fetch_resource(Z_RES_P(cublasHandleP), "handleResourceName", handleResourceNum);

    /**
     * CUDA event for synchronization.
     */
    cudaEvent_t stop;

    /**
     * Creates a CUDA event.
     *
     * @param stop Pointer to the CUDA event.
     */
    cudaEventCreate(&stop);

    /**
     * Calls the cublasDscal function to scale matrix AP on the device.
     *
     * @param cudaHandleStructP Pointer to the cublasHandleStruct.
     * @param elementNumA Number of elements in matrix AP.
     * @param alpha Scaling factor.
     * @param deviceAP Pointer to device memory for matrix AP.
     * @param increase Scaling increment.
     */
    checkCudaResult(cublasDscal(cudaHandleStructP->handle, elementNumA, &alpha, deviceAP, increase));

    /**
     * Synchronizes the CUDA event.
     *
     * @param stop Pointer to the CUDA event.
     */
    cudaEventSynchronize(stop);

    /**
     * Copies matrix AP from device to host.
     *
     * @param hostAP Pointer to host memory for matrix AP.
     * @param deviceAP Pointer to device memory for matrix AP.
     * @param elementNumA Number of elements in matrix AP.
     */
    cudaMemcpy(hostAP, deviceAP, elementNumA * sizeof(double), cudaMemcpyDeviceToHost);

    /**
     * Zval for returning the result array.
     */
    zval returnZval;

    /**
     * Initializes the returnZval as an array with a given size.
     *
     * @param returnZval Pointer to the returnZval zval.
     * @param elementNumA Number of elements in the result array.
     */
    array_init_size(&returnZval, elementNumA);

    /**
     * Loops through each element of hostAP and adds it to the returnZval array.
     *
     * @param tempI Loop counter.
     * @param elementNumA Number of elements in hostAP.
     * @param returnZval Pointer to the returnZval zval.
     * @param hostAP Pointer to host memory for matrix AP.
     */
    for (int tempI = 0; tempI < elementNumA; tempI++) {
        add_next_index_double(&returnZval, hostAP[tempI]);
    }

    /**
     * Sets the return value to the returnZval zval.
     *
     * @param returnZval Pointer to the returnZval zval.
     */
    RETVAL_ZVAL(&returnZval, 1, 1);

    /**
     * Frees the host memory allocated for matrix AP.
     *
     * @param hostAP Pointer to host memory for matrix AP.
     */
    free(hostAP);

    /**
     * Frees the device memory allocated for matrix AP.
     *
     * @param deviceAP Pointer to device memory for matrix AP.
     */
    cudaFree(deviceAP);

    /**
     * Exits the function and returns to the calling code.
     */
    return;
}

/**
 * @brief Information about the arguments for the BLAS::scalS method.
 *
 * This macro defines the argument information for the BLAS::scalS method.
 *
 * @param alpha The scaling factor.
 * @param oneDimensionArrAP The input array A.
 * @param increase The stride value (optional, default: 1).
 */
ZEND_BEGIN_ARG_INFO_EX(BLAS_scalS_ArgInfo, 0, 0, 2)
    ZEND_ARG_INFO(0, alpha)
    ZEND_ARG_ARRAY_INFO(1, oneDimensionArrAP, 0)
    ZEND_ARG_INFO(0, increase)
ZEND_END_ARG_INFO()

/**
 * @brief BLAS::scalS method.
 *
 * Scales the elements of a one-dimensional float array by a constant factor.
 *
 * @param alpha The scaling factor.
 * @param oneDimensionArrAP The input array A.
 * @param increase The stride value (optional, default: 1).
 * @return An array with scaled elements.
 */
PHP_METHOD(BLAS, scalS) {
    /**
     * The scalar value to be used in computations.
     */
    float alpha;

    /**
     * Temporary variable for storing alpha as a double.
     */
    double alphaTemp = 1.0;

    /**
     * Pointer to a zval representing a one-dimensional array.
     */
    zval *oneDimensionArrAP = nullptr;

    /**
     * Value to indicate the increment for the computation.
     */
    zend_long increase = 1;

    /**
     * Parses the parameters passed to the function.
     * Expects 2 to 3 parameters:
     * - alphaTemp: The scalar value as a double.
     * - oneDimensionArrAP: The input array.
     * - increase (optional): The increment value.
     */
    ZEND_PARSE_PARAMETERS_START(2, 3)
        Z_PARAM_DOUBLE(alphaTemp)
        Z_PARAM_ARRAY_EX(oneDimensionArrAP, 0, 1)
        Z_PARAM_OPTIONAL
        Z_PARAM_LONG(increase)
    ZEND_PARSE_PARAMETERS_END();

    /**
     * Convert alphaTemp from double to float and assign it to alpha.
     */
    alpha = static_cast<float>(alphaTemp);

    /**
     * Obtains the HashTable from the zval representing the input array.
     */
    HashTable *hashTableAP = Z_ARRVAL_P(oneDimensionArrAP);

    /**
     * Computes the number of elements in the hash table.
     */
    int elementNumA = zend_hash_num_elements(hashTableAP);

    /**
     * Allocates memory for the hostAP array based on the number of elements.
     */
    float *hostAP = (float *)malloc(elementNumA * sizeof(float));

    /**
     * Duplicates the contents of the hash table to the hostAP array.
     */
    dup_HashTableTo1DArrOneS(hashTableAP, hostAP);

    /**
     * Allocates memory for the deviceAP array on the GPU.
     */
    float *deviceAP;
    cudaMalloc((float**)&deviceAP, elementNumA * sizeof(float));

    /**
     * Copies the contents of the hostAP array to the deviceAP array on the GPU.
     */
    cudaMemcpy(deviceAP, hostAP, elementNumA * sizeof(float), cudaMemcpyHostToDevice);

    /**
     * Retrieves the current object.
     */
    zval *obj = getThis();

    /**
     * Temporary zval pointer.
     */
    zval *tempZVal;

    /**
     * Retrieves the cublasHandle property from the current object.
     */
    zval *cublasHandleP = zend_read_property(BLAS_ce, obj, "cublasHandle", sizeof("cublasHandle") - 1, 1, tempZVal TSRMLS_CC);

    /**
     * Converts the cublasHandle resource to the cublasHandleStruct pointer.
     */
    cublasHandleStruct *cudaHandleStructP = (cublasHandleStruct *)zend_fetch_resource(Z_RES_P(cublasHandleP), "handleResourceName", handleResourceNum);

    /**
     * Creates a CUDA event for timing.
     */
    cudaEvent_t stop;
    cudaEventCreate(&stop);

    /**
     * Performs the cublasSscal operation using the cublasHandle and other parameters.
     */
    checkCudaResult(
            cublasSscal(cudaHandleStructP->handle,
                    elementNumA,
                    &alpha, deviceAP, increase
            )
    );

    /**
     * Synchronizes the CUDA event.
     */
    cudaEventSynchronize(stop);

    /**
     * Copies the deviceAP array to the hostAP array.
     */
    cudaMemcpy(hostAP, deviceAP, elementNumA * sizeof(float), cudaMemcpyDeviceToHost);

    /**
     * Initializes a zval for storing the return value.
     */
    zval returnZval;
    array_init_size(&returnZval, elementNumA);

    /**
     * Adds each element of hostAP to the returnZval array.
     */
    for (int tempI = 0; tempI < elementNumA; tempI++) {
        add_next_index_double(&returnZval, hostAP[tempI]);
    }

    /**
     * Sets the return value to the prepared returnZval array.
     */
    RETVAL_ZVAL(&returnZval, 1, 1);

    /**
     * Frees the memory allocated for hostAP.
     */
    free(hostAP);

    /**
     * Frees the memory allocated for deviceAP.
     */
    cudaFree(deviceAP);

    /**
     * Ends the function.
     */
    return;
}

/**
 * @brief Information about the arguments for the BLAS::amax method.
 *
 * This macro defines the argument information for the BLAS::amax method.
 *
 * @param oneDimensionArrAP The input array A.
 * @param increase The stride value (optional, default: 1).
 */
ZEND_BEGIN_ARG_INFO_EX(BLAS_amax_ArgInfo, 0, 0, 2)
    ZEND_ARG_ARRAY_INFO(1, oneDimensionArrAP, 0)
    ZEND_ARG_INFO(0, increase)
ZEND_END_ARG_INFO()

/**
 * @brief BLAS::amax method.
 *
 * Finds the index of the maximum absolute value in a one-dimensional array.
 *
 * @param oneDimensionArrAP The input array A.
 * @param increase The stride value (optional, default: 1).
 * @return The index of the maximum absolute value.
 */
PHP_METHOD(BLAS, amax) {
    /**
     * Pointer to a zval representing a one-dimensional array.
     */
    zval *oneDimensionArrAP = nullptr;

    /**
     * Value to indicate the increment for the computation.
     */
    zend_long increase = 1;

    /**
     * Variable to store the result.
     */
    int result = 0;

    /**
     * Parses the parameters passed to the function.
     * Expects 1 to 2 parameters:
     * - oneDimensionArrAP: The input array.
     * - increase (optional): The increment value.
     */
    ZEND_PARSE_PARAMETERS_START(1, 2)
        Z_PARAM_ARRAY_EX(oneDimensionArrAP, 0, 1)
        Z_PARAM_OPTIONAL
        Z_PARAM_LONG(increase)
    ZEND_PARSE_PARAMETERS_END();

    /**
     * Obtains the HashTable from the zval representing the input array.
     */
    HashTable *hashTableAP = Z_ARRVAL_P(oneDimensionArrAP);

    /**
     * Computes the number of elements in the hash table.
     */
    int elementNumA = zend_hash_num_elements(hashTableAP);

    /**
     * Allocates memory for the hostAP array based on the number of elements.
     */
    double *hostAP = (double *)malloc(elementNumA * sizeof(double));

    /**
     * Duplicates the contents of the hash table to the hostAP array.
     */
    dup_HashTableTo1DArrOne(hashTableAP, hostAP);

    /**
     * Allocates memory for the deviceAP array on the GPU.
     */
    double *deviceAP;
    cudaMalloc((void**)&deviceAP, elementNumA * sizeof(double));

    /**
     * Copies the contents of the hostAP array to the deviceAP array on the GPU.
     */
    cudaMemcpy(deviceAP, hostAP, elementNumA * sizeof(double), cudaMemcpyHostToDevice);

    /**
     * Retrieves the current object.
     */
    zval *obj = getThis();

    /**
     * Temporary zval pointer.
     */
    zval *tempZVal;

    /**
     * Retrieves the cublasHandle property from the current object.
     */
    zval *cublasHandleP = zend_read_property(BLAS_ce, obj, "cublasHandle", sizeof("cublasHandle") - 1, 1, tempZVal TSRMLS_CC);

    /**
     * Converts the cublasHandle resource to the cublasHandleStruct pointer.
     */
    cublasHandleStruct *cudaHandleStructP = (cublasHandleStruct *)zend_fetch_resource(Z_RES_P(cublasHandleP), "handleResourceName", handleResourceNum);

    /**
     * Creates a CUDA event for timing.
     */
    cudaEvent_t stop;
    cudaEventCreate(&stop);

    /**
     * Performs the cublasIdamax operation using the cublasHandle and other parameters.
     */
    checkCudaResult(
            cublasIdamax(cudaHandleStructP->handle,
                    elementNumA,
                    deviceAP, increase, &result
            )
    );

    /**
     * Synchronizes the CUDA event.
     */
    cudaEventSynchronize(stop);

    /**
     * Adjusts the result to be zero-based instead of one-based.
     */
    result = result - 1;

    /**
     * Sets the return value to the result as a zend_long.
     */
    RETVAL_LONG(static_cast<zend_long>(result));

    /**
     * Frees the memory allocated for hostAP.
     */
    free(hostAP);

    /**
     * Frees the memory allocated for deviceAP.
     */
    cudaFree(deviceAP);

    /**
     * Ends the function.
     */
    return;
}

/**
 * @brief Information about the arguments for the BLAS::amaxS method.
 *
 * This macro defines the argument information for the BLAS::amaxS method.
 *
 * @param oneDimensionArrAP The input array A.
 * @param increase The stride value (optional, default: 1).
 */
ZEND_BEGIN_ARG_INFO_EX(BLAS_amaxS_ArgInfo, 0, 0, 2)
    ZEND_ARG_ARRAY_INFO(1, oneDimensionArrAP, 0)
    ZEND_ARG_INFO(0, increase)
ZEND_END_ARG_INFO()

/**
 * @brief BLAS::amaxS method.
 *
 * Finds the index of the maximum absolute value in a one-dimensional float array.
 *
 * @param oneDimensionArrAP The input array A.
 * @param increase The stride value (optional, default: 1).
 * @return The index of the maximum absolute value.
 */
PHP_METHOD(BLAS, amaxS) {
    /**
     * Pointer to a zval representing a one-dimensional array.
     */
    zval *oneDimensionArrAP = nullptr;

    /**
     * Value to indicate the increment for the computation.
     */
    zend_long increase = 1;

    /**
     * Variable to store the result.
     */
    int result = 0;

    /**
     * Parses the parameters passed to the function.
     * Expects 1 to 2 parameters:
     * - oneDimensionArrAP: The input array.
     * - increase (optional): The increment value.
     */
    ZEND_PARSE_PARAMETERS_START(1, 2)
        Z_PARAM_ARRAY_EX(oneDimensionArrAP, 0, 1)
        Z_PARAM_OPTIONAL
        Z_PARAM_LONG(increase)
    ZEND_PARSE_PARAMETERS_END();

    /**
     * Obtains the HashTable from the zval representing the input array.
     */
    HashTable *hashTableAP = Z_ARRVAL_P(oneDimensionArrAP);

    /**
     * Computes the number of elements in the hash table.
     */
    int elementNumA = zend_hash_num_elements(hashTableAP);

    /**
     * Allocates memory for the hostAP array based on the number of elements.
     */
    float *hostAP = (float *)malloc(elementNumA * sizeof(float));

    /**
     * Duplicates the contents of the hash table to the hostAP array.
     */
    dup_HashTableTo1DArrOneS(hashTableAP, hostAP);

    /**
     * Allocates memory for the deviceAP array on the GPU.
     */
    float *deviceAP;
    cudaMalloc((void**)&deviceAP, elementNumA * sizeof(float));

    /**
     * Copies the contents of the hostAP array to the deviceAP array on the GPU.
     */
    cudaMemcpy(deviceAP, hostAP, elementNumA * sizeof(float), cudaMemcpyHostToDevice);

    /**
     * Retrieves the current object.
     */
    zval *obj = getThis();

    /**
     * Temporary zval pointer.
     */
    zval *tempZVal;

    /**
     * Retrieves the cublasHandle property from the current object.
     */
    zval *cublasHandleP = zend_read_property(BLAS_ce, obj, "cublasHandle", sizeof("cublasHandle") - 1, 1, tempZVal TSRMLS_CC);

    /**
     * Converts the cublasHandle resource to the cublasHandleStruct pointer.
     */
    cublasHandleStruct *cudaHandleStructP = (cublasHandleStruct *)zend_fetch_resource(Z_RES_P(cublasHandleP), "handleResourceName", handleResourceNum);

    /**
     * Creates a CUDA event for timing.
     */
    cudaEvent_t stop;
    cudaEventCreate(&stop);

    /**
     * Performs the cublasIsamax operation using the cublasHandle and other parameters.
     */
    checkCudaResult(
            cublasIsamax(cudaHandleStructP->handle,
                    elementNumA,
                    deviceAP, increase, &result
            )
    );

    /**
     * Synchronizes the CUDA event.
     */
    cudaEventSynchronize(stop);

    /**
     * Adjusts the result to be zero-based instead of one-based.
     */
    result = result - 1;

    /**
     * Sets the return value to the result as a zend_long.
     */
    RETVAL_LONG(static_cast<zend_long>(result));

    /**
     * Frees the memory allocated for hostAP.
     */
    free(hostAP);

    /**
     * Frees the memory allocated for deviceAP.
     */
    cudaFree(deviceAP);

    /**
     * Ends the function.
     */
    return;
}

/**
 * @brief Information about the arguments for the BLAS::amin method.
 *
 * This macro defines the argument information for the BLAS::amin method.
 *
 * @param oneDimensionArrAP The input array A.
 * @param increase The stride value (optional, default: 1).
 */
ZEND_BEGIN_ARG_INFO_EX(BLAS_amin_ArgInfo, 0, 0, 2)
    ZEND_ARG_ARRAY_INFO(1, oneDimensionArrAP, 0)
    ZEND_ARG_INFO(0, increase)
ZEND_END_ARG_INFO()

/**
 * @brief BLAS::amin method.
 *
 * Finds the index of the minimum absolute value in a one-dimensional array.
 *
 * @param oneDimensionArrAP The input array A.
 * @param increase The stride value (optional, default: 1).
 * @return The index of the minimum absolute value.
 */
PHP_METHOD(BLAS, amin) {
    /**
     * Pointer to a zval representing a one-dimensional array.
     */
    zval *oneDimensionArrAP = nullptr;

    /**
     * Value to indicate the increment for the computation.
     */
    zend_long increase = 1;

    /**
     * Variable to store the result.
     */
    int result = 0;

    /**
     * Parses the parameters passed to the function.
     * Expects 1 to 2 parameters:
     * - oneDimensionArrAP: The input array.
     * - increase (optional): The increment value.
     */
    ZEND_PARSE_PARAMETERS_START(1, 2)
        Z_PARAM_ARRAY_EX(oneDimensionArrAP, 0, 1)
        Z_PARAM_OPTIONAL
        Z_PARAM_LONG(increase)
    ZEND_PARSE_PARAMETERS_END();

    /**
     * Obtains the HashTable from the zval representing the input array.
     */
    HashTable *hashTableAP = Z_ARRVAL_P(oneDimensionArrAP);

    /**
     * Computes the number of elements in the hash table.
     */
    int elementNumA = zend_hash_num_elements(hashTableAP);

    /**
     * Allocates memory for the hostAP array based on the number of elements.
     */
    double *hostAP = (double *)malloc(elementNumA * sizeof(double));

    /**
     * Duplicates the contents of the hash table to the hostAP array.
     */
    dup_HashTableTo1DArrOne(hashTableAP, hostAP);

    /**
     * Allocates memory for the deviceAP array on the GPU.
     */
    double *deviceAP;
    cudaMalloc((void**)&deviceAP, elementNumA * sizeof(double));

    /**
     * Copies the contents of the hostAP array to the deviceAP array on the GPU.
     */
    cudaMemcpy(deviceAP, hostAP, elementNumA * sizeof(double), cudaMemcpyHostToDevice);

    /**
     * Retrieves the current object.
     */
    zval *obj = getThis();

    /**
     * Temporary zval pointer.
     */
    zval *tempZVal;

    /**
     * Retrieves the cublasHandle property from the current object.
     */
    zval *cublasHandleP = zend_read_property(BLAS_ce, obj, "cublasHandle", sizeof("cublasHandle") - 1, 1, tempZVal TSRMLS_CC);

    /**
     * Converts the cublasHandle resource to the cublasHandleStruct pointer.
     */
    cublasHandleStruct *cudaHandleStructP = (cublasHandleStruct *)zend_fetch_resource(Z_RES_P(cublasHandleP), "handleResourceName", handleResourceNum);

    /**
     * Creates a CUDA event for timing.
     */
    cudaEvent_t stop;
    cudaEventCreate(&stop);

    /**
     * Performs the cublasIdamin operation using the cublasHandle and other parameters.
     */
    checkCudaResult(
            cublasIdamin(cudaHandleStructP->handle,
                    elementNumA,
                    deviceAP, increase, &result
            )
    );

    /**
     * Synchronizes the CUDA event.
     */
    cudaEventSynchronize(stop);

    /**
     * Adjusts the result to be zero-based instead of one-based.
     */
    result = result - 1;

    /**
     * Sets the return value to the result as a zend_long.
     */
    RETVAL_LONG(static_cast<zend_long>(result));

    /**
     * Frees the memory allocated for hostAP.
     */
    free(hostAP);

    /**
     * Frees the memory allocated for deviceAP.
     */
    cudaFree(deviceAP);

    /**
     * Ends the function.
     */
    return;

}

/**
 * @brief Information about the arguments for the BLAS::aminS method.
 *
 * This macro defines the argument information for the BLAS::aminS method.
 *
 * @param oneDimensionArrAP The input array A.
 * @param increase The stride value (optional, default: 1).
 */
ZEND_BEGIN_ARG_INFO_EX(BLAS_aminS_ArgInfo, 0, 0, 2)
    ZEND_ARG_ARRAY_INFO(1, oneDimensionArrAP, 0)
    ZEND_ARG_INFO(0, increase)
ZEND_END_ARG_INFO()

/**
 * @brief BLAS::aminS method.
 *
 * Finds the index of the minimum absolute value in a one-dimensional float array.
 *
 * @param oneDimensionArrAP The input array A.
 * @param increase The stride value (optional, default: 1).
 * @return The index of the minimum absolute value.
 */
PHP_METHOD(BLAS, aminS) {
    /**
     * Pointer to a zval representing a one-dimensional array.
     */
    zval *oneDimensionArrAP = nullptr;

    /**
     * Value to indicate the increment for the computation.
     */
    zend_long increase = 1;

    /**
     * Variable to store the result.
     */
    int result = 0;

    /**
     * Parses the parameters passed to the function.
     * Expects 1 to 2 parameters:
     * - oneDimensionArrAP: The input array.
     * - increase (optional): The increment value.
     */
    ZEND_PARSE_PARAMETERS_START(1, 2)
        Z_PARAM_ARRAY_EX(oneDimensionArrAP, 0, 1)
        Z_PARAM_OPTIONAL
        Z_PARAM_LONG(increase)
    ZEND_PARSE_PARAMETERS_END();

    /**
     * Obtains the HashTable from the zval representing the input array.
     */
    HashTable *hashTableAP = Z_ARRVAL_P(oneDimensionArrAP);

    /**
     * Computes the number of elements in the hash table.
     */
    int elementNumA = zend_hash_num_elements(hashTableAP);

    /**
     * Allocates memory for the hostAP array based on the number of elements.
     */
    float *hostAP = (float *)malloc(elementNumA * sizeof(float));

    /**
     * Duplicates the contents of the hash table to the hostAP array.
     */
    dup_HashTableTo1DArrOneS(hashTableAP, hostAP);

    /**
     * Allocates memory for the deviceAP array on the GPU.
     */
    float *deviceAP;
    cudaMalloc((void**)&deviceAP, elementNumA * sizeof(float));

    /**
     * Copies the contents of the hostAP array to the deviceAP array on the GPU.
     */
    cudaMemcpy(deviceAP, hostAP, elementNumA * sizeof(float), cudaMemcpyHostToDevice);

    /**
     * Retrieves the current object.
     */
    zval *obj = getThis();

    /**
     * Temporary zval pointer.
     */
    zval *tempZVal;

    /**
     * Retrieves the cublasHandle property from the current object.
     */
    zval *cublasHandleP = zend_read_property(BLAS_ce, obj, "cublasHandle", sizeof("cublasHandle") - 1, 1, tempZVal TSRMLS_CC);

    /**
     * Converts the cublasHandle resource to the cublasHandleStruct pointer.
     */
    cublasHandleStruct *cudaHandleStructP = (cublasHandleStruct *)zend_fetch_resource(Z_RES_P(cublasHandleP), "handleResourceName", handleResourceNum);

    /**
     * Creates a CUDA event for timing.
     */
    cudaEvent_t stop;
    cudaEventCreate(&stop);

    /**
     * Performs the cublasIsamin operation using the cublasHandle and other parameters.
     */
    checkCudaResult(
            cublasIsamin(cudaHandleStructP->handle,
                    elementNumA,
                    deviceAP, increase, &result
            )
    );

    /**
     * Synchronizes the CUDA event.
     */
    cudaEventSynchronize(stop);

    /**
     * Adjusts the result to be zero-based instead of one-based.
     */
    result = result - 1;

    /**
     * Sets the return value to the result as a zend_long.
     */
    RETVAL_LONG(static_cast<zend_long>(result));

    /**
     * Frees the memory allocated for hostAP.
     */
    free(hostAP);

    /**
     * Frees the memory allocated for deviceAP.
     */
    cudaFree(deviceAP);

    /**
     * Ends the function.
     */
    return;

}
/**
 * @brief Information about the arguments for the BLAS::axpy method.
 *
 * This macro defines the argument information for the BLAS::axpy method.
 *
 * @param oneDimensionArrAP The input array A.
 * @param oneDimensionArrBP The input array B.
 * @param alpha The scalar value alpha (optional, default: 1.0).
 * @param strideA The stride value for array A (optional, default: 1).
 * @param strideB The stride value for array B (optional, default: 1).
 */
ZEND_BEGIN_ARG_INFO_EX( BLAS_axpy_ArgInfo, 0, 0, 2)
    ZEND_ARG_ARRAY_INFO( 1, oneDimensionArrAP, 0 )
    ZEND_ARG_ARRAY_INFO( 1, oneDimensionArrBP, 0 )
    ZEND_ARG_INFO( 0, alpha )
    ZEND_ARG_INFO( 0, strideA )
    ZEND_ARG_INFO( 0, strideB )
ZEND_END_ARG_INFO()
/**
 * @brief BLAS::axpy method.
 *
 * Performs the operation B = alpha * A + B, where A and B are one-dimensional arrays.
 *
 * @param oneDimensionArrAP The input array A.
 * @param oneDimensionArrBP The input array B.
 * @param alpha The scalar value alpha (optional, default: 1.0).
 * @param strideA The stride value for array A (optional, default: 1).
 * @param strideB The stride value for array B (optional, default: 1).
 * @return The updated array B.
 */
PHP_METHOD(BLAS, axpy){

    /**
     * Pointer to the first input zval representing a one-dimensional array.
     */
    zval *oneDimensionArrAP = NULL;

    /**
     * Pointer to the second input zval representing a one-dimensional array.
     */
    zval *oneDimensionArrBP = NULL;

    /**
     * Scalar value to be used in the computation.
     */
    double alpha = 1.0;

    /**
     * Stride value for the first input array.
     */
    zend_long strideA = 1;

    /**
     * Stride value for the second input array.
     */
    zend_long strideB = 1;

    /**
     * Parses the parameters passed to the function.
     * Expects 2 to 5 parameters:
     * - oneDimensionArrAP: The first input array.
     * - oneDimensionArrBP: The second input array.
     * - alpha (optional): The scalar value.
     * - strideA (optional): The stride value for the first input array.
     * - strideB (optional): The stride value for the second input array.
     */
    ZEND_PARSE_PARAMETERS_START(2, 5)
        Z_PARAM_ARRAY_EX(oneDimensionArrAP, 0, 1)
        Z_PARAM_ARRAY_EX(oneDimensionArrBP, 0, 1)
        Z_PARAM_OPTIONAL
        Z_PARAM_DOUBLE(alpha)
        Z_PARAM_LONG(strideA)
        Z_PARAM_LONG(strideB)
    ZEND_PARSE_PARAMETERS_END();

    /**
     * Obtains the HashTable from the first input zval representing the array.
     */
    HashTable *hashTableAP = Z_ARRVAL_P(oneDimensionArrAP);

    /**
     * Computes the number of elements in the first input array.
     */
    int elementNumA = zend_hash_num_elements(hashTableAP);

    /**
     * Obtains the HashTable from the second input zval representing the array.
     */
    HashTable *hashTableBP = Z_ARRVAL_P(oneDimensionArrBP);

    /**
     * Computes the number of elements in the second input array.
     */
    int elementNumB = zend_hash_num_elements(hashTableBP);

    /**
     * Allocates memory for the hostAP array based on the number of elements in the first input array.
     */
    double *hostAP = (double *)malloc(elementNumA * sizeof(double));

    /**
     * Allocates memory for the hostBP array based on the number of elements in the second input array.
     */
    double *hostBP = (double *)malloc(elementNumB * sizeof(double));

    /**
     * Duplicates the contents of the first input array to the hostAP array.
     */
    dup_HashTableTo1DArrOne(hashTableAP, hostAP);

    /**
     * Duplicates the contents of the second input array to the hostBP array.
     */
    dup_HashTableTo1DArrOne(hashTableBP, hostBP);

    /**
     * Allocates memory for the deviceAP and deviceBP arrays on the GPU.
     */
    double *deviceAP, *deviceBP;
    cudaMalloc((void**)&deviceAP, elementNumA * sizeof(double));
    cudaMalloc((void**)&deviceBP, elementNumB * sizeof(double));

    /**
     * Copies the contents of the hostAP array to the deviceAP array on the GPU.
     */
    cudaMemcpy(deviceAP, hostAP, elementNumA * sizeof(double), cudaMemcpyHostToDevice);

    /**
     * Copies the contents of the hostBP array to the deviceBP array on the GPU.
     */
    cudaMemcpy(deviceBP, hostBP, elementNumB * sizeof(double), cudaMemcpyHostToDevice);

    /**
     * Retrieves the current object.
     */
    zval *obj = getThis();

    /**
     * Temporary zval pointer.
     */
    zval *tempZVal;
    zval *cublasHandleP = zend_read_property(BLAS_ce, obj, "cublasHandle", sizeof("cublasHandle") - 1, 1, tempZVal TSRMLS_CC);
    cublasHandleStruct * cudaHandleStructP = (cublasHandleStruct *)zend_fetch_resource(Z_RES_P(cublasHandleP), "handleResourceName", handleResourceNum);

    cudaEvent_t stop;
    cudaEventCreate(&stop);



    checkCudaResult(
            cublasDaxpy(cudaHandleStructP->handle,
                    elementNumA,
                    &alpha,
                    deviceAP, strideA,
                    deviceBP, strideB
            )
    );

    cudaEventSynchronize(stop);

    cudaMemcpy( hostBP, deviceBP, elementNumB * sizeof(double), cudaMemcpyDeviceToHost );


    zval returnZval; array_init_size( &returnZval, elementNumB );

    for( int tempI = 0; tempI < elementNumB; tempI++ ){
        add_next_index_double( &returnZval, hostBP[ tempI ] );
    }

    RETVAL_ZVAL( &returnZval, 1, 1 );

    free(hostAP);
    free(hostBP);
    cudaFree(deviceAP);
    cudaFree(deviceBP);

    return ;

}
/**
 * \brief Arguments information for the axpyS method.
 */
ZEND_BEGIN_ARG_INFO_EX( BLAS_axpyS_ArgInfo, 0, 0, 2)
    ZEND_ARG_INFO( 0, alpha )
    ZEND_ARG_ARRAY_INFO( 1, oneDimensionArrAP, 0 )
    ZEND_ARG_INFO( 0, increase )
ZEND_END_ARG_INFO()
/**
 * \brief Implements the axpyS method.
 *
 * @param execute_data The execute data structure.
 * @param return_value The return value zval.
 */
PHP_METHOD(BLAS, axpyS){

    /**
     * @var zval* oneDimensionArrAP
     * The input associative array oneDimensionArrAP.
     */
    zval * oneDimensionArrAP = NULL;
    /**
     * @var zval* oneDimensionArrBP
     * The input associative array oneDimensionArrBP.
     */
    zval * oneDimensionArrBP = NULL;
    /**
     * @var float alpha
     * The alpha parameter.
     */
    float alpha;
    /**
     * @var double alphaTemp
     * The temporary alpha parameter.
     */
    double alphaTemp = 1.0;
    /**
     * @var zend_long strideA
     * The strideA parameter.
     */
    zend_long strideA = 1;
    /**
     * @var zend_long strideB
     * The strideB parameter.
     */
    zend_long strideB = 1;

    ZEND_PARSE_PARAMETERS_START(2, 5)
        Z_PARAM_ARRAY_EX( oneDimensionArrAP, 0, 1 )
        Z_PARAM_ARRAY_EX( oneDimensionArrBP, 0, 1 )
        Z_PARAM_OPTIONAL
        Z_PARAM_DOUBLE(alphaTemp)
        Z_PARAM_LONG(strideA)
        Z_PARAM_LONG(strideB)
    ZEND_PARSE_PARAMETERS_END();

    alpha = (float)alphaTemp;

    HashTable * hashTableAP = Z_ARRVAL_P(oneDimensionArrAP);
    int elementNumA = zend_hash_num_elements(hashTableAP);

    HashTable * hashTableBP = Z_ARRVAL_P(oneDimensionArrBP);
    int elementNumB = zend_hash_num_elements(hashTableBP);

    float * hostAP = ( float * )malloc( elementNumA * sizeof(float) );
    float * hostBP = ( float * )malloc( elementNumB * sizeof(float) );

    dup_HashTableTo1DArrOneS( hashTableAP, hostAP );
    dup_HashTableTo1DArrOneS( hashTableBP, hostBP );

    float * deviceAP, * deviceBP;
    cudaMalloc( (void**)&deviceAP, elementNumA * sizeof(float) );
    cudaMalloc( (void**)&deviceBP, elementNumB * sizeof(float) );

    cudaMemcpy( deviceAP, hostAP, elementNumA * sizeof(float), cudaMemcpyHostToDevice );
    cudaMemcpy( deviceBP, hostBP, elementNumB * sizeof(float), cudaMemcpyHostToDevice );

    zval *obj = getThis();
    zval *tempZVal;
    zval *cublasHandleP = zend_read_property(BLAS_ce, obj, "cublasHandle", sizeof("cublasHandle") - 1, 1, tempZVal TSRMLS_CC);
    cublasHandleStruct * cudaHandleStructP = (cublasHandleStruct *)zend_fetch_resource(Z_RES_P(cublasHandleP), "handleResourceName", handleResourceNum);

    cudaEvent_t stop;
    cudaEventCreate(&stop);



    checkCudaResult(
            cublasSaxpy(cudaHandleStructP->handle,
                    elementNumA,
                    &alpha,
                    deviceAP, strideA,
                    deviceBP, strideB
            )
    );

    cudaEventSynchronize(stop);

    cudaMemcpy( hostBP, deviceBP, elementNumB * sizeof(float), cudaMemcpyDeviceToHost );


    zval returnZval; array_init_size( &returnZval, elementNumB );

    for( int tempI = 0; tempI < elementNumB; tempI++ ){
        add_next_index_double( &returnZval, hostBP[ tempI ] );
    }

    RETVAL_ZVAL( &returnZval, 1, 1 );

    free(hostAP);
    free(hostBP);
    cudaFree(deviceAP);
    cudaFree(deviceBP);

    return ;

}

ZEND_BEGIN_ARG_INFO_EX( BLAS_gemv_ArgInfo, 0, 0, 2)
    ZEND_ARG_ARRAY_INFO( 1, tensorhubAP, 0 )
    ZEND_ARG_ARRAY_INFO( 1, oneDimensionArrXP, 0 )
    ZEND_ARG_ARRAY_INFO( 1, oneDimensionArrYP, 0 )
    ZEND_ARG_INFO( 0, alpha )
    ZEND_ARG_INFO( 0, beta )
    ZEND_ARG_INFO( 0, strideX )
    ZEND_ARG_INFO( 0, strideY )
ZEND_END_ARG_INFO()

PHP_METHOD(BLAS, gemv){

    /**
     * @var zval* tensorhubAP
     * The input associative array tensorhubAP.
     */
    zval * tensorhubAP = NULL;
    /**
     * @var zval* oneDimensionArrXP
     * The input associative array oneDimensionArrXP.
     */
    zval * oneDimensionArrXP = NULL;
    /**
     * @var zval* oneDimensionArrYP
     * The input associative array oneDimensionArrYP.
     */
    zval * oneDimensionArrYP = NULL;
    /**
     * @var double alpha
     * The alpha parameter.
     */
    double alpha = 1.0;
    /**
     * @var double beta
     * The beta parameter.
     */
    double beta = 1.0;
    /**
     * @var zend_long strideX
     * The strideX parameter.
     */
    zend_long strideX = 1;
    /**
     * @var zend_long strideY
     * The strideY parameter.
     */
    zend_long strideY = 1;

    /**
     * \brief Parses and validates the parameters passed to the gemv method.
     *
     * @param argc The number of arguments.
     * @param ... The arguments to be parsed.
     */
    ZEND_PARSE_PARAMETERS_START(2, 7)
        Z_PARAM_ARRAY_EX(tensorhubAP, 0, 1)
        Z_PARAM_ARRAY_EX(oneDimensionArrXP, 0, 1)
        Z_PARAM_OPTIONAL
        Z_PARAM_ARRAY_EX(oneDimensionArrYP, 0, 1)
        Z_PARAM_DOUBLE(alpha)
        Z_PARAM_DOUBLE(beta)
        Z_PARAM_LONG(strideX)
        Z_PARAM_LONG(strideY)
    ZEND_PARSE_PARAMETERS_END();

    /**
     * \brief Retrieves the hash table from the tensorhubAP array and calculates the heightA.
     */
    HashTable* hashTableAP = Z_ARRVAL_P(tensorhubAP);
    int heightA = zend_hash_num_elements(hashTableAP);

    // Check if the value of the first element of hashTableAP is an array
    if (Z_TYPE((hashTableAP->arData)->val) != IS_ARRAY) {
        zend_throw_exception_ex(NULL, 2006, duc_getErrorMsg(2006), NULL);
    }

    int widthA = zend_hash_num_elements( Z_ARRVAL( (hashTableAP->arData)->val ) );



    HashTable * hashTableXP = Z_ARRVAL_P(oneDimensionArrXP);
    int elementNumX = zend_hash_num_elements(hashTableXP);

    HashTable * hashTableYP = NULL;
    int elementNumY = 1  + ( widthA - 1 ) * fabs( (int)strideY );

    if( oneDimensionArrYP != NULL ){
        hashTableYP = Z_ARRVAL_P(oneDimensionArrYP);
        int tempNumY = zend_hash_num_elements(hashTableYP);
        if( tempNumY != 0 ){
            elementNumY = tempNumY;
        }
    }

    if( elementNumX < 1 + ( heightA - 1 ) * fabs( (int)strideX ) ){
        zend_throw_exception_ex( NULL, 2007, duc_getErrorMsg(2007), NULL );
    }


    double * hostAP = ( double * )malloc( heightA * widthA * sizeof(double) );
    double * hostXP = ( double * )malloc( elementNumX * sizeof(double) );
    double * hostYP = ( double * )calloc( elementNumY, sizeof(double) );

    dup_HashTableTo1DArr( hashTableAP, hostAP );
    dup_HashTableTo1DArrOne( hashTableXP, hostXP );

    if( oneDimensionArrYP != NULL ){
        dup_HashTableTo1DArrOne( hashTableYP, hostYP );
    }

    double * deviceAP, * deviceXP, * deviceYP;
    cudaMalloc( (void**)&deviceAP, heightA * widthA * sizeof(double) );
    cudaMalloc( (void**)&deviceXP, elementNumX * sizeof(double) );
    cudaMalloc( (void**)&deviceYP, elementNumY * sizeof(double) );

    cudaMemcpy( deviceAP, hostAP, heightA * widthA * sizeof(double), cudaMemcpyHostToDevice );
    cudaMemcpy( deviceXP, hostXP, elementNumX * sizeof(double), cudaMemcpyHostToDevice );
    cudaMemcpy( deviceYP, hostYP, elementNumY * sizeof(double), cudaMemcpyHostToDevice );

    zval *obj = getThis();
    zval *tempZVal;
    zval *cublasHandleP = zend_read_property(BLAS_ce, obj, "cublasHandle", sizeof("cublasHandle") - 1, 1, tempZVal TSRMLS_CC);
    cublasHandleStruct * cudaHandleStructP = (cublasHandleStruct *)zend_fetch_resource(Z_RES_P(cublasHandleP), "handleResourceName", handleResourceNum);

    cudaEvent_t stop;
    cudaEventCreate(&stop);



    checkCudaResult(
            cublasDgemv(cudaHandleStructP->handle,
                    CUBLAS_OP_N,
                    widthA, heightA,
                    &alpha,
                    deviceAP, widthA,
                    deviceXP, (int)strideX,
                    &beta,
                    deviceYP, (int)strideY
            )
    );

    cudaEventSynchronize(stop);

    cudaMemcpy( hostYP, deviceYP, elementNumY * sizeof(double), cudaMemcpyDeviceToHost );


    zval returnZval; array_init_size( &returnZval, elementNumY );

    for( int tempI = 0; tempI < elementNumY; tempI++ ){
        add_next_index_double( &returnZval, hostYP[ tempI ] );
    }

    RETVAL_ZVAL( &returnZval, 1, 1 );

    free(hostAP);
    free(hostXP);
    free(hostYP);
    cudaFree(deviceAP);
    cudaFree(deviceXP);
    cudaFree(deviceYP);

    return ;

}

/**
 * Function: ZEND_BEGIN_ARG_INFO_EX
 *
 * Description of the function.
 *
 * @param This is the first parameter.
 * @param This is the second parameter.
 * @param This is the third parameter.
 *
 * @return This is the return value.
 */
ZEND_BEGIN_ARG_INFO_EX( BLAS_gemvS_ArgInfo, 0, 0, 2)
    ZEND_ARG_ARRAY_INFO( 1, tensorhubAP, 0 )
    ZEND_ARG_ARRAY_INFO( 1, oneDimensionArrXP, 0 )
    ZEND_ARG_ARRAY_INFO( 1, oneDimensionArrYP, 0 )
    ZEND_ARG_INFO( 0, alpha )
    ZEND_ARG_INFO( 0, beta )
    ZEND_ARG_INFO( 0, strideX )
    ZEND_ARG_INFO( 0, strideY )
ZEND_END_ARG_INFO()
/**
 * Method: BLAS->gemvS()
 *
 * Description of the method.
 *
 * @param This is the first parameter.
 * @param This is the second parameter.
 * @param This is the third parameter.
 *
 * @return This is the return value.
 */
PHP_METHOD(BLAS, gemvS){

    zval * tensorhubAP = NULL;
    zval * oneDimensionArrXP = NULL;
    zval * oneDimensionArrYP = NULL;
    float alpha;double alphaTemp = 1.0;
    float beta;double betaTemp = 1.0;
    zend_long strideX = 1;
    zend_long strideY = 1;

    ZEND_PARSE_PARAMETERS_START(2, 7)
        Z_PARAM_ARRAY_EX( tensorhubAP, 0, 1 )
        Z_PARAM_ARRAY_EX( oneDimensionArrXP, 0, 1 )
        Z_PARAM_OPTIONAL
        Z_PARAM_ARRAY_EX( oneDimensionArrYP, 0, 1 )
        Z_PARAM_DOUBLE(alphaTemp)
        Z_PARAM_DOUBLE(betaTemp)
        Z_PARAM_LONG(strideX)
        Z_PARAM_LONG(strideY)
    ZEND_PARSE_PARAMETERS_END();

    alpha = alphaTemp;
    beta = betaTemp;

    HashTable * hashTableAP = Z_ARRVAL_P(tensorhubAP);
    int heightA = zend_hash_num_elements(hashTableAP);

    if( Z_TYPE( (hashTableAP->arData)->val ) != IS_ARRAY ){
        zend_throw_exception_ex(NULL, 2006, duc_getErrorMsg(2006), NULL );
    }

    int widthA = zend_hash_num_elements( Z_ARRVAL( (hashTableAP->arData)->val ) );



    HashTable * hashTableXP = Z_ARRVAL_P(oneDimensionArrXP);
    int elementNumX = zend_hash_num_elements(hashTableXP);

    HashTable * hashTableYP = NULL;
    int elementNumY = 1  + ( widthA - 1 ) * fabs( (int)strideY );

    if( oneDimensionArrYP != NULL ){
        hashTableYP = Z_ARRVAL_P(oneDimensionArrYP);
        int tempNumY = zend_hash_num_elements(hashTableYP);
        if( tempNumY != 0 ){
            elementNumY = tempNumY;
        }
    }

    if( elementNumX < 1 + ( heightA - 1 ) * fabs( (int)strideX ) ){
        zend_throw_exception_ex( NULL, 2007, duc_getErrorMsg(2007), NULL );
    }


    /**
     * Allocates memory for the host array hostAP.
     *
     * @param float* hostAP - Pointer to the host array.
     * @param size_t size - The size of the memory to allocate.
     *
     * @return void
     */
    float* hostAP = (float*)malloc(heightA * widthA * sizeof(float));

    /**
     * Allocates memory for the host array hostXP.
     *
     * @param float* hostXP - Pointer to the host array.
     * @param size_t size - The size of the memory to allocate.
     *
     * @return void
     */
    float* hostXP = (float*)malloc(elementNumX * sizeof(float));

    /**
     * Allocates memory for the host array hostYP and initializes it with zeros.
     *
     * @param float* hostYP - Pointer to the host array.
     * @param size_t num - The number of elements to allocate.
     *
     * @return void
     */
    float* hostYP = (float*)calloc(elementNumY, sizeof(float));

    /**
     * Duplicates the contents of the hash table hashTableAP to the one-dimensional array hostAP.
     *
     * @param HashTable* hashTableAP - Pointer to the hash table.
     * @param float* hostAP - Pointer to the host array.
     *
     * @return void
     */
    dup_HashTableTo1DArrS(hashTableAP, hostAP);

    /**
     * Duplicates the contents of the hash table hashTableXP to the one-dimensional array hostXP.
     *
     * @param HashTable* hashTableXP - Pointer to the hash table.
     * @param float* hostXP - Pointer to the host array.
     *
     * @return void
     */
    dup_HashTableTo1DArrOneS(hashTableXP, hostXP);

    if (oneDimensionArrYP != NULL) {
        /**
         * Duplicates the contents of the hash table hashTableYP to the one-dimensional array hostYP.
         *
         * @param HashTable* hashTableYP - Pointer to the hash table.
         * @param float* hostYP - Pointer to the host array.
         *
         * @return void
         */
        dup_HashTableTo1DArrOneS(hashTableYP, hostYP);
    }

    float* deviceAP, * deviceXP, * deviceYP;

    /**
     * Allocates device memory for the device array deviceAP.
     *
     * @param float** deviceAP - Pointer to the device array.
     * @param size_t size - The size of the memory to allocate.
     *
     * @return void
     */
    cudaMalloc((void**)&deviceAP, heightA * widthA * sizeof(float));

    /**
     * Allocates device memory for the device array deviceXP.
     *
     * @param float** deviceXP - Pointer to the device array.
     * @param size_t size - The size of the memory to allocate.
     *
     * @return void
     */
    cudaMalloc((void**)&deviceXP, elementNumX * sizeof(float));

    /**
     * Allocates device memory for the device array deviceYP.
     *
     * @param float** deviceYP - Pointer to the device array.
     * @param size_t size - The size of the memory to allocate.
     *
     * @return void
     */
    cudaMalloc((void**)&deviceYP, elementNumY * sizeof(float));

    /**
     * Copies data from host memory to device memory for the device array deviceAP.
     *
     * @param void* deviceAP - Pointer to the device memory destination.
     * @param const void* hostAP - Pointer to the host memory source.
     * @param size_t size - The size of the memory to copy.
     * @param cudaMemcpyKind kind - The type of memory copy.
     *
     * @return void
     */
    cudaMemcpy(deviceAP, hostAP, heightA * widthA * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy( deviceXP, hostXP, elementNumX * sizeof(float), cudaMemcpyHostToDevice );
    cudaMemcpy( deviceYP, hostYP, elementNumY * sizeof(float), cudaMemcpyHostToDevice );

    zval *obj = getThis();
    zval *tempZVal;
    zval *cublasHandleP = zend_read_property(BLAS_ce, obj, "cublasHandle", sizeof("cublasHandle") - 1, 1, tempZVal TSRMLS_CC);
    cublasHandleStruct * cudaHandleStructP = (cublasHandleStruct *)zend_fetch_resource(Z_RES_P(cublasHandleP), "handleResourceName", handleResourceNum);

    cudaEvent_t stop;
    cudaEventCreate(&stop);



    checkCudaResult(
            cublasSgemv(cudaHandleStructP->handle,
                    CUBLAS_OP_N,
                    widthA, heightA,
                    &alpha,
                    deviceAP, widthA,
                    deviceXP, (int)strideX,
                    &beta,
                    deviceYP, (int)strideY
            )
    );

    /**
     * Synchronizes a CUDA event.
     *
     * @param cudaEvent_t stop - The CUDA event to synchronize.
     */
    cudaEventSynchronize(stop);

    /**
     * Copies data from device memory to host memory.
     *
     * @param void* hostYP - Pointer to the host memory destination.
     * @param const void* deviceYP - Pointer to the device memory source.
     * @param size_t size - The size of the memory to copy.
     * @param cudaMemcpyKind kind - The type of memory copy.
     */
    cudaMemcpy(hostYP, deviceYP, elementNumY * sizeof(float), cudaMemcpyDeviceToHost);

    /**
     * zval variable for storing the return value.
     */
    zval returnZval;

    /**
     * Initializes an array zval with a given size.
     *
     * @param zval* returnZval - Pointer to the zval variable.
     * @param zend_long size - The size of the array.
     */
    array_init_size(&returnZval, elementNumY);

    for (int tempI = 0; tempI < elementNumY; tempI++) {
        /**
         * Adds the next double value to the return zval array.
         *
         * @param zval* returnZval - Pointer to the return zval array.
         * @param double value - The double value to be added.
         */
        add_next_index_double(&returnZval, hostYP[tempI]);
    }

    /**
     * Returns a zval value as the return value.
     *
     * @param zval* return_value - Pointer to the return value.
     * @param zend_bool dtor - The destructor flag.
     * @param zend_bool separate - The separate flag.
     */
    RETVAL_ZVAL(&returnZval, 1, 1);

    /**
     * Frees the memory allocated for hostAP.
     */
    free(hostAP);

    /**
     * Frees the memory allocated for hostXP.
     */
    free(hostXP);

    /**
     * Frees the memory allocated for hostYP.
     */
    free(hostYP);

    /**
     * Frees the memory allocated for deviceAP.
     */
    cudaFree(deviceAP);

    /**
     * Frees the memory allocated for deviceXP.
     */
    cudaFree(deviceXP);

    /**
     * Frees the memory allocated for deviceYP.
     */
    cudaFree(deviceYP);

    /**
     * Returns from the function.
     */
    return;

}

/**
 * Class: BLAS
 *
 * Description of the class.
 */
const zend_function_entry BLAS_functions[] = {
    /**
     * Method: BLAS->__construct()
     *
     * Description of the method.
     *
     * @param This is the first parameter.
     *
     * @return This is the return value.
     */
    PHP_ME(BLAS, __construct, BLAS_construct_ArgInfo, ZEND_ACC_PUBLIC)

    /**
     * Method: BLAS->getHandle()
     *
     * Description of the method.
     *
     * @param This is the first parameter.
     *
     * @return This is the return value.
     */
    PHP_ME(BLAS, getHandle, BLAS_getHandle_ArgInfo, ZEND_ACC_PUBLIC)

    /**
     * Method: BLAS->setHandle()
     *
     * Description of the method.
     *
     * @param This is the first parameter.
     *
     * @return This is the return value.
     */
    PHP_ME(BLAS, setHandle, BLAS_setHandle_ArgInfo, ZEND_ACC_PUBLIC)
    /**
     * Method: BLAS->multiply()
     *
     * Description of the method.
     *
     * @param This is the first parameter.
     *
     * @return This is the return value.
     */
    PHP_ME(BLAS, multiply, BLAS_multiply_ArgInfo, ZEND_ACC_PUBLIC)

    /**
     * Method: BLAS->multiplyS()
     *
     * Description of the method.
     *
     * @param This is the first parameter.
     *
     * @return This is the return value.
     */
    PHP_ME(BLAS, multiplyS, BLAS_multiplyS_ArgInfo, ZEND_ACC_PUBLIC)

    /**
     * Method: BLAS->dot()
     *
     * Description of the method.
     *
     * @param This is the first parameter.
     *
     * @return This is the return value.
     */
    PHP_ME(BLAS, dot, BLAS_dot_ArgInfo, ZEND_ACC_PUBLIC)

    /**
     * Method: BLAS->dotS()
     *
     * Description of the method.
     *
     * @param This is the first parameter.
     *
     * @return This is the return value.
     */
    PHP_ME(BLAS, dotS, BLAS_dotS_ArgInfo, ZEND_ACC_PUBLIC)

    /**
     * Method: BLAS->scal()
     *
     * Description of the method.
     *
     * @param This is the first parameter.
     *
     * @return This is the return value.
     */
    PHP_ME(BLAS, scal, BLAS_scal_ArgInfo, ZEND_ACC_PUBLIC)

    /**
     * Method: BLAS->scalS()
     *
     * Description of the method.
     *
     * @param This is the first parameter.
     *
     * @return This is the return value.
     */
    PHP_ME(BLAS, scalS, BLAS_scalS_ArgInfo, ZEND_ACC_PUBLIC)

    /**
     * Method: BLAS->amax()
     *
     * Description of the method.
     *
     * @param This is the first parameter.
     *
     * @return This is the return value.
     */
    PHP_ME(BLAS, amax, BLAS_amax_ArgInfo, ZEND_ACC_PUBLIC)

    /**
     * Method: BLAS->amaxS()
     *
     * Description of the method.
     *
     * @param This is the first parameter.
     *
     * @return This is the return value.
     */
    PHP_ME(BLAS, amaxS, BLAS_amaxS_ArgInfo, ZEND_ACC_PUBLIC)

    /**
     * Method: BLAS->amin()
     *
     * Description of the method.
     *
     * @param This is the first parameter.
     *
     * @return This is the return value.
     */
    PHP_ME(BLAS, amin, BLAS_amin_ArgInfo, ZEND_ACC_PUBLIC)

    /**
     * Method: BLAS->aminS()
     *
     * Description of the method.
     *
     * @param This is the first parameter.
     *
     * @return This is the return value.
     */
    PHP_ME(BLAS, aminS, BLAS_aminS_ArgInfo, ZEND_ACC_PUBLIC)

    /**
     * Method: BLAS->axpy()
     *
     * Description of the method.
     *
     * @param This is the first parameter.
     *
     * @return This is the return value.
     */
    PHP_ME(BLAS, axpy, BLAS_axpy_ArgInfo, ZEND_ACC_PUBLIC)

    /**
     * Method: BLAS->axpyS()
     *
     * Description of the method.
     *
     * @param This is the first parameter.
     *
     * @return This is the return value.
     */
    PHP_ME(BLAS, axpyS, BLAS_axpyS_ArgInfo, ZEND_ACC_PUBLIC)

    /**
     * Method: BLAS->gemv()
     *
     * Description of the method.
     *
     * @param This is the first parameter.
     *
     * @return This is the return value.
     */
    PHP_ME(BLAS, gemv, BLAS_gemv_ArgInfo, ZEND_ACC_PUBLIC)

    /**
     * Method: BLAS->gemvS()
     *
     * Description of the method.
     *
     * @param This is the first parameter.
     *
     * @return This is the return value.
     */
    PHP_ME(BLAS, gemvS, BLAS_gemvS_ArgInfo, ZEND_ACC_PUBLIC)
    PHP_FE_END
};

/**
 * PHP_MINIT_FUNCTION for the bs_tensorhub module.
 *
 * @param module The module entry.
 *
 * @return SUCCESS.
 */
PHP_MINIT_FUNCTION(bs_tensorhub)
{
    /**
     * Temporary zend_class_entry variable for the BLAS class.
     */
    zend_class_entry temp_BLAS_ce;

    /**
     * Initializes the namespace class entry for the BLAS class.
     *
     * @param zend_class_entry temp_BLAS_ce - Temporary zend_class_entry variable.
     * @param char* namespace - The namespace of the class.
     * @param char* class_name - The name of the class.
     * @param zend_function_entry* functions - The function entry for the class.
     */
    INIT_NS_CLASS_ENTRY(temp_BLAS_ce, "BS\\tensorhub", "BLAS", BLAS_functions);

    /**
     * Registers the internal class with the given class entry.
     *
     * @param zend_class_entry temp_BLAS_ce - Temporary zend_class_entry variable.
     */
    BLAS_ce = zend_register_internal_class(&temp_BLAS_ce TSRMLS_CC);

    /**
     * Registers the resource destructors for the handle resource.
     *
     * @param zend_resource* handleResource - The handle resource to be destructed.
     * @param zend_resource_type* le_handleResource - Pointer to the handle resource type.
     * @param char* handleResourceName - The name of the handle resource.
     * @param int module_number - The module number.
     */
    handleResourceNum = zend_register_list_destructors_ex(handleResourceDescontructor, NULL, "handleResourceName", module_number);

    /**
     * Declares a property with a null value for the BLAS class.
     *
     * @param zend_class_entry BLAS_ce - zend_class_entry for the BLAS class.
     * @param char* name - The name of the property.
     * @param int name_length - The length of the property name.
     * @param int access_type - The access type of the property.
     */
    zend_declare_property_null(BLAS_ce, "cublasHandle", sizeof("cublasHandle") - 1, ZEND_ACC_PROTECTED TSRMLS_CC);

    /**
     * Temporary zend_class_entry variable for the Util class.
     */
    zend_class_entry temp_Util_ce;

    /**
     * Initializes the namespace class entry for the Util class.
     *
     * @param zend_class_entry temp_Util_ce - Temporary zend_class_entry variable.
     * @param char* namespace - The namespace of the class.
     * @param char* class_name - The name of the class.
     * @param zend_function_entry* functions - The function entry for the class.
     */
    INIT_NS_CLASS_ENTRY(temp_Util_ce, "BS\\tensorhub", "Util", Util_functions);

    /**
     * Registers the internal class with the given class entry.
     *
     * @param zend_class_entry temp_Util_ce - Temporary zend_class_entry variable.
     */
    Util_ce = zend_register_internal_class(&temp_Util_ce TSRMLS_CC);

    /**
     * Temporary zend_class_entry variable for the Math class.
     */
    zend_class_entry temp_Math_ce;

    /**
     * Initializes the namespace class entry for the Math class.
     *
     * @param zend_class_entry temp_Math_ce - Temporary zend_class_entry variable.
     * @param char* namespace - The namespace of the class.
     * @param char* class_name - The name of the class.
     * @param zend_function_entry* functions - The function entry for the class.
     */
    INIT_NS_CLASS_ENTRY(temp_Math_ce, "BS\\tensorhub", "Math", Math_functions);

    /**
     * Registers the internal class with the given class entry.
     *
     * @param zend_class_entry temp_Math_ce - Temporary zend_class_entry variable.
     */
    Math_ce = zend_register_internal_class(&temp_Math_ce TSRMLS_CC);
    /**
     * Declares a property with a long value for the Math class.
     *
     * @param zend_class_entry Math_ce - zend_class_entry for the Math class.
     * @param char* name - The name of the property.
     * @param int name_length - The length of the property name.
     * @param zend_long value - The value of the property.
     * @param int access_type - The access type of the property.
     */
    zend_declare_property_long(Math_ce, "DEVICE_ID", sizeof("DEVICE_ID") - 1, 0, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC TSRMLS_CC);

    /**
     * Returns the value indicating a successful operation.
     *
     * @return int - The success value.
     */
    return SUCCESS;
}

/**
 * Argument information for the bs_tensorhub_test1 function.
 *
 * @param arr The input array.
 */
ZEND_BEGIN_ARG_INFO_EX(arginfo_bs_tensorhub_test1, 0, 0, 1)
    ZEND_ARG_INFO(0, arr)
ZEND_END_ARG_INFO()

/**
 * bs_tensorhub_test1 function.
 *
 * @param execute_data The execution data for the function.
 * @param return_value The return value of the function.
 */
PHP_FUNCTION(bs_tensorhub_test1)
{
    /**
     * Prints a formatted string indicating the loaded and working state of the extension.
     *
     * @param char* extensionName - The name of the extension.
     */
    php_printf("The extension %s is loaded and working!\r\n", "bs_tensorhub");

    /**
     * Pointer to a zval variable.
     */
    zval* arrPointer = NULL;

    ZEND_PARSE_PARAMETERS_START(1, 1)
        /**
         * Parses the parameters and assigns the zval parameter to arrPointer.
         *
         * @param zval* arrPointer - Pointer to the zval parameter.
         */
        Z_PARAM_ZVAL(arrPointer)
    ZEND_PARSE_PARAMETERS_END();

    /**
     * Pointer to the hash table of arrPointer.
     */
    HashTable* hashTablePointer = Z_ARRVAL_P(arrPointer);

    /**
     * Unsigned long integer used as a hash key.
     */
    zend_ulong hash;

    /**
     * Pointer to a string used as a key.
     */
    zend_string* key;

    /**
     * Pointer to a zval value.
     */
    zval* zvalue;

    /**
     * Unsigned long integer used as a hash key for nested hash tables.
     */
    zend_ulong h;

    /**
     * Pointer to a string used as a key for nested hash tables.
     */
    zend_string* k;

    /**
     * Pointer to a zval value for nested hash tables.
     */
    zval* zv;

    /**
     * Pointer to a bucket in the hash table.
     */
    Bucket* p;

    /**
     * The number of elements in the hash table.
     *
     * @var int height - The height of the hash table.
     */
    int height = zend_hash_num_elements(hashTablePointer);

    /**
     * The number of elements in the nested hash table.
     *
     * @var int width - The width of the nested hash table.
     */
    int width = zend_hash_num_elements(Z_ARRVAL((hashTablePointer->arData)->val));

    /**
     * Allocates memory for a host array based on the height and width.
     *
     * @var double* hostAPointer - Pointer to the host array.
     */
    double* hostAPointer = (double*)malloc(height * width * sizeof(double));

    /**
     * Counter for iterating through the host array.
     */
    int count = 0;

    ZEND_HASH_FOREACH_KEY_VAL(hashTablePointer, hash, key, zvalue) {
        /**
         * Iterates over each element in the hash table.
         *
         * @param zend_ulong hash - The hash key.
         * @param zend_string* key - The string key.
         * @param zval* zvalue - The zval value.
         */
        ZEND_HASH_FOREACH_KEY_VAL(Z_ARRVAL_P(zvalue), h, k, zv) {
            /**
             * Iterates over each element in the nested hash table.
             *
             * @param zend_ulong h - The hash key of the nested hash table.
             * @param zend_string* k - The string key of the nested hash table.
             * @param zval* zv - The zval value of the nested hash table.
             */
            hostAPointer[count] = zval_get_double_func(zv);

            count++;
        } ZEND_HASH_FOREACH_END();
    } ZEND_HASH_FOREACH_END();

    /**
     * zval variable for storing the return value.
     */
    zval returnZval;
    array_init_size(&returnZval, height);

    for (int tempI = 0; tempI < height; tempI++) {
        /**
         * zval variable for storing temporary values in the nested array.
         */
        zval tempZval;
        array_init_size(&tempZval, width);

        for (int tempJ = 0; tempJ < width; tempJ++) {
            /**
             * Adds the next double value to the temporary zval array.
             *
             * @param zval* tempZval - Pointer to the temporary zval array.
             * @param double value - The double value to be added.
             */
            add_next_index_double(&tempZval, hostAPointer[tempI * width + tempJ]);
    }

    RETURN_ZVAL(&returnZval, 1, 1);
}

/**
 * PHP_RINIT_FUNCTION for the bs_tensorhub module.
 *
 * @param module The module entry.
 *
 * @return SUCCESS.
 */
PHP_RINIT_FUNCTION(bs_tensorhub)
{
#if defined(ZTS) && defined(COMPILE_DL_BS_tensorhub)
    ZEND_TSRMLS_CACHE_UPDATE();
#endif

    return SUCCESS;
}

/**
 * PHP_MINFO_FUNCTION for the bs_tensorhub module.
 *
 * @param module The module entry.
 */
PHP_MINFO_FUNCTION(bs_tensorhub)
{
    php_info_print_table_start();
    php_info_print_table_header(2, "bs_tensorhub support", "enabled");
    php_info_print_table_end();
}

/**
 * The function entry for bs_tensorhub functions.
 */
static const zend_function_entry bs_tensorhub_functions[] = {
    PHP_FE(bs_tensorhub_test1, arginfo_bs_tensorhub_test1)
    PHP_FE_END
};

/**
 * The module entry for the bs_tensorhub module.
 */
zend_module_entry bs_tensorhub_module_entry = {
    /**
     * Standard module header.
     */
    STANDARD_MODULE_HEADER,

    /**
     * Name of the extension.
     *
     * @var char* "bs_tensorhub"
     */
    "bs_tensorhub",

    /**
     * Function entry for the extension.
     *
     * @var zend_function_entry bs_tensorhub_functions[]
     */
    bs_tensorhub_functions,

    /**
     * PHP module initialization function for the extension.
     *
     * @param int type - The type of initialization.
     * @param int module_number - The module number.
     *
     * @return int - The success value.
     */
    PHP_MINIT(bs_tensorhub),

    /**
     * PHP module shutdown function for the extension.
     *
     * @return int - The success value.
     */
    NULL,

    /**
     * PHP request initialization function for the extension.
     *
     * @param int module_number - The module number.
     *
     * @return int - The success value.
     */
    PHP_RINIT(bs_tensorhub),

    /**
     * PHP request shutdown function for the extension.
     *
     * @return int - The success value.
     */
    NULL,

    /**
     * PHP module information function for the extension.
     *
     * @param zend_module_entry* module - Pointer to the module entry.
     *
     * @return void
     */
    PHP_MINFO(bs_tensorhub),

    /**
     * Version of the extension.
     *
     * @var char* PHP_BS_tensorhub_VERSION
     */
    PHP_BS_tensorhub_VERSION,

    /**
     * Standard module properties.
     */
    STANDARD_MODULE_PROPERTIES
};

#ifdef COMPILE_DL_BS_tensorhub
# ifdef ZTS
ZEND_TSRMLS_CACHE_DEFINE()
# endif
ZEND_GET_MODULE(bs_tensorhub)
#endif
