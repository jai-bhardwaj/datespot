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
#include "math.cuh"
#include "php_bs_math.h"
#include "dev_util_c.h"
#include "dev_util_p.h"

/**
 * The class entry for the Math class.
 */
zend_class_entry* Math_ce;

/**
 * Retrieves the device context.
 *
 * @return The device context.
 */
deviceContextStruct* getDeviceContext()
{
    deviceContextStruct* deviceContextStructP = (deviceContextStruct*)malloc(sizeof(deviceContextStruct));

    zval* tempZVal;
    tempZVal = zend_read_static_property(Math_ce, "DEVICE_ID", sizeof("DEVICE_ID") - 1, 0);

    deviceContextStructP->deviceId = Z_LVAL_P(tempZVal);

    return deviceContextStructP;
}

/**
 * Argument information for the Math::setDeviceId method.
 *
 * @param deviceId The device ID to set.
 */
ZEND_BEGIN_ARG_INFO_EX(Math_setDeviceId_ArgInfo, 0, 0, 1)
    ZEND_ARG_INFO(0, deviceId)
ZEND_END_ARG_INFO()

/**
 * Sets the device ID.
 *
 * @param execute_data The execution data for the method.
 * @param return_value The return value of the method.
 */
PHP_METHOD(Math, setDeviceId)
{
    zend_long deviceId = 0;

    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_LONG(deviceId)
    ZEND_PARSE_PARAMETERS_END();

    zend_update_static_property_long(Math_ce, "DEVICE_ID", sizeof("DEVICE_ID") - 1, deviceId);
}

/**
 * Argument information for the Math::getDeviceId method.
 */
ZEND_BEGIN_ARG_INFO_EX(Math_getDeviceId_ArgInfo, 0, 0, 0)
ZEND_END_ARG_INFO()

/**
 * Retrieves the device ID.
 *
 * @param execute_data The execution data for the method.
 * @param return_value The return value of the method.
 */
PHP_METHOD(Math, getDeviceId)
{
    zval* tempZVal;

    tempZVal = zend_read_static_property(Math_ce, "DEVICE_ID", sizeof("DEVICE_ID") - 1, 0);

    RETURN_ZVAL(tempZVal, 1, 0);
}

/**
 * Argument information for the Math::arrayAdd method.
 *
 * @param arrAP An array parameter representing the input array.
 * @param alpha The value to add to each element.
 */
ZEND_BEGIN_ARG_INFO_EX(Math_arrayAdd_ArgInfo, 0, 0, 2)
    ZEND_ARG_ARRAY_INFO(1, arrAP, 0)
    ZEND_ARG_INFO(0, alpha)
ZEND_END_ARG_INFO()

/**
 * Adds the given value to each element of an input array.
 *
 * @param execute_data The execution data for the method.
 * @param return_value The return value of the method.
 */
PHP_METHOD(Math, arrayAdd)
{
    /**
     * @brief Declaration and initialization of zval pointer for array A.
     */
    zval* arrAP = NULL;

    /**
     * @brief Value for alpha parameter.
     */
    double alpha = 1.0;

    /**
     * @brief Parsing and validating function parameters.
     * 
     * @param 2 minimum required parameters.
     * @param 2 maximum allowed parameters.
     */
    ZEND_PARSE_PARAMETERS_START(2, 2)
        Z_PARAM_ARRAY_EX(arrAP, 0, 1)  // Parse the first parameter as an array (optional).
        Z_PARAM_DOUBLE(alpha)          // Parse the second parameter as a double.
    ZEND_PARSE_PARAMETERS_END();

    /**
     * @brief Conversion of zval array A to a HashTable.
     */
    HashTable* hashTableAP = Z_ARRVAL_P(arrAP);

    /**
     * @brief Declaration of a zval for storing a one-dimensional array.
     */
    zval oneDimensionZval;

    /**
     * @brief Initialize the zval as an empty array.
     */
    array_init(&oneDimensionZval);

    /**
     * @brief Allocation of memory for shape information.
     * 
     * @param 10 is the size of the shape information array.
     * @param sizeof(int) is the size of each element in the array (integer).
     */
    int* shapeInfo = (int*)calloc(10, sizeof(int));

    /**
     * @brief Index used to keep track of the shape information array.
     */
    int shapeInfoIndex = 0;

    /**
     * @brief Duplicate the HashTable of array A to a one-dimensional zval array.
     * 
     * @param hashTableAP is the HashTable of array A.
     * @param oneDimensionZval is the zval array to store the one-dimensional representation of array A.
     * @param shapeInfo is the shape information array.
     * @param shapeInfoIndex is the index of the shape information array.
     */
    dup_hashTableTo1DZval(hashTableAP, oneDimensionZval, shapeInfo, &shapeInfoIndex);

    /**
     * @brief Get the number of elements in the one-dimensional zval array.
     */
    int elementNum = zend_hash_num_elements(Z_ARRVAL(oneDimensionZval));

    /**
     * @brief Allocation of memory for host array A.
     * 
     * @param elementNum is the number of elements in the array.
     * @param sizeof(double) is the size of each element in the array (double).
     */
    double* hostAP = (double*)calloc(elementNum, sizeof(double));

    /**
     * @brief Duplicate the one-dimensional zval array A to a double pointer array.
     * 
     * @param &oneDimensionZval is the zval array representing the one-dimensional array A.
     * @param hostAP is the double pointer array to store the elements of array A.
     */
    dup_oneDimensionZavl
    /**
     * @brief Reshape the one-dimensional double pointer array to a zval array.
     * 
     * @param hostAP is the host array to be reshaped.
     * @param reshapedZval is the zval array to store the reshaped array.
     * @param shapeInfo is the shape information array.
     * @param shapeInfoIndex is the index of the shape information array.
     * @param previousCount is the count of elements processed during reshaping.
     */
    dup_oneDimesnPointerArrReshapeToZval(hostAP, reshapedZval, shapeInfo, &shapeInfoIndex, &previousCount);

    /**
     * @brief Set the reshaped zval array as the return value.
     * 
     * @param &reshapedZval is the zval array to be returned.
     * @param 1 indicates the return value should be destroyed by the engine after returning.
     * @param 1 indicates that the return value should be separated from the original zval array.
     */
    RETVAL_ZVAL(&reshapedZval, 1, 1);

    /**
     * @brief Free the memory allocated for host array A.
     */
    free(hostAP);
}

/**
 * Argument information for the Math::subtractArray method.
 *
 * @param alpha The value to subtract from each element.
 * @param arrAP An array parameter representing the input array.
 */
ZEND_BEGIN_ARG_INFO_EX(Math_subtractArray_ArgInfo, 0, 0, 2)
    ZEND_ARG_INFO(0, alpha)
    ZEND_ARG_ARRAY_INFO(1, arrAP, 0)
ZEND_END_ARG_INFO()

/**
 * Subtracts the given value from each element of an input array.
 *
 * @param execute_data The execution data for the method.
 * @param return_value The return value of the method.
 */
PHP_METHOD(Math, subtractArray)
{
    /**
     * @brief Value for alpha parameter.
     */
    double alpha = 1.0;

    /**
     * @brief Declaration and initialization of zval pointer for array A.
     */
    zval* arrAP = NULL;

    /**
     * @brief Parsing and validating function parameters.
     * 
     * @param 2 minimum required parameters.
     * @param 2 maximum allowed parameters.
     */
    ZEND_PARSE_PARAMETERS_START(2, 2)
        Z_PARAM_DOUBLE(alpha)
        Z_PARAM_ARRAY_EX(arrAP, 0, 1)
    ZEND_PARSE_PARAMETERS_END();

    /**
     * @brief Conversion of zval array A to a HashTable.
     */
    HashTable* hashTableAP = Z_ARRVAL_P(arrAP);

    /**
     * @brief Declaration of a zval for storing a one-dimensional array.
     */
    zval oneDimensionZval;

    /**
     * @brief Initialize the zval as an empty array.
     */
    array_init(&oneDimensionZval);

    /**
     * @brief Allocation of memory for shape information.
     * 
     * @param 10 is the size of the shape information array.
     * @param sizeof(int) is the size of each element in the array (integer).
     */
    int* shapeInfo = (int*)calloc(10, sizeof(int));

    /**
     * @brief Index used to keep track of the shape information array.
     */
    int shapeInfoIndex = 0;

    /**
     * @brief Duplicate the HashTable of array A to a one-dimensional zval array.
     * 
     * @param hashTableAP is the HashTable of array A.
     * @param oneDimensionZval is the zval array to store the one-dimensional representation of array A.
     * @param shapeInfo is the shape information array.
     * @param shapeInfoIndex is the index of the shape information array.
     */
    dup_hashTableTo1DZval(hashTableAP, oneDimensionZval, shapeInfo, &shapeInfoIndex);

    /**
     * @brief Get the number of elements in the one-dimensional zval array.
     */
    int elementNum = zend_hash_num_elements(Z_ARRVAL(oneDimensionZval));

    /**
     * @brief Allocation of memory for host array A.
     * 
     * @param elementNum is the number of elements in the array.
     * @param sizeof(double) is the size of each element in the array (double).
     */
    double* hostAP = (double*)calloc(elementNum, sizeof(double));

    /**
     * @brief Duplicate the one-dimensional zval array A to a double pointer array.
     * 
     * @param &oneDimensionZval is the zval array representing the one-dimensional array A.
     * @param hostAP is the double pointer array to store the elements of array A.
     */
    dup_oneDimensionZavlToPointerArr(&oneDimensionZval, hostAP);

    /**
     * @brief Perform the subtraction operation on host array A.
     * 
     * @param getDeviceContext() is a function to get the device context.
     * @param alpha is the value to subtract from the array elements.
     * @param hostAP is the host array A.
     * @param elementNum is the number of elements in the array.
     */
    subtractArray(getDeviceContext(), alpha, hostAP, elementNum);

    /**
     * @brief Declaration of a zval for storing the reshaped array.
     */
    zval reshapedZval;

    /**
     * @brief Initialize the zval as an empty array.
     */
    array_init(&reshapedZval);

    /**
     * @brief Reset the shape information index.
     */
    shapeInfoIndex = 0;

    /**
     * @brief Counter to keep track of the previous count during reshaping.
     */
    int previousCount = 0;
    /**
     * @brief Reshape the one-dimensional double pointer array to a zval array.
     * 
     * @param hostAP is the host array to be reshaped.
     * @param reshapedZval is the zval array to store the reshaped array.
     * @param shapeInfo is the shape information array.
     * @param shapeInfoIndex is the index of the shape information array.
     * @param previousCount is the count of elements processed during reshaping.
     */
    dup_oneDimesnPointerArrReshapeToZval(hostAP, reshapedZval, shapeInfo, &shapeInfoIndex, &previousCount);

    /**
     * @brief Set the reshaped zval array as the return value.
     * 
     * @param &reshapedZval is the zval array to be returned.
     * @param 1 indicates the return value should be destroyed by the engine after returning.
     * @param 1 indicates that the return value should be separated from the original zval array.
     */
    RETVAL_ZVAL(&reshapedZval, 1, 1);

    /**
     * @brief Free the memory allocated for host array A.
     */
    free(hostAP);
}

/**
 * Argument information for the Math::arrayMultiply method.
 *
 * @param arrAP An array parameter representing the input array.
 * @param alpha The multiplier value.
 */
ZEND_BEGIN_ARG_INFO_EX(Math_arrayMultiply_ArgInfo, 0, 0, 2)
    ZEND_ARG_ARRAY_INFO(1, arrAP, 0)
    ZEND_ARG_INFO(0, alpha)
ZEND_END_ARG_INFO()

/**
 * Multiplies each element of an input array by the given multiplier.
 *
 * @param execute_data The execution data for the method.
 * @param return_value The return value of the method.
 */
PHP_METHOD(Math, arrayMultiply)
{
    /**
     * @brief Declaration and initialization of zval pointer for array A.
     */
    zval* arrAP = NULL;

    /**
     * @brief Value for alpha parameter.
     */
    double alpha = 1.0;

    /**
     * @brief Parsing and validating function parameters.
     * 
     * @param 2 minimum required parameters.
     * @param 2 maximum allowed parameters.
     */
    ZEND_PARSE_PARAMETERS_START(2, 2)
        Z_PARAM_ARRAY_EX(arrAP, 0, 1)
        Z_PARAM_DOUBLE(alpha)
    ZEND_PARSE_PARAMETERS_END();

    /**
     * @brief Conversion of zval array A to a HashTable.
     */
    HashTable* hashTableAP = Z_ARRVAL_P(arrAP);

    /**
     * @brief Declaration of a zval for storing a one-dimensional array.
     */
    zval oneDimensionZval;

    /**
     * @brief Initialize the zval as an empty array.
     */
    array_init(&oneDimensionZval);

    /**
     * @brief Allocation of memory for shape information.
     * 
     * @param 10 is the size of the shape information array.
     * @param sizeof(int) is the size of each element in the array (integer).
     */
    int* shapeInfo = (int*)calloc(10, sizeof(int));

    /**
     * @brief Index used to keep track of the shape information array.
     */
    int shapeInfoIndex = 0;

    /**
     * @brief Duplicate the HashTable of array A to a one-dimensional zval array.
     * 
     * @param hashTableAP is the HashTable of array A.
     * @param oneDimensionZval is the zval array to store the one-dimensional representation of array A.
     * @param shapeInfo is the shape information array.
     * @param shapeInfoIndex is the index of the shape information array.
     */
    dup_hashTableTo1DZval(hashTableAP, oneDimensionZval, shapeInfo, &shapeInfoIndex);

    /**
     * @brief Get the number of elements in the one-dimensional zval array.
     */
    int elementNum = zend_hash_num_elements(Z_ARRVAL(oneDimensionZval));

    /**
     * @brief Allocation of memory for host array A.
     * 
     * @param elementNum is the number of elements in the array.
     * @param sizeof(double) is the size of each element in the array (double).
     */
    double* hostAP = (double*)calloc(elementNum, sizeof(double));

    /**
     * @brief Duplicate the one-dimensional zval array A to a double pointer array.
     * 
     * @param &oneDimensionZval is the zval array representing the one-dimensional array A.
     * @param hostAP is the double pointer array to store the elements of array A.
     */
    dup_oneDimensionZavlToPointerArr(&oneDimensionZval, hostAP);

    /**
     * @brief Perform the multiplication operation on host array A.
     * 
     * @param getDeviceContext() is a function to get the device context.
     * @param hostAP is the host array A.
     * @param elementNum is the number of elements in the array.
     * @param alpha is the value by which the array elements will be multiplied.
     */
    arrayMultiply(getDeviceContext(), hostAP, elementNum, alpha);

    /**
     * @brief Declaration of a zval for storing the reshaped array.
     */
    zval reshapedZval;

    /**
     * @brief Initialize the zval as an empty array.
     */
    array_init(&reshapedZval);

    /**
     * @brief Reset the shape information index.
     */
    shapeInfoIndex = 0;

    /**
     * @brief Counter to keep track of the previous count during reshaping.
     */
    int previousCount = 0;
    /**
     * @brief Reshape the one-dimensional double pointer array to a zval array.
     * 
     * @param hostAP is the host array to be reshaped.
     * @param reshapedZval is the zval array to store the reshaped array.
     * @param shapeInfo is the shape information array.
     * @param shapeInfoIndex is the index of the shape information array.
     * @param previousCount is the count of elements processed during reshaping.
     */
    dup_oneDimesnPointerArrReshapeToZval(hostAP, reshapedZval, shapeInfo, &shapeInfoIndex, &previousCount);

    /**
     * @brief Set the reshaped zval array as the return value.
     * 
     * @param &reshapedZval is the zval array to be returned.
     * @param 1 indicates the return value should be destroyed by the engine after returning.
     * @param 1 indicates that the return value should be separated from the original zval array.
     */
    RETVAL_ZVAL(&reshapedZval, 1, 1);

    /**
     * @brief Free the memory allocated for host array A.
     */
    free(hostAP);
}

/**
 * Argument information for the Math::divideArray method.
 *
 * @param alpha The divisor value.
 * @param arrAP An array parameter representing the input array.
 */
ZEND_BEGIN_ARG_INFO_EX(Math_divideArray_ArgInfo, 0, 0, 2)
    ZEND_ARG_INFO(0, alpha)
    ZEND_ARG_ARRAY_INFO(1, arrAP, 0)
ZEND_END_ARG_INFO()

/**
 * Divides each element of an input array by the given divisor.
 *
 * @param execute_data The execution data for the method.
 * @param return_value The return value of the method.
 */
PHP_METHOD(Math, divideArray)
{
    /**
     * @brief Value for alpha parameter.
     */
    double alpha = 1.0;

    /**
     * @brief Declaration and initialization of zval pointer for array A.
     */
    zval* arrAP = NULL;

    /**
     * @brief Parsing and validating function parameters.
     * 
     * @param 2 minimum required parameters.
     * @param 2 maximum allowed parameters.
     */
    ZEND_PARSE_PARAMETERS_START(2, 2)
        Z_PARAM_DOUBLE(alpha)
        Z_PARAM_ARRAY_EX(arrAP, 0, 1)
    ZEND_PARSE_PARAMETERS_END();

    /**
     * @brief Conversion of zval array A to a HashTable.
     */
    HashTable* hashTableAP = Z_ARRVAL_P(arrAP);

    /**
     * @brief Declaration of a zval for storing a one-dimensional array.
     */
    zval oneDimensionZval;

    /**
     * @brief Initialize the zval as an empty array.
     */
    array_init(&oneDimensionZval);

    /**
     * @brief Allocation of memory for shape information.
     * 
     * @param 10 is the size of the shape information array.
     * @param sizeof(int) is the size of each element in the array (integer).
     */
    int* shapeInfo = (int*)calloc(10, sizeof(int));

    /**
     * @brief Index used to keep track of the shape information array.
     */
    int shapeInfoIndex = 0;

    /**
     * @brief Duplicate the HashTable of array A to a one-dimensional zval array.
     * 
     * @param hashTableAP is the HashTable of array A.
     * @param oneDimensionZval is the zval array to store the one-dimensional representation of array A.
     * @param shapeInfo is the shape information array.
     * @param shapeInfoIndex is the index of the shape information array.
     */
    dup_hashTableTo1DZval(hashTableAP, oneDimensionZval, shapeInfo, &shapeInfoIndex);

    /**
     * @brief Get the number of elements in the one-dimensional zval array.
     */
    int elementNum = zend_hash_num_elements(Z_ARRVAL(oneDimensionZval));

    /**
     * @brief Allocation of memory for host array A.
     * 
     * @param elementNum is the number of elements in the array.
     * @param sizeof(double) is the size of each element in the array (double).
     */
    double* hostAP = (double*)calloc(elementNum, sizeof(double));

    /**
     * @brief Duplicate the one-dimensional zval array A to a double pointer array.
     * 
     * @param &oneDimensionZval is the zval array representing the one-dimensional array A.
     * @param hostAP is the double pointer array to store the elements of array A.
     */
    dup_oneDimensionZavlToPointerArr(&oneDimensionZval, hostAP);

    /**
     * @brief Perform the division operation on host array A.
     * 
     * @param getDeviceContext() is a function to get the device context.
     * @param alpha is the value by which the array elements will be divided.
     * @param hostAP is the host array A.
     * @param elementNum is the number of elements in the array.
     */
    divideArray(getDeviceContext(), alpha, hostAP, elementNum);

    /**
     * @brief Declaration of a zval for storing the reshaped array.
     */
    zval reshapedZval;

    /**
     * @brief Initialize the zval as an empty array.
     */
    array_init(&reshapedZval);

    /**
     * @brief Reset the shape information index.
     */
    shapeInfoIndex = 0;

    /**
     * @brief Counter to keep track of the previous count during reshaping.
     */
    int previousCount = 0;
    /**
     * @brief Reshape the one-dimensional double pointer array to a zval array.
     * 
     * @param hostAP is the host array to be reshaped.
     * @param reshapedZval is the zval array to store the reshaped array.
     * @param shapeInfo is the shape information array.
     * @param shapeInfoIndex is the index of the shape information array.
     * @param previousCount is the count of elements processed during reshaping.
     */
    dup_oneDimesnPointerArrReshapeToZval(hostAP, reshapedZval, shapeInfo, &shapeInfoIndex, &previousCount);

    /**
     * @brief Set the reshaped zval array as the return value.
     * 
     * @param &reshapedZval is the zval array to be returned.
     * @param 1 indicates the return value should be destroyed by the engine after returning.
     * @param 1 indicates that the return value should be separated from the original zval array.
     */
    RETVAL_ZVAL(&reshapedZval, 1, 1);

    /**
     * @brief Free the memory allocated for host array A.
     */
    free(hostAP);
}

/**
 * Argument information for the Math::arrayPower method.
 *
 * @param arrAP An array parameter representing the input array.
 * @param alpha The exponent value.
 */
ZEND_BEGIN_ARG_INFO_EX(Math_arrayPower_ArgInfo, 0, 0, 2)
    ZEND_ARG_ARRAY_INFO(1, arrAP, 0)
    ZEND_ARG_INFO(0, alpha)
ZEND_END_ARG_INFO()

/**
 * Raises each element of an input array to the power of the given exponent.
 *
 * @param execute_data The execution data for the method.
 * @param return_value The return value of the method.
 */
PHP_METHOD(Math, arrayPower)
{
    /**
     * @brief Declaration and initialization of zval pointer for array A.
     */
    zval* arrAP = NULL;

    /**
     * @brief Value for alpha parameter.
     */
    double alpha = 1.0;

    /**
     * @brief Parsing and validating function parameters.
     * 
     * @param 2 minimum required parameters.
     * @param 2 maximum allowed parameters.
     */
    ZEND_PARSE_PARAMETERS_START(2, 2)
        Z_PARAM_ARRAY_EX(arrAP, 0, 1)
        Z_PARAM_DOUBLE(alpha)
    ZEND_PARSE_PARAMETERS_END();

    /**
     * @brief Conversion of zval array A to a HashTable.
     */
    HashTable* hashTableAP = Z_ARRVAL_P(arrAP);

    /**
     * @brief Declaration of a zval for storing a one-dimensional array.
     */
    zval oneDimensionZval;

    /**
     * @brief Initialize the zval as an empty array.
     */
    array_init(&oneDimensionZval);

    /**
     * @brief Allocation of memory for shape information.
     * 
     * @param 10 is the size of the shape information array.
     * @param sizeof(int) is the size of each element in the array (integer).
     */
    int* shapeInfo = (int*)calloc(10, sizeof(int));

    /**
     * @brief Index used to keep track of the shape information array.
     */
    int shapeInfoIndex = 0;

    /**
     * @brief Duplicate the HashTable of array A to a one-dimensional zval array.
     * 
     * @param hashTableAP is the HashTable of array A.
     * @param oneDimensionZval is the zval array to store the one-dimensional representation of array A.
     * @param shapeInfo is the shape information array.
     * @param shapeInfoIndex is the index of the shape information array.
     */
    dup_hashTableTo1DZval(hashTableAP, oneDimensionZval, shapeInfo, &shapeInfoIndex);

    /**
     * @brief Get the number of elements in the one-dimensional zval array.
     */
    int elementNum = zend_hash_num_elements(Z_ARRVAL(oneDimensionZval));

    /**
     * @brief Allocation of memory for host array A.
     * 
     * @param elementNum is the number of elements in the array.
     * @param sizeof(double) is the size of each element in the array (double).
     */
    double* hostAP = (double*)calloc(elementNum, sizeof(double));

    /**
     * @brief Duplicate the one-dimensional zval array A to a double pointer array.
     * 
     * @param &oneDimensionZval is the zval array representing the one-dimensional array A.
     * @param hostAP is the double pointer array to store the elements of array A.
     */
    dup_oneDimensionZavlToPointerArr(&oneDimensionZval, hostAP);

    /**
     * @brief Perform the power operation on host array A.
     * 
     * @param getDeviceContext() is a function to get the device context.
     * @param hostAP is the host array A.
     * @param elementNum is the number of elements in the array.
     * @param alpha is the power value to apply to the array elements.
     */
    arrayPower(getDeviceContext(), hostAP, elementNum, alpha);

    /**
     * @brief Declaration of a zval for storing the reshaped array.
     */
    zval reshapedZval;

    /**
     * @brief Initialize the zval as an empty array.
     */
    array_init(&reshapedZval);

    /**
     * @brief Reset the shape information index.
     */
    shapeInfoIndex = 0;

    /**
     * @brief Counter to keep track of the previous count during reshaping.
     */
    int previousCount = 0;
    /**
     * @brief Reshape the one-dimensional double pointer array to a zval array.
     * 
     * @param hostAP is the host array to be reshaped.
     * @param reshapedZval is the zval array to store the reshaped array.
     * @param shapeInfo is the shape information array.
     * @param shapeInfoIndex is the index of the shape information array.
     * @param previousCount is the count of elements processed during reshaping.
     */
    dup_oneDimesnPointerArrReshapeToZval(hostAP, reshapedZval, shapeInfo, &shapeInfoIndex, &previousCount);

    /**
     * @brief Set the reshaped zval array as the return value.
     * 
     * @param &reshapedZval is the zval array to be returned.
     * @param 1 indicates the return value should be destroyed by the engine after returning.
     * @param 1 indicates that the return value should be separated from the original zval array.
     */
    RETVAL_ZVAL(&reshapedZval, 1, 1);

    /**
     * @brief Free the memory allocated for host array A.
     */
    free(hostAP);
}

/**
 * Argument information for the Math::arraySquareRoot method.
 *
 * @param arrAP An array parameter representing the input array.
 */
ZEND_BEGIN_ARG_INFO_EX(Math_arraySquareRoot_ArgInfo, 0, 0, 1)
    ZEND_ARG_ARRAY_INFO(1, arrAP, 0)
ZEND_END_ARG_INFO()

/**
 * Calculates the square root of an input array.
 *
 * @param execute_data The execution data for the method.
 * @param return_value The return value of the method.
 */
PHP_METHOD(Math, arraySquareRoot)
{
    /**
     * @brief Declaration and initialization of zval pointer for array A.
     */
    zval* arrAP = NULL;

    /**
     * @brief Parsing and validating function parameters.
     * 
     * @param 1 minimum required parameter.
     * @param 1 maximum allowed parameter.
     */
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_ARRAY_EX(arrAP, 0, 1)
    ZEND_PARSE_PARAMETERS_END();

    /**
     * @brief Conversion of zval array A to a HashTable.
     */
    HashTable* hashTableAP = Z_ARRVAL_P(arrAP);

    /**
     * @brief Declaration of a zval for storing a one-dimensional array.
     */
    zval oneDimensionZval;

    /**
     * @brief Initialize the zval as an empty array.
     */
    array_init(&oneDimensionZval);

    /**
     * @brief Allocation of memory for shape information.
     * 
     * @param 10 is the size of the shape information array.
     * @param sizeof(int) is the size of each element in the array (integer).
     */
    int* shapeInfo = (int*)calloc(10, sizeof(int));

    /**
     * @brief Index used to keep track of the shape information array.
     */
    int shapeInfoIndex = 0;

    /**
     * @brief Duplicate the HashTable of array A to a one-dimensional zval array.
     * 
     * @param hashTableAP is the HashTable of array A.
     * @param oneDimensionZval is the zval array to store the one-dimensional representation of array A.
     * @param shapeInfo is the shape information array.
     * @param shapeInfoIndex is the index of the shape information array.
     */
    dup_hashTableTo1DZval(hashTableAP, oneDimensionZval, shapeInfo, &shapeInfoIndex);

    /**
     * @brief Get the number of elements in the one-dimensional zval array.
     */
    int elementNum = zend_hash_num_elements(Z_ARRVAL(oneDimensionZval));

    /**
     * @brief Allocation of memory for host array A.
     * 
     * @param elementNum is the number of elements in the array.
     * @param sizeof(double) is the size of each element in the array (double).
     */
    double* hostAP = (double*)calloc(elementNum, sizeof(double));

    /**
     * @brief Duplicate the one-dimensional zval array A to a double pointer array.
     * 
     * @param &oneDimensionZval is the zval array representing the one-dimensional array A.
     * @param hostAP is the double pointer array to store the elements of array A.
     */
    dup_oneDimensionZavlToPointerArr(&oneDimensionZval, hostAP);

    /**
     * @brief Perform the square root operation on host array A.
     * 
     * @param getDeviceContext() is a function to get the device context.
     * @param hostAP is the host array A.
     * @param elementNum is the number of elements in the array.
     */
    arraySquareRoot(getDeviceContext(), hostAP, elementNum);

    /**
     * @brief Declaration of a zval for storing the reshaped array.
     */
    zval reshapedZval;

    /**
     * @brief Initialize the zval as an empty array.
     */
    array_init(&reshapedZval);

    /**
     * @brief Reset the shape information index.
     */
    shapeInfoIndex = 0;

    /**
     * @brief Counter to keep track of the previous count during reshaping.
     */
    int previousCount = 0;
    /**
     * @brief Reshape the one-dimensional double pointer array to a zval array.
     * 
     * @param hostAP is the host array to be reshaped.
     * @param reshapedZval is the zval array to store the reshaped array.
     * @param shapeInfo is the shape information array.
     * @param shapeInfoIndex is the index of the shape information array.
     * @param previousCount is the count of elements processed during reshaping.
     */
    dup_oneDimesnPointerArrReshapeToZval(hostAP, reshapedZval, shapeInfo, &shapeInfoIndex, &previousCount);

    /**
     * @brief Set the reshaped zval array as the return value.
     * 
     * @param &reshapedZval is the zval array to be returned.
     * @param 1 indicates the return value should be destroyed by the engine after returning.
     * @param 1 indicates that the return value should be separated from the original zval array.
     */
    RETVAL_ZVAL(&reshapedZval, 1, 1);

    /**
     * @brief Free the memory allocated for host array A.
     */
    free(hostAP);
}

/**
 * Argument information for the Math::arrayCubeRoot method.
 *
 * @param arrAP An array parameter representing the input array.
 */
ZEND_BEGIN_ARG_INFO_EX(Math_arrayCubeRoot_ArgInfo, 0, 0, 1)
    ZEND_ARG_ARRAY_INFO(1, arrAP, 0)
ZEND_END_ARG_INFO()

/**
 * Calculates the cube root of an input array.
 *
 * @param execute_data The execution data for the method.
 * @param return_value The return value of the method.
 */
PHP_METHOD(Math, arrayCubeRoot)
{
    /**
     * @brief Declaration and initialization of zval pointer for array A.
     */
    zval* arrAP = NULL;

    /**
     * @brief Parsing and validating function parameters.
     * 
     * @param 1 minimum required parameter.
     * @param 1 maximum allowed parameter.
     */
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_ARRAY_EX(arrAP, 0, 1)
    ZEND_PARSE_PARAMETERS_END();

    /**
     * @brief Conversion of zval array A to a HashTable.
     */
    HashTable* hashTableAP = Z_ARRVAL_P(arrAP);

    /**
     * @brief Declaration of a zval for storing a one-dimensional array.
     */
    zval oneDimensionZval;

    /**
     * @brief Initialize the zval as an empty array.
     */
    array_init(&oneDimensionZval);

    /**
     * @brief Allocation of memory for shape information.
     * 
     * @param 10 is the size of the shape information array.
     * @param sizeof(int) is the size of each element in the array (integer).
     */
    int* shapeInfo = (int*)calloc(10, sizeof(int));

    /**
     * @brief Index used to keep track of the shape information array.
     */
    int shapeInfoIndex = 0;

    /**
     * @brief Duplicate the HashTable of array A to a one-dimensional zval array.
     * 
     * @param hashTableAP is the HashTable of array A.
     * @param oneDimensionZval is the zval array to store the one-dimensional representation of array A.
     * @param shapeInfo is the shape information array.
     * @param shapeInfoIndex is the index of the shape information array.
     */
    dup_hashTableTo1DZval(hashTableAP, oneDimensionZval, shapeInfo, &shapeInfoIndex);

    /**
     * @brief Get the number of elements in the one-dimensional zval array.
     */
    int elementNum = zend_hash_num_elements(Z_ARRVAL(oneDimensionZval));

    /**
     * @brief Allocation of memory for host array A.
     * 
     * @param elementNum is the number of elements in the array.
     * @param sizeof(double) is the size of each element in the array (double).
     */
    double* hostAP = (double*)calloc(elementNum, sizeof(double));

    /**
     * @brief Duplicate the one-dimensional zval array A to a double pointer array.
     * 
     * @param &oneDimensionZval is the zval array representing the one-dimensional array A.
     * @param hostAP is the double pointer array to store the elements of array A.
     */
    dup_oneDimensionZavlToPointerArr(&oneDimensionZval, hostAP);

    /**
     * @brief Perform the cube root operation on host array A.
     * 
     * @param getDeviceContext() is a function to get the device context.
     * @param hostAP is the host array A.
     * @param elementNum is the number of elements in the array.
     */
    arrayCubeRoot(getDeviceContext(), hostAP, elementNum);

    /**
     * @brief Declaration of a zval for storing the reshaped array.
     */
    zval reshapedZval;

    /**
     * @brief Initialize the zval as an empty array.
     */
    array_init(&reshapedZval);

    /**
     * @brief Reset the shape information index.
     */
    shapeInfoIndex = 0;

    /**
     * @brief Counter to keep track of the previous count during reshaping.
     */
    int previousCount = 0;
    /**
     * @brief Reshape the one-dimensional double pointer array to a zval array.
     * 
     * @param hostAP is the host array to be reshaped.
     * @param reshapedZval is the zval array to store the reshaped array.
     * @param shapeInfo is the shape information array.
     * @param shapeInfoIndex is the index of the shape information array.
     * @param previousCount is the count of elements processed during reshaping.
     */
    dup_oneDimesnPointerArrReshapeToZval(hostAP, reshapedZval, shapeInfo, &shapeInfoIndex, &previousCount);

    /**
     * @brief Set the reshaped zval array as the return value.
     * 
     * @param &reshapedZval is the zval array to be returned.
     * @param 1 indicates the return value should be destroyed by the engine after returning.
     * @param 1 indicates that the return value should be separated from the original zval array.
     */
    RETVAL_ZVAL(&reshapedZval, 1, 1);

    /**
     * @brief Free the memory allocated for host array A.
     */
    free(hostAP);
}

/**
 * Argument information for the Math::logEArray method.
 *
 * @param arrAP An array parameter representing the input array.
 */
ZEND_BEGIN_ARG_INFO_EX(Math_logEArray_ArgInfo, 0, 0, 1)
    ZEND_ARG_ARRAY_INFO(1, arrAP, 0)
ZEND_END_ARG_INFO()

/**
 * Calculates the natural logarithm (base-e logarithm) of an input array.
 *
 * @param execute_data The execution data for the method.
 * @param return_value The return value of the method.
 */
PHP_METHOD(Math, logEArray)
{
    /**
     * @brief Declaration and initialization of zval pointer for array A.
     */
    zval* arrAP = NULL;

    /**
     * @brief Parsing and validating function parameters.
     * 
     * @param 1 minimum required parameter.
     * @param 1 maximum allowed parameter.
     */
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_ARRAY_EX(arrAP, 0, 1)
    ZEND_PARSE_PARAMETERS_END();

    /**
     * @brief Conversion of zval array A to a HashTable.
     */
    HashTable* hashTableAP = Z_ARRVAL_P(arrAP);

    /**
     * @brief Declaration of a zval for storing a one-dimensional array.
     */
    zval oneDimensionZval;

    /**
     * @brief Initialize the zval as an empty array.
     */
    array_init(&oneDimensionZval);

    /**
     * @brief Allocation of memory for shape information.
     * 
     * @param 10 is the size of the shape information array.
     * @param sizeof(int) is the size of each element in the array (integer).
     */
    int* shapeInfo = (int*)calloc(10, sizeof(int));

    /**
     * @brief Index used to keep track of the shape information array.
     */
    int shapeInfoIndex = 0;

    /**
     * @brief Duplicate the HashTable of array A to a one-dimensional zval array.
     * 
     * @param hashTableAP is the HashTable of array A.
     * @param oneDimensionZval is the zval array to store the one-dimensional representation of array A.
     * @param shapeInfo is the shape information array.
     * @param shapeInfoIndex is the index of the shape information array.
     */
    dup_hashTableTo1DZval(hashTableAP, oneDimensionZval, shapeInfo, &shapeInfoIndex);

    /**
     * @brief Get the number of elements in the one-dimensional zval array.
     */
    int elementNum = zend_hash_num_elements(Z_ARRVAL(oneDimensionZval));

    /**
     * @brief Allocation of memory for host array A.
     * 
     * @param elementNum is the number of elements in the array.
     * @param sizeof(double) is the size of each element in the array (double).
     */
    double* hostAP = (double*)calloc(elementNum, sizeof(double));

    /**
     * @brief Duplicate the one-dimensional zval array A to a double pointer array.
     * 
     * @param &oneDimensionZval is the zval array representing the one-dimensional array A.
     * @param hostAP is the double pointer array to store the elements of array A.
     */
    dup_oneDimensionZavlToPointerArr(&oneDimensionZval, hostAP);

    /**
     * @brief Perform the natural logarithm (base e) operation on host array A.
     * 
     * @param getDeviceContext() is a function to get the device context.
     * @param hostAP is the host array A.
     * @param elementNum is the number of elements in the array.
     */
    logEArray(getDeviceContext(), hostAP, elementNum);

    /**
     * @brief Declaration of a zval for storing the reshaped array.
     */
    zval reshapedZval;

    /**
     * @brief Initialize the zval as an empty array.
     */
    array_init(&reshapedZval);

    /**
     * @brief Reset the shape information index.
     */
    shapeInfoIndex = 0;

    /**
     * @brief Counter to keep track of the previous count during reshaping.
     */
    int previousCount = 0;
    /**
     * @brief Reshape the one-dimensional double pointer array to a zval array.
     * 
     * @param hostAP is the host array to be reshaped.
     * @param reshapedZval is the zval array to store the reshaped array.
     * @param shapeInfo is the shape information array.
     * @param shapeInfoIndex is the index of the shape information array.
     * @param previousCount is the count of elements processed during reshaping.
     */
    dup_oneDimesnPointerArrReshapeToZval(hostAP, reshapedZval, shapeInfo, &shapeInfoIndex, &previousCount);

    /**
     * @brief Set the reshaped zval array as the return value.
     * 
     * @param &reshapedZval is the zval array to be returned.
     * @param 1 indicates the return value should be destroyed by the engine after returning.
     * @param 1 indicates that the return value should be separated from the original zval array.
     */
    RETVAL_ZVAL(&reshapedZval, 1, 1);

    /**
     * @brief Free the memory allocated for host array A.
     */
    free(hostAP);
}

/**
 * Argument information for the Math::log2Array method.
 *
 * @param arrAP An array parameter representing the input array.
 */
ZEND_BEGIN_ARG_INFO_EX(Math_log2Array_ArgInfo, 0, 0, 1)
    ZEND_ARG_ARRAY_INFO(1, arrAP, 0)
ZEND_END_ARG_INFO()

/**
 * Calculates the base-2 logarithm of an input array.
 *
 * @param execute_data The execution data for the method.
 * @param return_value The return value of the method.
 */
PHP_METHOD(Math, log2Array)
{
    /**
     * @brief Declaration and initialization of zval pointer for array A.
     */
    zval* arrAP = NULL;

    /**
     * @brief Parsing and validating function parameters.
     * 
     * @param 1 minimum required parameter.
     * @param 1 maximum allowed parameter.
     */
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_ARRAY_EX(arrAP, 0, 1)
    ZEND_PARSE_PARAMETERS_END();

    /**
     * @brief Conversion of zval array A to a HashTable.
     */
    HashTable* hashTableAP = Z_ARRVAL_P(arrAP);

    /**
     * @brief Declaration of a zval for storing a one-dimensional array.
     */
    zval oneDimensionZval;

    /**
     * @brief Initialize the zval as an empty array.
     */
    array_init(&oneDimensionZval);

    /**
     * @brief Allocation of memory for shape information.
     * 
     * @param 10 is the size of the shape information array.
     * @param sizeof(int) is the size of each element in the array (integer).
     */
    int* shapeInfo = (int*)calloc(10, sizeof(int));

    /**
     * @brief Index used to keep track of the shape information array.
     */
    int shapeInfoIndex = 0;

    /**
     * @brief Duplicate the HashTable of array A to a one-dimensional zval array.
     * 
     * @param hashTableAP is the HashTable of array A.
     * @param oneDimensionZval is the zval array to store the one-dimensional representation of array A.
     * @param shapeInfo is the shape information array.
     * @param shapeInfoIndex is the index of the shape information array.
     */
    dup_hashTableTo1DZval(hashTableAP, oneDimensionZval, shapeInfo, &shapeInfoIndex);

    /**
     * @brief Get the number of elements in the one-dimensional zval array.
     */
    int elementNum = zend_hash_num_elements(Z_ARRVAL(oneDimensionZval));

    /**
     * @brief Allocation of memory for host array A.
     * 
     * @param elementNum is the number of elements in the array.
     * @param sizeof(double) is the size of each element in the array (double).
     */
    double* hostAP = (double*)calloc(elementNum, sizeof(double));

    /**
     * @brief Duplicate the one-dimensional zval array A to a double pointer array.
     * 
     * @param &oneDimensionZval is the zval array representing the one-dimensional array A.
     * @param hostAP is the double pointer array to store the elements of array A.
     */
    dup_oneDimensionZavlToPointerArr(&oneDimensionZval, hostAP);

    /**
     * @brief Perform the logarithm base 2 operation on host array A.
     * 
     * @param getDeviceContext() is a function to get the device context.
     * @param hostAP is the host array A.
     * @param elementNum is the number of elements in the array.
     */
    log2Array(getDeviceContext(), hostAP, elementNum);

    /**
     * @brief Declaration of a zval for storing the reshaped array.
     */
    zval reshapedZval;

    /**
     * @brief Initialize the zval as an empty array.
     */
    array_init(&reshapedZval);

    /**
     * @brief Reset the shape information index.
     */
    shapeInfoIndex = 0;

    /**
     * @brief Counter to keep track of the previous count during reshaping.
     */
    int previousCount = 0;
    /**
     * @brief Reshape the one-dimensional double pointer array to a zval array.
     * 
     * @param hostAP is the host array to be reshaped.
     * @param reshapedZval is the zval array to store the reshaped array.
     * @param shapeInfo is the shape information array.
     * @param shapeInfoIndex is the index of the shape information array.
     * @param previousCount is the count of elements processed during reshaping.
     */
    dup_oneDimesnPointerArrReshapeToZval(hostAP, reshapedZval, shapeInfo, &shapeInfoIndex, &previousCount);

    /**
     * @brief Set the reshaped zval array as the return value.
     * 
     * @param &reshapedZval is the zval array to be returned.
     * @param 1 indicates the return value should be destroyed by the engine after returning.
     * @param 1 indicates that the return value should be separated from the original zval array.
     */
    RETVAL_ZVAL(&reshapedZval, 1, 1);

    /**
     * @brief Free the memory allocated for host array A.
     */
    free(hostAP);
}

/**
 * Argument information for the Math::log10Array method.
 *
 * @param arrAP An array parameter representing the input array.
 */
ZEND_BEGIN_ARG_INFO_EX(Math_log10Array_ArgInfo, 0, 0, 1)
    ZEND_ARG_ARRAY_INFO(1, arrAP, 0)
ZEND_END_ARG_INFO()

/**
 * Calculates the base-10 logarithm of an input array.
 *
 * @param execute_data The execution data for the method.
 * @param return_value The return value of the method.
 */
PHP_METHOD(Math, log10Array)
{
    /**
     * @brief Declaration and initialization of zval pointer for array A.
     */
    zval* arrAP = NULL;

    /**
     * @brief Parsing and validating function parameters.
     * 
     * @param 1 minimum required parameter.
     * @param 1 maximum allowed parameter.
     */
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_ARRAY_EX(arrAP, 0, 1)
    ZEND_PARSE_PARAMETERS_END();

    /**
     * @brief Conversion of zval array A to a HashTable.
     */
    HashTable* hashTableAP = Z_ARRVAL_P(arrAP);

    /**
     * @brief Declaration of a zval for storing a one-dimensional array.
     */
    zval oneDimensionZval;

    /**
     * @brief Initialize the zval as an empty array.
     */
    array_init(&oneDimensionZval);

    /**
     * @brief Allocation of memory for shape information.
     * 
     * @param 10 is the size of the shape information array.
     * @param sizeof(int) is the size of each element in the array (integer).
     */
    int* shapeInfo = (int*)calloc(10, sizeof(int));

    /**
     * @brief Index used to keep track of the shape information array.
     */
    int shapeInfoIndex = 0;

    /**
     * @brief Duplicate the HashTable of array A to a one-dimensional zval array.
     * 
     * @param hashTableAP is the HashTable of array A.
     * @param oneDimensionZval is the zval array to store the one-dimensional representation of array A.
     * @param shapeInfo is the shape information array.
     * @param shapeInfoIndex is the index of the shape information array.
     */
    dup_hashTableTo1DZval(hashTableAP, oneDimensionZval, shapeInfo, &shapeInfoIndex);

    /**
     * @brief Get the number of elements in the one-dimensional zval array.
     */
    int elementNum = zend_hash_num_elements(Z_ARRVAL(oneDimensionZval));

    /**
     * @brief Allocation of memory for host array A.
     * 
     * @param elementNum is the number of elements in the array.
     * @param sizeof(double) is the size of each element in the array (double).
     */
    double* hostAP = (double*)calloc(elementNum, sizeof(double));

    /**
     * @brief Duplicate the one-dimensional zval array A to a double pointer array.
     * 
     * @param &oneDimensionZval is the zval array representing the one-dimensional array A.
     * @param hostAP is the double pointer array to store the elements of array A.
     */
    dup_oneDimensionZavlToPointerArr(&oneDimensionZval, hostAP);

    /**
     * @brief Perform the log10 operation on host array A.
     * 
     * @param getDeviceContext() is a function to get the device context.
     * @param hostAP is the host array A.
     * @param elementNum is the number of elements in the array.
     */
    log10Array(getDeviceContext(), hostAP, elementNum);

    /**
     * @brief Declaration of a zval for storing the reshaped array.
     */
    zval reshapedZval;

    /**
     * @brief Initialize the zval as an empty array.
     */
    array_init(&reshapedZval);

    /**
     * @brief Reset the shape information index.
     */
    shapeInfoIndex = 0;

    /**
     * @brief Counter to keep track of the previous count during reshaping.
     */
    int previousCount = 0;
    /**
     * @brief Reshape the one-dimensional double pointer array to a zval array.
     * 
     * @param hostAP is the host array to be reshaped.
     * @param reshapedZval is the zval array to store the reshaped array.
     * @param shapeInfo is the shape information array.
     * @param shapeInfoIndex is the index of the shape information array.
     * @param previousCount is the count of elements processed during reshaping.
     */
    dup_oneDimesnPointerArrReshapeToZval(hostAP, reshapedZval, shapeInfo, &shapeInfoIndex, &previousCount);

    /**
     * @brief Set the reshaped zval array as the return value.
     * 
     * @param &reshapedZval is the zval array to be returned.
     * @param 1 indicates the return value should be destroyed by the engine after returning.
     * @param 1 indicates that the return value should be separated from the original zval array.
     */
    RETVAL_ZVAL(&reshapedZval, 1, 1);

    /**
     * @brief Free the memory allocated for host array A.
     */
    free(hostAP);
}


/**
 * Argument information for the Math::hadamardProduct method.
 *
 * @param arrAP An array parameter representing the first input array.
 * @param arrBP An array parameter representing the second input array.
 */
ZEND_BEGIN_ARG_INFO_EX(Math_hadamardProduct_ArgInfo, 0, 0, 2)
    ZEND_ARG_ARRAY_INFO(1, arrAP, 0)
    ZEND_ARG_ARRAY_INFO(1, arrBP, 0)
ZEND_END_ARG_INFO()

/**
 * Calculates the Hadamard product of two input arrays.
 *
 * @param execute_data The execution data for the method.
 * @param return_value The return value of the method.
 */
PHP_METHOD(Math, hadamardProduct)
{
    /**
     * @brief Declaration and initialization of zval pointers for arrays A and B.
     */
    zval* arrAP = NULL;
    zval* arrBP = NULL;

    /**
     * @brief Parsing and validating function parameters.
     * 
     * @param 2 minimum required parameters.
     * @param 2 maximum allowed parameters.
     */
    ZEND_PARSE_PARAMETERS_START(2, 2)
        Z_PARAM_ARRAY_EX(arrAP, 0, 1)
        Z_PARAM_ARRAY_EX(arrBP, 0, 1)
    ZEND_PARSE_PARAMETERS_END();

    /**
     * @brief Conversion of zval array A to a HashTable.
     */
    HashTable* hashTableAP = Z_ARRVAL_P(arrAP);

    /**
     * @brief Declaration of a zval for storing a one-dimensional array.
     */
    zval oneDimensionAZval;

    /**
     * @brief Initialize the zval as an empty array.
     */
    array_init(&oneDimensionAZval);

    /**
     * @brief Allocation of memory for shape information of array A.
     * 
     * @param 10 is the size of the shape information array.
     * @param sizeof(int) is the size of each element in the array (integer).
     */
    int* shapeInfoA = (int*)calloc(10, sizeof(int));

    /**
     * @brief Index used to keep track of the shape information array for array A.
     */
    int shapeInfoIndexA = 0;

    /**
     * @brief Duplicate the HashTable of array A to a one-dimensional zval array.
     * 
     * @param hashTableAP is the HashTable of array A.
     * @param oneDimensionAZval is the zval array to store the one-dimensional representation of array A.
     * @param shapeInfoA is the shape information array for array A.
     * @param shapeInfoIndexA is the index of the shape information array for array A.
     */
    dup_hashTableTo1DZval(hashTableAP, oneDimensionAZval, shapeInfoA, &shapeInfoIndexA);

    /**
     * @brief Conversion of zval array B to a HashTable.
     */
    HashTable* hashTableBP = Z_ARRVAL_P(arrBP);

    /**
     * @brief Declaration of a zval for storing a one-dimensional array.
     */
    zval oneDimensionBZval;

    /**
     * @brief Initialize the zval as an empty array.
     */
    array_init(&oneDimensionBZval);

    /**
     * @brief Allocation of memory for shape information of array B.
     * 
     * @param 10 is the size of the shape information array.
     * @param sizeof(int) is the size of each element in the array (integer).
     */
    int* shapeInfoB = (int*)calloc(10, sizeof(int));

    /**
     * @brief Index used to keep track of the shape information array for array B.
     */
    int shapeInfoIndexB = 0;

    /**
     * @brief Duplicate the HashTable of array B to a one-dimensional zval array.
     * 
     * @param hashTableBP is the HashTable of array B.
     * @param oneDimensionBZval is the zval array to store the one-dimensional representation of array B.
     * @param shapeInfoB is the shape information array for array B.
     * @param shapeInfoIndexB is the index of the shape information array for array B.
     */
    dup_hashTableTo1DZval(hashTableBP, oneDimensionBZval, shapeInfoB, &shapeInfoIndexB);

    /**
     * @brief Get the number of elements in the one-dimensional zval array.
     */
    int elementNum = zend_hash_num_elements(Z_ARRVAL(oneDimensionAZval));

    /**
     * @brief Allocation of memory for host array A.
     * 
     * @param elementNum is the number of elements in the array.
     * @param sizeof(double) is the size of each element in the array (double).
     */
    double* hostAP = (double*)calloc(elementNum, sizeof(double));

    /**
     * @brief Duplicate the one-dimensional zval array A to a double pointer array.
     * 
     * @param &oneDimensionAZval is the zval array representing the one-dimensional array A.
     * @param hostAP is the double pointer array to store the elements of array A.
     */
    dup_oneDimensionZavlToPointerArr(&oneDimensionAZval, hostAP);

    /**
     * @brief Allocation of memory for host array B.
     * 
     * @param elementNum is the number of elements in the array.
     * @param sizeof(double) is the size of each element in the array (double).
     */
    double* hostBP = (double*)calloc(elementNum, sizeof(double));

    /**
     * @brief Duplicate the one-dimensional zval array B to a double pointer array.
     * 
     * @param &oneDimensionBZval is the zval array representing the one-dimensional array B.
     * @param hostBP is the double pointer array to store the elements of array B.
     */
    dup_oneDimensionZavlToPointerArr(&oneDimensionBZval, hostBP);

    /**
     * @brief Perform the Hadamard product operation on host arrays A and B.
     * 
     * @param getDeviceContext() is a function to get the device context.
     * @param hostAP is the host array A.
     * @param hostBP is the host array B.
     * @param elementNum is the number of elements in the arrays.
     */
    hadamardProduct(getDeviceContext(), hostAP, hostBP, elementNum);

    /**
     * @brief Declaration of a zval for storing the reshaped array.
     */
    zval reshapedZval;

    /**
     * @brief Initialize the zval as an empty array.
     */
    array_init(&reshapedZval);

    /**
     * @brief Reset the shape information index for array A.
     */
    shapeInfoIndexA = 0;

    /**
     * @brief Counter to keep track of the previous count during reshaping.
     */
    int previousCount = 0;

    /**
     * @brief Reshape the one-dimensional double pointer array to a zval array.
     * 
     * @param hostAP is the host array to be reshaped.
     * @param reshapedZval is the zval array to store the reshaped array.
     * @param shapeInfoA is the shape information array for array A.
     * @param shapeInfoIndexA is the index of the shape information array for array A.
     * @param previousCount is the count of elements processed during reshaping.
     */
    dup_oneDimesnPointerArrReshapeToZval(hostAP, reshapedZval, shapeInfoA, &shapeInfoIndexA, &previousCount);

    /**
     * @brief Set the reshaped zval array as the return value.
     * 
     * @param &reshapedZval is the zval array to be returned.
     * @param 1 indicates the return value should be destroyed by the engine after returning.
     * @param 1 indicates that the return value should be separated from the original zval array.
     */
    RETVAL_ZVAL(&reshapedZval, 1, 1);

    /**
     * @brief Free the memory allocated for host array A.
     */
    free(hostAP);

    /**
     * @brief Free the memory allocated for host array B.
     */
    free(hostBP);

}

/**
 * Argument information for the Math::transpose method.
 *
 * @param tensorhubAP An array parameter representing the input matrix.
 */
ZEND_BEGIN_ARG_INFO_EX(Math_transpose_ArgInfo, 0, 0, 1)
    ZEND_ARG_ARRAY_INFO(1, tensorhubAP, 0)
ZEND_END_ARG_INFO()

/**
 * Transposes the input matrix.
 *
 * @param execute_data The execution data for the method.
 * @param return_value The return value of the method.
 */
PHP_METHOD(Math, transpose)
{
    /**
     * @brief Pointer to a zval representing the tensorhubAP variable.
     */
    zval* tensorhubAP = NULL;

    /**
     * @brief Parse parameters from the PHP function call.
     *
     * @param min_num_args Minimum number of arguments required.
     * @param max_num_args Maximum number of arguments allowed.
     */
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_ARRAY_EX(tensorhubAP, 0, 1)
    ZEND_PARSE_PARAMETERS_END();

    /**
     * @brief Pointer to the HashTable structure obtained from tensorhubAP zval.
     */
    HashTable* hashTableAP = Z_ARRVAL_P(tensorhubAP);

    /**
     * @brief zval representing a one-dimensional array.
     */
    zval oneDimensionZval;

    /**
     * @brief Initialize the oneDimensionZval as an empty array.
     */
    array_init(&oneDimensionZval);

    /**
     * @brief Pointer to an array representing shape information.
     */
    int* shapeInfo = (int*)calloc(10, sizeof(int));

    /**
     * @brief Index for accessing shapeInfo array.
     */
    int shapeInfoIndex = 0;

    /**
     * @brief Duplicate the contents of the hashTableAP into a one-dimensional zval array.
     *
     * @param hashTableAP Pointer to the source HashTable.
     * @param oneDimensionZval Target zval for the duplicated data.
     * @param shapeInfo Pointer to the array holding shape information.
     * @param shapeInfoIndex Pointer to the index for accessing shapeInfo.
     */
    dup_hashTableTo1DZval(hashTableAP, oneDimensionZval, shapeInfo, &shapeInfoIndex);

    /**
     * @brief Check the validity of shapeInfo and throw an exception if it's invalid.
     */
    if (shapeInfo[0] < 1 || shapeInfo[1] < 1 || shapeInfo[2] != 0) {
        zend_throw_exception_ex(NULL, 2008, duc_getErrorMsg(2008), "tensorhubA");
    }

    /**
     * @brief Get the number of elements in the oneDimensionZval array.
     */
    int elementNum = zend_hash_num_elements(Z_ARRVAL(oneDimensionZval));

    /**
     * @brief Allocate memory for the hostAP array based on the number of elements.
     */
    double* hostAP = (double*)calloc(elementNum, sizeof(double));

    /**
     * @brief Duplicate the oneDimensionZval array to the hostAP pointer array.
     *
     * @param oneDimensionZval Pointer to the source zval array.
     * @param hostAP Pointer to the target pointer array.
     */
    dup_oneDimensionZavlToPointerArr(&oneDimensionZval, hostAP);

    /**
     * @brief Transpose the hostAP pointer array using the provided parameters.
     *
     * @param deviceContext Device context.
     * @param hostAP Pointer to the host array.
     * @param elementNum Number of elements in the array.
     * @param shapeInfo[0] First dimension size.
     * @param shapeInfo[1] Second dimension size.
     */
    transpose(getDeviceContext(), hostAP, elementNum, shapeInfo[0], shapeInfo[1]);

    /**
     * @brief zval representing the reshaped array.
     */
    zval reshapedZval;

    /**
     * @brief Initialize the reshapedZval as an empty array.
     */
    array_init(&reshapedZval);
    /**
     * @brief Reset the shapeInfoIndex to 0.
     */
    shapeInfoIndex = 0;

    /**
     * @brief Reset the previousCount to 0.
     */
    int previousCount = 0;

    /**
     * @brief Swap the values of the first and second dimensions in shapeInfo.
     *
     * @param shapeInfo Pointer to the array holding shape information.
     */
    int temp = shapeInfo[0];
    shapeInfo[0] = shapeInfo[1];
    shapeInfo[1] = temp;

    /**
     * @brief Reshape the hostAP pointer array into a zval array.
     *
     * @param hostAP Pointer to the source pointer array.
     * @param reshapedZval Target zval for the reshaped array.
     * @param shapeInfo Pointer to the array holding shape information.
     * @param shapeInfoIndex Pointer to the index for accessing shapeInfo.
     * @param previousCount Pointer to the variable holding the previous count.
     */
    dup_oneDimesnPointerArrReshapeToZval(hostAP, reshapedZval, shapeInfo, &shapeInfoIndex, &previousCount);

    /**
     * @brief Set the return value as the reshapedZval.
     *
     * @param reshapedZval Pointer to the zval to be returned.
     * @param copy Copy the zval or not.
     * @param dtor Destruct the zval or not.
     */
    RETVAL_ZVAL(&reshapedZval, 1, 1);

    /**
     * @brief Free the memory allocated for the hostAP pointer array.
     */
    free(hostAP);
}

/**
 * Math_functions[] is an array of zend_function_entry structures that define
 * the PHP methods for the Math class.
 */
zend_function_entry Math_functions[] = {
    /**
     * PHP_ME is a macro that defines a PHP method entry.
     * Math::setDeviceId is a PHP method that sets the device ID.
     * Math_setDeviceId_ArgInfo is the argument information for the method.
     * ZEND_ACC_PUBLIC and ZEND_ACC_STATIC are access modifiers for the method.
     */
    PHP_ME(Math, setDeviceId, Math_setDeviceId_ArgInfo, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC )

    /**
     * PHP_ME is a macro that defines a PHP method entry.
     * Math::getDeviceId is a PHP method that retrieves the device ID.
     * Math_getDeviceId_ArgInfo is the argument information for the method.
     * ZEND_ACC_PUBLIC and ZEND_ACC_STATIC are access modifiers for the method.
     */
    PHP_ME(Math, getDeviceId, Math_getDeviceId_ArgInfo, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC )

    /**
     * PHP_ME is a macro that defines a PHP method entry.
     * Math::arrayAdd is a PHP method that adds arrays.
     * Math_arrayAdd_ArgInfo is the argument information for the method.
     * ZEND_ACC_PUBLIC and ZEND_ACC_STATIC are access modifiers for the method.
     */
    PHP_ME(Math, arrayAdd, Math_arrayAdd_ArgInfo, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC )

    /**
     * PHP_ME is a macro that defines a PHP method entry.
     * Math::subtractArray is a PHP method that subtracts arrays.
     * Math_subtractArray_ArgInfo is the argument information for the method.
     * ZEND_ACC_PUBLIC and ZEND_ACC_STATIC are access modifiers for the method.
     */
    PHP_ME(Math, subtractArray, Math_subtractArray_ArgInfo, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC )

    /**
     * PHP_ME is a macro that defines a PHP method entry.
     * Math::arrayMultiply is a PHP method that multiplies arrays.
     * Math_arrayMultiply_ArgInfo is the argument information for the method.
     * ZEND_ACC_PUBLIC and ZEND_ACC_STATIC are access modifiers for the method.
     */
    PHP_ME(Math, arrayMultiply, Math_arrayMultiply_ArgInfo, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC )

    /**
     * PHP_ME is a macro that defines a PHP method entry.
     * Math::divideArray is a PHP method that divides arrays.
     * Math_divideArray_ArgInfo is the argument information for the method.
     * ZEND_ACC_PUBLIC and ZEND_ACC_STATIC are access modifiers for the method.
     */
    PHP_ME(Math, divideArray, Math_divideArray_ArgInfo, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC )

    /**
     * PHP_ME is a macro that defines a PHP method entry.
     * Math::arrayPower is a PHP method that raises arrays to a power.
     * Math_arrayPower_ArgInfo is the argument information for the method.
     * ZEND_ACC_PUBLIC and ZEND_ACC_STATIC are access modifiers for the method.
     */
    PHP_ME(Math, arrayPower, Math_arrayPower_ArgInfo, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC )

    /**
     * PHP_ME is a macro that defines a PHP method entry.
     * Math::arraySquareRoot is a PHP method that calculates the square root of arrays.
     * Math_arraySquareRoot_ArgInfo is the argument information for the method.
     * ZEND_ACC_PUBLIC and ZEND_ACC_STATIC are access modifiers for the method.
     */
    PHP_ME(Math, arraySquareRoot, Math_arraySquareRoot_ArgInfo, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC )
    /**
     * PHP_ME is a macro that defines a PHP method entry.
     * Math::arrayCubeRoot is a PHP method that calculates the cube root of arrays.
     * Math_arrayCubeRoot_ArgInfo is the argument information for the method.
     * ZEND_ACC_PUBLIC and ZEND_ACC_STATIC are access modifiers for the method.
     */
    PHP_ME(Math, arrayCubeRoot, Math_arrayCubeRoot_ArgInfo, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC )

    /**
     * PHP_ME is a macro that defines a PHP method entry.
     * Math::logEArray is a PHP method that calculates the natural logarithm of arrays.
     * Math_logEArray_ArgInfo is the argument information for the method.
     * ZEND_ACC_PUBLIC and ZEND_ACC_STATIC are access modifiers for the method.
     */
    PHP_ME(Math, logEArray, Math_logEArray_ArgInfo, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC )

    /**
     * PHP_ME is a macro that defines a PHP method entry.
     * Math::log2Array is a PHP method that calculates the base-2 logarithm of arrays.
     * Math_log2Array_ArgInfo is the argument information for the method.
     * ZEND_ACC_PUBLIC and ZEND_ACC_STATIC are access modifiers for the method.
     */
    PHP_ME(Math, log2Array, Math_log2Array_ArgInfo, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC )
    /**
     * PHP_ME is a macro that defines a PHP method entry.
     * Math::log10Array is a PHP method that calculates the base-10 logarithm of arrays.
     * Math_log10Array_ArgInfo is the argument information for the method.
     * ZEND_ACC_PUBLIC and ZEND_ACC_STATIC are access modifiers for the method.
     */
    PHP_ME(Math, log10Array, Math_log10Array_ArgInfo, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC )

    /**
     * PHP_ME is a macro that defines a PHP method entry.
     * Math::hadamardProduct is a PHP method that calculates the Hadamard product of arrays.
     * Math_hadamardProduct_ArgInfo is the argument information for the method.
     * ZEND_ACC_PUBLIC and ZEND_ACC_STATIC are access modifiers for the method.
     */
    PHP_ME(Math, hadamardProduct, Math_hadamardProduct_ArgInfo, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC )

    /**
     * PHP_ME is a macro that defines a PHP method entry.
     * Math::transpose is a PHP method that transposes arrays.
     * Math_transpose_ArgInfo is the argument information for the method.
     * ZEND_ACC_PUBLIC and ZEND_ACC_STATIC are access modifiers for the method.
     */
    PHP_ME(Math, transpose, Math_transpose_ArgInfo, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC )

    PHP_FE_END
};
