#include <cstdio>
#include <cassert>
#include <cuda/runtime_api.h>
#include <cuda/cublas_v2.h>
#include "php.h"
#include <zend_exceptions.h>
#include "ext/standard/info.h"
#include "php_bs_util.h"
#include "dev_util_p.h"

zend_class_entry* Util_ce;

/**
 * @brief Initializes a new array with the specified size.
 *
 * This function creates a new PHP array with the specified size.
 *
 * @param size The size of the array to initialize.
 * @return The initialized PHP array.
 */
ZEND_BEGIN_ARG_WITH_RETURN_TYPE_MASK_EX(Util_initArrayBySize_ArgInfo, 0, 1, MAY_BE_ARRAY)
    ZEND_ARG_TYPE_INFO(0, size, IS_LONG, 0)
ZEND_END_ARG_INFO()

PHP_METHOD_DECL(Util, initArrayBySize) {
    zend_long size;

    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_LONG(size)
    ZEND_PARSE_PARAMETERS_END();

    zval returnZval;
    array_init_size(&returnZval, size);

    RETURN_ZVAL(&returnZval, 1, 1);
}

/**
 * @brief Retrieves the number of available CUDA devices.
 *
 * This function retrieves the number of available CUDA devices and returns
 * it as a PHP long value.
 *
 * @return The number of available CUDA devices.
 */
ZEND_BEGIN_ARG_INFO_EX(Util_cudaGetDeviceCount_ArgInfo, 0, 0, 0)
ZEND_END_ARG_INFO()

PHP_METHOD_DECL(Util, cudaGetDeviceCount){
    int deviceCount = 0;
    checkCudaResult( cudaGetDeviceCount(&deviceCount) );

    RETURN_LONG(deviceCount);
}

/**
 * @brief Retrieves the name of the CUDA device specified by the ID.
 *
 * This function sets the CUDA device to the one specified by the ID and
 * retrieves its properties to obtain the device name. The device name is
 * returned as a PHP string.
 *
 * @param deviceId The ID of the CUDA device.
 * @return The name of the CUDA device.
 */
ZEND_BEGIN_ARG_INFO_EX(Util_getDeviceNameById_ArgInfo, 0, 0, 1)
    ZEND_ARG_TYPE_INFO(0, deviceId, IS_LONG, 0)
ZEND_END_ARG_INFO()

PHP_METHOD_DECL(Util, getDeviceNameById){
    zend_long deviceId;

    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_LONG(deviceId)
    ZEND_PARSE_PARAMETERS_END();

    checkCudaResult( cudaSetDevice( (int)deviceId ) );
    cudaDeviceProp deviceProp;
    checkCudaResult( cudaGetDeviceProperties( &deviceProp, (int)deviceId ) );

    std::string deviceName = std::format("Device name: {}", deviceProp.name);
    RETURN_STRING(deviceName.c_str(), deviceName.length());
}

zend_function_entry Util_functions[] = {
    /**
     * Util::initArrayBySize
     */
    PHP_ME_DECL(Util, initArrayBySize, Util_initArrayBySize_ArgInfo, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC )
    /**
     * Util::cudaGetDeviceCount
     */
    PHP_ME_DECL(Util, cudaGetDeviceCount, Util_cudaGetDeviceCount_ArgInfo, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC )
    /**
     * Util::getDeviceNameById
     */
    PHP_ME_DECL(Util, getDeviceNameById, Util_getDeviceNameById_ArgInfo, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC )
    PHP_FE_END
};
