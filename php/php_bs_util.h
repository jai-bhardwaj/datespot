#pragma once

#include <php.h>

/**
 * The module entry for the bs_tensorhub extension.
 */
extern zend_module_entry bs_tensorhub_module_entry;

/**
 * Pointer to the module entry for the bs_tensorhub extension.
 */
inline zend_module_entry* phpext_bs_tensorhub_ptr = &bs_tensorhub_module_entry;

/**
 * The version of the bs_tensorhub extension.
 */
#define PHP_BS_tensorhub_VERSION "0.1.0"

#ifdef ZTS
#include <tsrm.h>
#endif

#ifdef COMPILE_DL_BS_tensorhub
extern "C" {
ZEND_TSRMLS_CACHE_EXTERN();
}
#endif

/**
 * Initializes an array by size.
 *
 * @param zend_execute_data* execute_data - Pointer to the execute data.
 * @param zval* return_value - Pointer to the return value.
 *
 * @return void
 */
PHP_FUNCTION(initArrayBySize);

/**
 * Util class for utility functions.
 */
class Util {
public:
    /**
     * Retrieves the number of CUDA devices available.
     *
     * @param zend_execute_data* execute_data - Pointer to the execute data.
     * @param zval* return_value - Pointer to the return value.
     *
     * @return void
     */
    static void cudaGetDeviceCount(zend_execute_data* execute_data, zval* return_value);

    /**
     * Retrieves the name of a CUDA device by its ID.
     *
     * @param zend_execute_data* execute_data - Pointer to the execute data.
     * @param zval* return_value - Pointer to the return value.
     *
     * @return void
     */
    static void getDeviceNameById(zend_execute_data* execute_data, zval* return_value);
};

/**
 * Retrieves the number of CUDA devices available.
 *
 * @param zend_execute_data* execute_data - Pointer to the execute data.
 * @param zval* return_value - Pointer to the return value.
 *
 * @return void
 */
PHP_METHOD(Util, cudaGetDeviceCount);

/**
 * Retrieves the name of a CUDA device by its ID.
 *
 * @param zend_execute_data* execute_data - Pointer to the execute data.
 * @param zval* return_value - Pointer to the return value.
 *
 * @return void
 */
PHP_METHOD(Util, getDeviceNameById);
