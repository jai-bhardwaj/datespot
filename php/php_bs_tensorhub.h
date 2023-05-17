#ifndef PHP_BS_tensorhub_H
# define PHP_BS_tensorhub_H

/**
 * @file php_bs_tensorhub.h
 * This file contains the declarations and definitions related to the PHP extension bs_tensorhub.
 */

extern zend_module_entry bs_tensorhub_module_entry;
# define phpext_bs_tensorhub_ptr &bs_tensorhub_module_entry

/**
 * @def PHP_BS_tensorhub_VERSION
 * The version number of the bs_tensorhub extension.
 */
# define PHP_BS_tensorhub_VERSION "0.1.0"

# if defined(ZTS) && defined(COMPILE_DL_BS_tensorhub)
ZEND_TSRMLS_CACHE_EXTERN()
# endif

#endif

/**
 * Constructor for the tensorhubTool class.
 */
PHP_METHOD(tensorhubTool, __construct);

/**
 * Sets the handle.
 */
PHP_METHOD(tensorhubTool, setHandle);

/**
 * Retrieves the handle.
 */
PHP_METHOD(tensorhubTool, getHandle);

/**
 * Performs matrix multiplication.
 */
PHP_METHOD(tensorhubTool, multiply);

/**
 * Performs matrix multiplication with scaling.
 */
PHP_METHOD(tensorhubTool, multiplyS);

/**
 * Calculates dot product.
 */
PHP_METHOD(tensorhubTool, dot);

/**
 * Calculates dot product with scaling.
 */
PHP_METHOD(tensorhubTool, dotS);

/**
 * Performs scalar multiplication.
 */
PHP_METHOD(tensorhubTool, scal);

/**
 * Performs scalar multiplication with scaling.
 */
PHP_METHOD(tensorhubTool, scalS);

/**
 * Calculates amax.
 */
PHP_METHOD(tensorhubTool, amax);

/**
 * Calculates amaxS.
 */
PHP_METHOD(tensorhubTool, amaxS);
/**
 * Calculates amin.
 */
PHP_METHOD(tensorhubTool, amin);

/**
 * Calculates aminS.
 */
PHP_METHOD(tensorhubTool, aminS);

/**
 * Performs axpy.
 */
PHP_METHOD(tensorhubTool, axpy);

/**
 * Performs axpyS.
 */
PHP_METHOD(tensorhubTool, axpyS);

/**
 * Performs gemv.
 */
PHP_METHOD(tensorhubTool, gemv);

/**
 * Performs gemvS.
 */
PHP_METHOD(tensorhubTool, gemvS);
