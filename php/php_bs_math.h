#pragma once

#include <iostream>
#include <vector>

namespace bs_tensorhub {

    /**
     * Declare module entry point with C++ attributes.
     */
    [[maybe_unused]] extern "C" auto const bs_tensorhub_module_entry
    {
        /**
         * The size of the module entry.
         */
        .size = sizeof(bs_tensorhub_module_entry),

        /**
         * The version of the Zend Engine API.
         */
        .zend_api = ZEND_MODULE_API_NO,

        /**
         * Flag indicating debug mode.
         */
        .zend_debug = ZEND_DEBUG,

        /**
         * Pointer to INI entry.
         */
        .ini_entry = nullptr,

        /**
         * Pointer to module dependencies.
         */
        .deps = nullptr,

        /**
         * The name of the module.
         */
        .name = "bs_tensorhub",

        /**
         * Pointer to function entries.
         */
        .functions = nullptr,

        /**
         * Pointer to module startup function.
         */
        .module_startup_func = nullptr,

        /**
         * Pointer to module shutdown function.
         */
        .module_shutdown_func = nullptr,

        /**
         * Pointer to request startup function.
         */
        .request_startup_func = nullptr,

        /**
         * Pointer to request shutdown function.
         */
        .request_shutdown_func = nullptr,

        /**
         * Pointer to module info function.
         */
        .info_func = nullptr,

        /**
         * The version of the module.
         */
        .version = PHP_BS_tensorhub_VERSION,

        /**
         * Pointer to module globals constructor.
         */
        .globals_ctor = nullptr,

        /**
         * Pointer to module globals destructor.
         */
        .globals_dtor = nullptr,

        /**
         * Pointer to post-deactivate function.
         */
        .post_deactivate_func = nullptr,

        /**
         * Flag indicating if the module is started.
         */
        .module_started = 0,

        /**
         * The type of the module.
         */
        .type = 0,

        /**
         * Pointer to the handle of the module.
         */
        .handle = nullptr,

        /**
         * The module number.
         */
        .module_number = 0,

        /**
         * The build ID of the module.
         */
        .build_id = ZEND_MODULE_BUILD_ID
    };

    // Define methods using C++ attributes
    class Math {
    public:
        /**
         * Retrieves the methods for the Math class.
         *
         * @return zend_function_entry const* - Pointer to the methods.
         */
        [[nodiscard]] static zend_function_entry const* getMethods() {
            static zend_function_entry const methods[] {
                /**
                 * PHP method: Math::arrayAdd
                 * Parameters: none
                 * Access: public, static
                 */
                PHP_ME(Math, arrayAdd, nullptr, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)

                /**
                 * PHP method: Math::subtractArray
                 * Parameters: none
                 * Access: public, static
                 */
                PHP_ME(Math, subtractArray, nullptr, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)

                /**
                 * PHP method: Math::arrayMultiply
                 * Parameters: none
                 * Access: public, static
                 */
                PHP_ME(Math, arrayMultiply, nullptr, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)

                /**
                 * PHP method: Math::divideArray
                 * Parameters: none
                 * Access: public, static
                 */
                PHP_ME(Math, divideArray, nullptr, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)

                /**
                 * PHP method: Math::arrayPower
                 * Parameters: none
                 * Access: public, static
                 */
                PHP_ME(Math, arrayPower, nullptr, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)

                /**
                 * PHP method: Math::arraySquareRoot
                 * Parameters: none
                 * Access: public, static
                 */
                PHP_ME(Math, arraySquareRoot, nullptr, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)

                /**
                 * PHP method: Math::arrayCubeRoot
                 * Parameters: none
                 * Access: public, static
                 */
                PHP_ME(Math, arrayCubeRoot, nullptr, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)

                /**
                 * PHP method: Math::logEArray
                 * Parameters: none
                 * Access: public, static
                 */
                PHP_ME(Math, logEArray, nullptr, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)

                /**
                 * PHP method: Math::log2Array
                 * Parameters: none
                 * Access: public, static
                 */
                PHP_ME(Math, log2Array, nullptr, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)

                /**
                 * PHP method: Math::log10Array
                 * Parameters: none
                 * Access: public, static
                 */
                PHP_ME(Math, log10Array, nullptr, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)

                /**
                 * PHP method: Math::hadamardProduct
                 * Parameters: none
                 * Access: public, static
                 */
                PHP_ME(Math, hadamardProduct, nullptr, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)

                /**
                 * PHP method: Math::transpose
                 * Parameters: none
                 * Access: public, static
                 */
                PHP_ME(Math, transpose, nullptr, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)

                /**
                 * PHP_FE_END
                 * Marks the end of the method entries.
                 */
                PHP_FE_END
            };
            return methods;
        }

        /**
         * Performs array addition.
         *
         * @param zend_execute_data* execute_data - Pointer to the execute data.
         * @param zval* return_value - Pointer to the return value.
         *
         * @return void
         */
        static void arrayAdd(zend_execute_data* execute_data, zval* return_value) {
        }

        /**
         * Performs array subtraction.
         *
         * @param zend_execute_data* execute_data - Pointer to the execute data.
         * @param zval* return_value - Pointer to the return value.
         *
         * @return void
         */
        static void subtractArray(zend_execute_data* execute_data, zval* return_value) {
        }
    };
} // namespace bs_tensorhub