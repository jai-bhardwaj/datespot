#include <php.h>
#include <ext/standard/info.h>
#include <dev_util_p.h>
#include <unordered_map>
#include <vector>

/**
 * Converts a multidimensional HashTable to a one-dimensional array of doubles.
 * Only the innermost array is used, and any non-double values are ignored.
 * 
 * @param hashTableP The HashTable to convert.
 * @param arrP The output array of doubles.
 */
void dup_HashTableTo1DArr(const HashTable* hashTableP, double* arrP) {
    std::vector<double> arr;
    std::unordered_map<zend_string*, zval*> map;

    ZEND_HASH_FOREACH_VAL(hashTableP, zvalue) {
        if(Z_TYPE_P(zvalue) != IS_ARRAY) {
            continue;
        }
        ZEND_HASH_FOREACH_STR_KEY_VAL(Z_ARRVAL_P(zvalue), key, zv) {
            if(Z_TYPE_P(zv) == IS_DOUBLE) {
                arr.push_back(Z_DVAL_P(zv));
            }
        } ZEND_HASH_FOREACH_END();
    } ZEND_HASH_FOREACH_END();

    std::copy(arr.begin(), arr.end(), arrP);
}

/**
 * Converts a multidimensional HashTable to a one-dimensional array of floats.
 * Only the innermost array is used, and any non-double values are ignored.
 * 
 * @param hashTableP The HashTable to convert.
 * @param arrP The output array of floats.
 */
void dup_HashTableTo1DArrS(const HashTable* hashTableP, float* arrP) {
    std::vector<float> arr;
    std::unordered_map<zend_string*, zval*> map;

    ZEND_HASH_FOREACH_VAL(hashTableP, zvalue) {
        if(Z_TYPE_P(zvalue) != IS_ARRAY) {
            continue;
        }
        ZEND_HASH_FOREACH_STR_KEY_VAL(Z_ARRVAL_P(zvalue), key, zv) {
            if(Z_TYPE_P(zv) == IS_DOUBLE) {
                arr.push_back(static_cast<float>(Z_DVAL_P(zv)));
            }
        } ZEND_HASH_FOREACH_END();
    } ZEND_HASH_FOREACH_END();

    std::copy(arr.begin(), arr.end(), arrP);
}

/**
 * @brief Converts the values of a HashTable into a 1D array of doubles.
 *
 * The function iterates over the values of the HashTable and adds any values of type double to a std::vector.
 * The values are then copied from the std::vector to the 1D array passed as a parameter.
 *
 * @param hashTableP A pointer to the HashTable to convert.
 * @param arrP A pointer to the 1D array where the values will be stored.
 */
void dup_HashTableTo1DArrOne(const HashTable* hashTableP, double* arrP) {
    std::vector<double> arr;

    ZEND_HASH_FOREACH_VAL(hashTableP, zvalue) {
        if(Z_TYPE_P(zvalue) == IS_DOUBLE) {
            arr.push_back(Z_DVAL_P(zvalue));
        }
    } ZEND_HASH_FOREACH_END();

    std::copy(arr.begin(), arr.end(), arrP);
}

/**
 * @brief Converts the values of a HashTable into a 1D array of floats.
 *
 * The function iterates over the values of the HashTable and adds any values of type double to a std::vector of floats.
 * The values are casted from double to float and stored in the std::vector.
 * The values are then copied from the std::vector to the 1D array passed as a parameter.
 *
 * @param hashTableP A pointer to the HashTable to convert.
 * @param arrP A pointer to the 1D array where the values will be stored.
 */
void dup_HashTableTo1DArrOneS(const HashTable* hashTableP, float* arrP) {
    std::vector<float> arr;

    ZEND_HASH_FOREACH_VAL(hashTableP, zvalue) {
        if(Z_TYPE_P(zvalue) == IS_DOUBLE) {
            arr.push_back(static_cast<float>(Z_DVAL_P(zvalue)));
        }
    } ZEND_HASH_FOREACH_END();

    std::copy(arr.begin(), arr.end(), arrP);
}

/**
 * @brief Converts a HashTable into a 1D zval array and stores the shape information of the original HashTable.
 *
 * The function iterates over the values of the HashTable and adds any values of type double to a 1D zval array.
 * If a value is of type array, the function is called recursively on that array.
 * The shape information of the original HashTable is stored in the shapeInfo array.
 *
 * @param hashTableP A pointer to the HashTable to convert.
 * @param oneDimensionzval A pointer to the 1D zval array where the values will be stored.
 * @param shapeInfo An array where the shape information of the original HashTable will be stored.
 * @param shapeInfoIndex A pointer to the current index in the shapeInfo array.
 */
void dup_hashTableTo1DZval(const HashTable* hashTableP, zval* oneDimensionzval, int* shapeInfo, int* shapeInfoIndex) {
    std::vector<int> shape;
    std::vector<zval*> arrays;
    int tempCount = 0;

    ZEND_HASH_FOREACH_VAL(hashTableP, zvalue) {
        if(Z_TYPE_P(zvalue) == IS_ARRAY) {
            shape.push_back(zend_hash_num_elements(Z_ARRVAL_P(zvalue)));
            arrays.push_back(zvalue);
        } else {
            add_next_index_double(oneDimensionzval, zval_get_double_func(zvalue));
            tempCount++;
        }
    } ZEND_HASH_FOREACH_END();

    shapeInfo[*shapeInfoIndex] = tempCount;
    (*shapeInfoIndex)++;

    for(zval* array : arrays) {
        dup_hashTableTo1DZval(Z_ARRVAL_P(array), oneDimensionzval, shapeInfo, shapeInfoIndex);
    }

    (*shapeInfoIndex)--;
}

/**
 * @brief Reshapes a 1D array of doubles into a zval array according to the shape information.
 *
 * The function uses the shape information to convert the 1D array of doubles into a multi-dimensional zval array.
 * The reshaped array is stored in the reshapedZval parameter.
 *
 * @param arrP A pointer to the 1D array of doubles to reshape.
 * @param reshapedZval The zval array where the reshaped array will be stored.
 * @param shapeInfo An array containing the shape information.
 * @param shapeInfoIndex A pointer to the current index in the shapeInfo array.
 * @param previousCount A pointer to the current count of elements in the 1D array.
 */
void dup_oneDimensionPointerArrReshapeToZval(double* arrP, zval reshapedZval, const int* shapeInfo, int* shapeInfoIndex, int* previousCount) {
    int shapeInfoCount = 0;
    while(shapeInfoCount < 10 && shapeInfo[shapeInfoCount] != 0) {
        shapeInfoCount++;
    }

    if(shapeInfo[*shapeInfoIndex] == 0) {
        return;
    }

    if(*shapeInfoIndex == shapeInfoCount - 1) {
        for(int i = 0; i < shapeInfo[*shapeInfoIndex]; i++) {
            add_next_index_double(&reshapedZval, arrP[*previousCount + i]);
        }
        (*previousCount) += shapeInfo[*shapeInfoIndex];
    } else {
        for(int i = 0; i < shapeInfo[*shapeInfoIndex]; i++) {
            zval tempZval;
            array_init(&tempZval);
            (*shapeInfoIndex)++;
            dup_oneDimensionPointerArrReshapeToZval(arrP, tempZval, shapeInfo, shapeInfoIndex, previousCount);
            (*shapeInfoIndex)--;
            add_next_index_zval(&reshapedZval, &tempZval);
        }
    }
}

/**
 * @brief Converts a 1D zval array into a 1D array of doubles.
 *
 * The function iterates over the values of the 1D zval array and adds any values of type double to a 1D array of doubles.
 *
 * @param oneDimensionZval A pointer to the 1D zval array to convert.
 * @param arrP A pointer to the 1D array of doubles where the values will be stored.
 */
void dup_oneDimensionZvalToPointerArr(const zval* oneDimensionZval, double* arrP) {
    zend_ulong index = 0;
    zval* zvalue;
    ZEND_HASH_FOREACH_VAL(Z_ARRVAL_P(oneDimensionZval), zvalue) {
        if(Z_TYPE_P(zvalue) == IS_DOUBLE) {
            arrP[index++] = Z_DVAL_P(zvalue);
        }
    } ZEND_HASH_FOREACH_END();
}
