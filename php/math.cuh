/**
 * @brief Structure to store device context information
 * 
 * @var deviceId ID of the device
 */
typedef struct {
    int deviceId;
} deviceContextStruct;

/**
 * @brief Adds a scalar to an array
 * 
 * @param deviceContextStructP pointer to the device context structure
 * @param hostAP pointer to the input array on the host
 * @param elementNum number of elements in the array
 * @param alpha scalar to be added to the array
 */
void arrayAdd( deviceContextStruct * deviceContextStructP, double * hostAP, int elementNum, double alpha );

/**
 * @brief Subtracts a scalar from an array
 * 
 * @param deviceContextStructP pointer to the device context structure
 * @param alpha scalar to be subtracted from the array
 * @param hostAP pointer to the input array on the host
 * @param elementNum number of elements in the array
 */
void subtractArray( deviceContextStruct * deviceContextStructP, double alpha, double * hostAP, int elementNum );

/**
 * @brief Multiplies an array by a scalar
 * 
 * @param deviceContextStructP pointer to the device context structure
 * @param hostAP pointer to the input array on the host
 * @param elementNum number of elements in the array
 * @param alpha scalar to be multiplied with the array
 */
void arrayMultiply( deviceContextStruct * deviceContextStructP, double * hostAP, int elementNum, double alpha );

/**
 * @brief Divides an array by a scalar
 * 
 * @param deviceContextStructP pointer to the device context structure
 * @param alpha scalar to divide the array by
 * @param hostAP pointer to the input array on the host
 * @param elementNum number of elements in the array
 */
void divideArray( deviceContextStruct * deviceContextStructP, double alpha, double * hostAP, int elementNum );

/**
 * @brief Raises an array to a scalar power
 * 
 * @param deviceContextStructP pointer to the device context structure
 * @param hostAP pointer to the input array on the host
 * @param elementNum number of elements in the array
 * @param alpha scalar to raise the array to
 */
void arrayPower( deviceContextStruct * deviceContextStructP, double * hostAP, int elementNum, double alpha );

/**
 * @brief Calculates the square root of elements in an array
 * 
 * @param deviceContextStructP pointer to the device context structure
 * @param hostAP pointer to the input array on the host
 * @param elementNum number of elements in the array
 */
void arraySquareRoot( deviceContextStruct * deviceContextStructP, double * hostAP, int elementNum );

/**
 * @brief Calculates the cube root of elements in an array
 * 
 * @param deviceContextStructP pointer to the device context structure
 * @param hostAP pointer to the input array on the host
 * @param elementNum number of elements in the array
 */
void arrayCubeRoot( deviceContextStruct * deviceContextStructP, double * hostAP, int elementNum );
/**
 * @brief Computes the natural logarithm of the elements in the array
 *
 * The function takes as input a device context structure, a host array of doubles, and the number of elements in the array.
 * The natural logarithm of each element in the array is computed and stored back in the array.
 *
 * @param deviceContextStructP Pointer to the device context structure
 * @param hostAP Pointer to the host array of doubles
 * @param elementNum The number of elements in the array
 */
void logEArray( deviceContextStruct * deviceContextStructP, double * hostAP, int elementNum );

/**
 * @brief Computes the base-2 logarithm of the elements in the array
 *
 * The function takes as input a device context structure, a host array of doubles, and the number of elements in the array.
 * The base-2 logarithm of each element in the array is computed and stored back in the array.
 *
 * @param deviceContextStructP Pointer to the device context structure
 * @param hostAP Pointer to the host array of doubles
 * @param elementNum The number of elements in the array
 */
void log2Array( deviceContextStruct * deviceContextStructP, double * hostAP, int elementNum );

/**
 * @brief Computes the base-10 logarithm of the elements in the array
 *
 * The function takes as input a device context structure, a host array of doubles, and the number of elements in the array.
 * The base-10 logarithm of each element in the array is computed and stored back in the array.
 *
 * @param deviceContextStructP Pointer to the device context structure
 * @param hostAP Pointer to the host array of doubles
 * @param elementNum The number of elements in the array
 */
void log10Array( deviceContextStruct * deviceContextStructP, double * hostAP, int elementNum );

/**
 * @brief Computes the Hadamard product of two arrays
 *
 * The function takes as input a device context structure, two host arrays of doubles, and the number of elements in the arrays.
 * The Hadamard product of the two arrays is computed and stored back in the first array.
 *
 * @param deviceContextStructP Pointer to the device context structure
 * @param hostAP Pointer to the first host array of doubles
 * @param hostBP Pointer to the second host array of doubles
 * @param elementNum The number of elements in the arrays
 */
void hadamardProduct( deviceContextStruct * deviceContextStructP, double * hostAP, double * hostBP, int elementNum );

/**
 * @brief Transposes a two-dimensional array
 *
 * The function takes as input a device context structure, a host array of doubles, the number of elements in the array,
 * and the width and height of the array. The array is transposed and stored back in the same array.
 *
 * @param deviceContextStructP Pointer to the device context structure
 * @param hostAP Pointer to the host array of doubles
 * @param elementNum The number of elements in the array
 * @param widthA The width of the array
 * @param heightA The height of the array
 */
void transpose( deviceContextStruct * deviceContextStructP, double
