
#ifndef KERNELS_H
#define KERNELS_H

/**
 * @brief Calculates the number of blocks based on the given size.
 *
 * @param size The size used to calculate the number of blocks.
 * @return The number of blocks as a 32-bit unsigned integer.
 */
uint32_t CalculateBlocks(uint64_t size);

/**
 * @brief Returns the sign of the given value.
 *
 * @tparam T The type of the input value.
 * @param x The input value.
 * @return The sign of the input value (-1 if negative, 0 if zero, 1 if positive).
 */
template<typename T>
__device__ T sgn(T x);

/**
 * @brief Sets the GPU data for kernels.
 */
void SetKernelsGpuData();

/**
 * @brief Gets the GPU data for kernels.
 */
void GetKernelsGpuData();

/**
 * @brief Sets the GPU data for KLoss.
 */
void SetKLossGpuData();

/**
 * @brief Gets the GPU data for KLoss.
 */
void GetKLossGpuData();

/**
 * @brief Sets the GPU data for KActivation.
 */
void SetKActivationGpuData();

/**
 * @brief Gets the GPU data for KActivation.
 */
void GetKActivationGpuData();

/**
 * @brief Sets the GPU data for KDelta.
 */
void SetKDeltaGpuData();

/**
 * @brief Gets the GPU data for KDelta.
 */
void GetKDeltaGpuData();

/**
 * \brief Scales and biases the data.
 * 
 * \param pData Pointer to the data.
 * \param size The size of the data.
 * \param scale The scaling factor.
 * \param bias The bias value.
 */
void kScaleAndBias(Float* pData, uint64_t size, Float scale, Float bias);

/**
 * \brief Adds bias to the unit.
 * 
 * \param pUnit Pointer to the unit.
 * \param pBias Pointer to the bias.
 * \param stride The stride.
 * \param batch The batch size.
 */
void kAddBias(Float* pUnit, Float* pBias, uint32_t stride, uint32_t batch);

/**
 * \brief Adds dual bias to the unit.
 * 
 * \param pUnit Pointer to the unit.
 * \param pBias1 Pointer to the first bias.
 * \param pBias2 Pointer to the second bias.
 * \param stride The stride.
 * \param batch The batch size.
 */
void kAddDualBias(Float* pUnit, Float* pBias1, Float* pBias2, uint32_t stride, uint32_t batch);

/**
 * \brief Adds triple bias to the unit.
 * 
 * \param pUnit Pointer to the unit.
 * \param pBias1 Pointer to the first bias.
 * \param pBias2 Pointer to the second bias.
 * \param pBias3 Pointer to the third bias.
 * \param stride The stride.
 * \param batch The batch size.
 */
void kAddTripleBias(Float* pUnit, Float* pBias1, Float* pBias2, Float* pBias3, uint32_t stride, uint32_t batch);

/**
 * \brief Adds quadruple bias to the unit.
 * 
 * \param pUnit Pointer to the unit.
 * \param pBias1 Pointer to the first bias.
 * \param pBias2 Pointer to the second bias.
 * \param pBias3 Pointer to the third bias.
 * \param pBias4 Pointer to the fourth bias.
 * \param stride The stride.
 * \param batch The batch size.
 */
void kAddQuadBias(Float* pUnit, Float* pBias1, Float* pBias2, Float* pBias3, Float* pBias4, uint32_t stride, uint32_t batch);

/**
 * \brief Clears the unit by setting it to zero.
 * 
 * \param pUnit Pointer to the unit.
 * \param pBias Pointer to the bias.
 * \param stride The stride.
 * \param batch The batch size.
 */
void kClearUnit(Float* pUnit, Float* pBias, uint32_t stride, uint32_t batch);

/**
 * \brief Clears the dual source unit by setting it to zero.
 * 
 * \param pUnit Pointer to the unit.
 * \param pBias1 Pointer to the first bias.
 * \param pBias2 Pointer to the second bias.
 * \param stride The stride.
 * \param batch The batch size.
 */
void kClearDualSourceUnit(Float* pUnit, Float* pBias1, Float* pBias2, uint32_t stride, uint32_t batch);

/**
 * \brief Clears the triple source unit by setting it to zero.
 * 
 * \param pUnit Pointer to the unit.
 * \param pBias1 Pointer to the first bias.
 * \param pBias2 Pointer to the second bias.
 * \param pBias3 Pointer to the third bias.
 * \param stride The stride.
 * \param batch The batch size.
 */
void kClearTripleSourceUnit(Float* pUnit, Float* pBias1, Float* pBias2, Float* pBias3, uint32_t stride, uint32_t batch);

/**
 * \brief Clears the quadruple source unit by setting it to zero.
 * 
 * \param pUnit Pointer to the unit.
 * \param pBias1 Pointer to the first bias.
 * \param pBias2 Pointer to the second bias.
 * \param pBias3 Pointer to the third bias.
 * \param pBias4 Pointer to the fourth bias.
 * \param stride The stride.
 * \param batch The batch size.
 */
void kClearQuadSourceUnit(Float* pUnit, Float* pBias1, Float* pBias2, Float* pBias3, Float* pBias4, uint32_t stride, uint32_t batch);

/**
 * \brief Updates the biases using delta values.
 * 
 * \param alpha The update rate.
 * \param batch The batch size.
 * \param width The width.
 * \param pDelta Pointer to the delta values.
 * \param pBias Pointer to the biases.
 */
void kUpdateBiases(Float alpha, uint32_t batch, uint32_t width, Float* pDelta, Float* pBias);

/**
 * \brief Calculates the output using key and value data.
 * 
 * \param pOutputKey Pointer to the output keys.
 * \param pKey Pointer to the keys.
 * \param pValue Pointer to the values.
 * \param batch The batch size.
 * \param width The width.
 * \param k The number of output values per key.
 */
void kCalculateOutput(Float* pOutputKey, Float *pKey, uint32_t* pValue, uint32_t batch, uint32_t width, uint32_t k);

/**
 * \brief Calculates the output using key and value data.
 * 
 * \param pOutputKey Pointer to the output keys.
 * \param pOutputValue Pointer to the output values.
 * \param pKey Pointer to the keys.
 * \param pValue Pointer to the values.
 * \param batch The batch size.
 * \param width The width.
 * \param k The number of output values per key.
 */
void kCalculateOutput(Float* pOutputKey, Float* pOutputValue, Float *pKey, Float* pValue, uint32_t batch, uint32_t width, uint32_t k);

/**
 * \brief Calculates the output using key and value data.
 * 
 * \param pOutputKey Pointer to the output keys.
 * \param pOutputValue Pointer to the output values.
 * \param pKey Pointer to the keys.
 * \param pValue Pointer to the values.
 * \param batch The batch size.
 * \param width The width.
 * \param k The number of output values per key.
 */
void kCalculateOutput(Float* pOutputKey, uint32_t* pOutputValue, Float *pKey, uint32_t * pValue, uint32_t batch, uint32_t width, uint32_t k);

/**
 * \brief Calculates the k-sparse representation of the unit.
 * 
 * \param pUnit Pointer to the unit.
 * \param batch The batch size.
 * \param stride The stride.
 * \param kSparse The sparsity level.
 */
void kCalculateKSparse(Float* pUnit, uint32_t batch, uint32_t stride, uint32_t kSparse);

/**
 * \brief Adds scaled buffers.
 * 
 * \param pDest Pointer to the destination buffer.
 * \param pSrc Pointer to the source buffer.
 * \param scale The scaling factor.
 * \param size The size of the buffers.
 */
void kAddScaleBuffers(Float* pDest, Float* pSrc, Float scale, uint64_t size);

/**
 * \brief Adds buffers.
 * 
 * \param pDest Pointer to the destination buffer.
 * \param pSrc Pointer to the source buffer.
 * \param size The size of the buffers.
 * \param stream The CUDA stream (default: 0).
 */
void kAddBuffers(Float* pDest, Float* pSrc, uint64_t size, cudaStream_t stream = 0);

/**
 * \brief Adds 2D buffers.
 * 
 * \param pDest Pointer to the destination buffer.
 * \param dpitch The pitch of the destination buffer.
 * \param pSrc Pointer to the source buffer.
 * \param spitch The pitch of the source buffer.
 * \param width The width of the buffers.
 * \param height The height of the buffers.
 * \param stream The CUDA stream (default: 0).
 */
void kAddBuffers2D(Float* pDest, uint32_t dpitch, Float* pSrc, uint32_t spitch, uint32_t width, uint32_t height, cudaStream_t stream = 0);

/**
 * \brief Copies 2D buffers.
 * 
 * \param pDest Pointer to the destination buffer.
 * \param dpitch The pitch of the destination buffer.
 * \param pSrc Pointer to the source buffer.
 * \param spitch The pitch of the source buffer.
 * \param width The width of the buffers.
 * \param height The height of the buffers.
 * \param stream The CUDA stream (default: 0).
 */
void kCopy2D(Float* pDest, uint32_t dpitch, Float* pSrc, uint32_t spitch, uint32_t width, uint32_t height, cudaStream_t stream = 0);

/**
 * \brief Initializes the sorting operation.
 * 
 * \tparam KeyType The key data type.
 * \tparam ValueType The value data type.
 * \param items The number of items to sort.
 * \param pbKey Pointer to the key buffer.
 * \param pbValue Pointer to the value buffer.
 * \return The size of the temporary buffer required.
 */
template<typename KeyType, typename ValueType>
size_t kInitSort(uint32_t items, GpuBuffer<KeyType>* pbKey, GpuBuffer<ValueType>* pbValue);

/**
 * \brief Sorts the key-value pairs.
 * 
 * \tparam KeyType The key data type.
 * \tparam ValueType The value data type.
 * \param items The number of items to sort.
 * \param pKey0 Pointer to the first set of keys.
 * \param pKey1 Pointer to the second set of keys.
 * \param pValue0 Pointer to the first set of values.
 * \param pValue1 Pointer to the second set of values.
 * \param pTemp Pointer to the temporary buffer.
 * \param tempBytes The size of the temporary buffer.
 * \return true if the sort operation is successful, false otherwise.
 */
template<typename KeyType, typename ValueType>
bool kSort(uint32_t items, KeyType* pKey0, KeyType* pKey1, ValueType* pValue0, ValueType* pValue1, char* pTemp, size_t tempBytes);


/**
 * \brief Loads input unit with data of type T.
 * 
 * \tparam T The data type.
 * \param position The position.
 * \param batch The batch size.
 * \param stride The stride.
 * \param pUnit Pointer to the unit.
 * \param pData Pointer to the data.
 */
template<typename T>
void kLoadInputUnit(uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit, T* pData);

/**
 * \brief Loads input unit with indexed data of type T.
 * 
 * \tparam T The data type.
 * \param position The position.
 * \param batch The batch size.
 * \param stride The stride.
 * \param pUnit Pointer to the unit.
 * \param pIndex Pointer to the index.
 * \param pData Pointer to the data.
 */
template<typename T>
void kLoadIndexedInputUnit(uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit, uint32_t* pIndex, T* pData);

/**
 * \brief Loads sparse input unit.
 * 
 * \param position The position.
 * \param batch The batch size.
 * \param stride The stride.
 * \param pUnit Pointer to the unit.
 * \param pSparseStart Pointer to the start indices of the sparse data.
 * \param pSparseEnd Pointer to the end indices of the sparse data.
 * \param pSparseIndex Pointer to the indices of the sparse data.
 * \param pDataWeight Pointer to the weights of the sparse data.
 */
void kLoadSparseInputUnit(uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, Float* pDataWeight);

/**
 * \brief Loads sparse input unit with indexed data.
 * 
 * \param position The position.
 * \param batch The batch size.
 * \param stride The stride.
 * \param pUnit Pointer to the unit.
 * \param pIndex Pointer to the index.
 * \param pSparseStart Pointer to the start indices of the sparse data.
 * \param pSparseEnd Pointer to the end indices of the sparse data.
 * \param pSparseIndex Pointer to the indices of the sparse data.
 * \param pDataWeight Pointer to the weights of the sparse data.
 */
void kLoadIndexedSparseInputUnit(uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, Float* pDataWeight);

/**
 * \brief Loads sparse denoised input unit.
 * 
 * \param position The position.
 * \param batch The batch size.
 * \param stride The stride.
 * \param pUnit Pointer to the unit.
 * \param pSparseStart Pointer to the start indices of the sparse data.
 * \param pSparseEnd Pointer to the end indices of the sparse data.
 * \param pSparseIndex Pointer to the indices of the sparse data.
 * \param pDataWeight Pointer to the weights of the sparse data.
 * \param pRandom Pointer to the random data.
 */
void kLoadSparseDenoisedInputUnit(uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, Float* pDataWeight, Float* pRandom);

/**
 * \brief Loads sparse denoised input unit with indexed data.
 * 
 * \param position The position.
 * \param batch The batch size.
 * \param stride The stride.
 * \param pUnit Pointer to the unit.
 * \param pIndex Pointer to the index.
 * \param pSparseStart Pointer to the start indices of the sparse data.
 * \param pSparseEnd Pointer to the end indices of the sparse data.
 * \param pSparseIndex Pointer to the indices of the sparse data.
 * \param pDataWeight Pointer to the weights of the sparse data.
 * \param pRandom Pointer to the random data.
 */
void kLoadIndexedSparseDenoisedInputUnit(uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, Float* pDataWeight, Float* pRandom);

/**
 * \brief Loads sparse analog input unit with data of type T.
 * 
 * \tparam T The data type.
 * \param position The position.
 * \param batch The batch size.
 * \param stride The stride.
 * \param pUnit Pointer to the unit.
 * \param pSparseStart Pointer to the start indices of the sparse data.
 * \param pSparseEnd Pointer to the end indices of the sparse data.
 * \param pSparseIndex Pointer to the indices of the sparse data.
 * \param pDataWeight Pointer to the weights of the sparse data.
 * \param pSparseData Pointer to the sparse data.
 */
template<typename T>
void kLoadSparseAnalogInputUnit(uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, Float* pDataWeight, T* pSparseData);

/**
 * \brief Loads sparse analog input unit with indexed data of type T.
 * 
 * \tparam T The data type.
 * \param position The position.
 * \param batch The batch size.
 * \param stride The stride.
 * \param pUnit Pointer to the unit.
 * \param pIndex Pointer to the index.
 * \param pSparseStart Pointer to the start indices of the sparse data.
 * \param pSparseEnd Pointer to the end indices of the sparse data.
 * \param pSparseIndex Pointer to the indices of the sparse data.
 * \param pDataWeight Pointer to the weights of the sparse data.
 * \param pSparseData Pointer to the sparse data.
 */
template<typename T>
void kLoadIndexedSparseAnalogInputUnit(uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, Float* pDataWeight, T* pSparseData);

/**
 * \brief Loads sparse analog denoised input unit with data of type T.
 * 
 * \tparam T The data type.
 * \param position The position.
 * \param batch The batch size.
 * \param stride The stride.
 * \param pUnit Pointer to the unit.
 * \param pSparseStart Pointer to the start indices of the sparse data.
 * \param pSparseEnd Pointer to the end indices of the sparse data.
 * \param pSparseIndex Pointer to the indices of the sparse data.
 * \param pDataWeight Pointer to the weights of the sparse data.
 * \param pSparseData Pointer to the sparse data.
 * \param pRandom Pointer to the random data.
 */
template<typename T>
void kLoadSparseAnalogDenoisedInputUnit(uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, Float* pDataWeight, T* pSparseData, Float* pRandom);

/**
 * \brief Loads sparse analog denoised input unit with indexed data of type T.
 * 
 * \tparam T The data type.
 * \param position The position.
 * \param batch The batch size.
 * \param stride The stride.
 * \param pUnit Pointer to the unit.
 * \param pIndex Pointer to the index.
 * \param pSparseStart Pointer to the start indices of the sparse data.
 * \param pSparseEnd Pointer to the end indices of the sparse data.
 * \param pSparseIndex Pointer to the indices of the sparse data.
 * \param pDataWeight Pointer to the weights of the sparse data.
 * \param pSparseData Pointer to the sparse data.
 * \param pRandom Pointer to the random data.
 */
template<typename T>
void kLoadIndexedSparseAnalogDenoisedInputUnit(uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, Float* pDataWeight, T* pSparseData, Float* pRandom);


/**
 * Calculates the sparse Z value.
 *
 * @param position                 The position.
 * @param batch                    The batch.
 * @param stride                   The stride.
 * @param pWeight                  Pointer to the weight.
 * @param pSparseStart             Pointer to the start of the sparse matrix.
 * @param pSparseEnd               Pointer to the end of the sparse matrix.
 * @param pSparseIndex             Pointer to the sparse matrix index.
 * @param pDataWeight              Pointer to the data weight.
 * @param pUnit                    Pointer to the unit.
 * @param beta                     The beta value.
 */
void kCalculateSparseZ(uint32_t position, uint32_t batch, uint32_t stride, Float* pWeight, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, Float* pDataWeight, Float* pUnit, Float beta);

/**
 * Calculates the indexed sparse Z value.
 *
 * @param position                 The position.
 * @param batch                    The batch.
 * @param stride                   The stride.
 * @param pWeight                  Pointer to the weight.
 * @param pIndex                   Pointer to the index.
 * @param pSparseStart             Pointer to the start of the sparse matrix.
 * @param pSparseEnd               Pointer to the end of the sparse matrix.
 * @param pSparseIndex             Pointer to the sparse matrix index.
 * @param pDataWeight              Pointer to the data weight.
 * @param pUnit                    Pointer to the unit.
 * @param beta                     The beta value.
 */
void kCalculateIndexedSparseZ(uint32_t position, uint32_t batch, uint32_t stride, Float* pWeight, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, Float* pDataWeight, Float* pUnit, Float beta);

/**
 * Calculates the sparse analog Z value.
 *
 * @param position                 The position.
 * @param batch                    The batch.
 * @param stride                   The stride.
 * @param pWeight                  Pointer to the weight.
 * @param pSparseStart             Pointer to the start of the sparse matrix.
 * @param pSparseEnd               Pointer to the end of the sparse matrix.
 * @param pSparseIndex             Pointer to the sparse matrix index.
 * @param pDataWeight              Pointer to the data weight.
 * @param pSparseData              Pointer to the sparse matrix data.
 * @param pUnit                    Pointer to the unit.
 * @param beta                     The beta value.
 */
template<typename T>
void kCalculateSparseAnalogZ(uint32_t position, uint32_t batch, uint32_t stride, Float* pWeight, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, Float* pDataWeight, T* pSparseData, Float* pUnit, Float beta);

/**
 * Calculates the indexed sparse analog Z value.
 *
 * @param position                 The position.
 * @param batch                    The batch.
 * @param stride                   The stride.
 * @param pWeight                  Pointer to the weight.
 * @param pIndex                   Pointer to the index.
 * @param pSparseStart             Pointer to the start of the sparse matrix.
 * @param pSparseEnd               Pointer to the end of the sparse matrix.
 * @param pSparseIndex             Pointer to the sparse matrix index.
 * @param pDataWeight              Pointer to the data weight.
 * @param pSparseData              Pointer to the sparse matrix data.
 * @param pUnit                    Pointer to the unit.
 * @param beta                     The beta value.
 */
template<typename T>
void kCalculateIndexedSparseAnalogZ(uint32_t position, uint32_t batch, uint32_t stride, Float* pWeight, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, Float* pDataWeight, T* pSparseData, Float* pUnit, Float beta);

/**
 * Calculates the denoised sparse Z value.
 *
 * @param position                 The position.
 * @param batch                    The batch.
 * @param stride                   The stride.
 * @param pWeight                  Pointer to the weight.
 * @param pSparseStart             Pointer to the start of the sparse matrix.
 * @param pSparseEnd               Pointer to the end of the sparse matrix.
 * @param pSparseIndex             Pointer to the sparse matrix index.
 * @param pDataWeight              Pointer to the data weight.
 * @param pRandom                  Pointer to the random data.
 * @param pUnit                    Pointer to the unit.
 * @param beta                     The beta value.
 */
void kCalculateSparseDenoisedZ(uint32_t position, uint32_t batch, uint32_t stride, Float* pWeight, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, Float* pDataWeight, Float* pRandom, Float* pUnit, Float beta);

/**
 * Calculates the denoised indexed sparse Z value.
 *
 * @param position                 The position.
 * @param batch                    The batch.
 * @param stride                   The stride.
 * @param pWeight                  Pointer to the weight.
 * @param pIndex                   Pointer to the index.
 * @param pSparseStart             Pointer to the start of the sparse matrix.
 * @param pSparseEnd               Pointer to the end of the sparse matrix.
 * @param pSparseIndex             Pointer to the sparse matrix index.
 * @param pDataWeight              Pointer to the data weight.
 * @param pRandom                  Pointer to the random data.
 * @param pUnit                    Pointer to the unit.
 * @param beta                     The beta value.
 */
void kCalculateIndexedSparseDenoisedZ(uint32_t position, uint32_t batch, uint32_t stride, Float* pWeight, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, Float* pDataWeight, Float* pRandom, Float* pUnit, Float beta);

/**
 * Calculates the denoised sparse analog Z value.
 *
 * @param position                 The position.
 * @param batch                    The batch.
 * @param stride                   The stride.
 * @param pWeight                  Pointer to the weight.
 * @param pSparseStart             Pointer to the start of the sparse matrix.
 * @param pSparseEnd               Pointer to the end of the sparse matrix.
 * @param pSparseIndex             Pointer to the sparse matrix index.
 * @param pDataWeight              Pointer to the data weight.
 * @param pSparseData              Pointer to the sparse matrix data.
 * @param pRandom                  Pointer to the random data.
 * @param pUnit                    Pointer to the unit.
 * @param beta                     The beta value.
 */
template<typename T>
void kCalculateSparseAnalogDenoisedZ(uint32_t position, uint32_t batch, uint32_t stride, Float* pWeight, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, Float* pDataWeight, T* pSparseData, Float* pRandom, Float* pUnit, Float beta);

/**
 * Calculates the denoised indexed sparse analog Z value.
 *
 * @param position                 The position.
 * @param batch                    The batch.
 * @param stride                   The stride.
 * @param pWeight                  Pointer to the weight.
 * @param pIndex                   Pointer to the index.
 * @param pSparseStart             Pointer to the start of the sparse matrix.
 * @param pSparseEnd               Pointer to the end of the sparse matrix.
 * @param pSparseIndex             Pointer to the sparse matrix index.
 * @param pDataWeight              Pointer to the data weight.
 * @param pSparseData              Pointer to the sparse matrix data.
 * @param pRandom                  Pointer to the random data.
 * @param pUnit                    Pointer to the unit.
 * @param beta                     The beta value.
 */
template<typename T>
void kCalculateIndexedSparseAnalogDenoisedZ(uint32_t position, uint32_t batch, uint32_t stride, Float* pWeight, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, Float* pDataWeight, T* pSparseData, Float* pRandom, Float* pUnit, Float beta);

/**
 * Calculates the sparse transposed matrix.
 *
 * @param position                 The position.
 * @param batch                    The batch.
 * @param pSparseStart             Pointer to the start of the sparse matrix.
 * @param pSparseEnd               Pointer to the end of the sparse matrix.
 * @param pSparseIndex             Pointer to the sparse matrix index.
 * @param pDataWeight              Pointer to the data weight.
 * @param pSparseTransposedEnd     Pointer to the end of the transposed sparse matrix.
 * @param pSparseTransposedIndex   Pointer to the transposed sparse matrix index.
 * @param pSparseTransposedData    Pointer to the transposed sparse matrix data.
 */
void kCalculateSparseTransposedMatrix(uint32_t position, uint32_t batch, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, Float* pDataWeight, uint32_t* pSparseTransposedEnd, uint32_t* pSparseTransposedIndex, Float* pSparseTransposedData);

/**
 * Calculates the indexed sparse transposed matrix.
 *
 * @param position                 The position.
 * @param batch                    The batch.
 * @param pIndex                   Pointer to the index.
 * @param pSparseStart             Pointer to the start of the sparse matrix.
 * @param pSparseEnd               Pointer to the end of the sparse matrix.
 * @param pSparseIndex             Pointer to the sparse matrix index.
 * @param pDataWeight              Pointer to the data weight.
 * @param pSparseTransposedEnd     Pointer to the end of the transposed sparse matrix.
 * @param pSparseTransposedIndex   Pointer to the transposed sparse matrix index.
 * @param pSparseTransposedData    Pointer to the transposed sparse matrix data.
 */
void kCalculateIndexedSparseTransposedMatrix(uint32_t position, uint32_t batch, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, Float* pDataWeight, uint32_t* pSparseTransposedEnd, uint32_t* pSparseTransposedIndex, Float* pSparseTransposedData);

/**
 * Calculates the sparse transposed denoised matrix.
 *
 * @param position                 The position.
 * @param batch                    The batch.
 * @param pSparseStart             Pointer to the start of the sparse matrix.
 * @param pSparseEnd               Pointer to the end of the sparse matrix.
 * @param pSparseIndex             Pointer to the sparse matrix index.
 * @param pDataWeight              Pointer to the data weight.
 * @param pRandom                  Pointer to the random data.
 * @param pSparseTransposedEnd     Pointer to the end of the transposed sparse matrix.
 * @param pSparseTransposedIndex   Pointer to the transposed sparse matrix index.
 * @param pSparseTransposedData    Pointer to the transposed sparse matrix data.
 */
void kCalculateSparseTransposedDenoisedMatrix(uint32_t position, uint32_t batch, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, Float* pDataWeight, Float* pRandom, uint32_t* pSparseTransposedEnd, uint32_t* pSparseTransposedIndex, Float* pSparseTransposedData);

/**
 * Calculates the indexed sparse transposed denoised matrix.
 *
 * @param position                 The position.
 * @param batch                    The batch.
 * @param pIndex                   Pointer to the index.
 * @param pSparseStart             Pointer to the start of the sparse matrix.
 * @param pSparseEnd               Pointer to the end of the sparse matrix.
 * @param pSparseIndex             Pointer to the sparse matrix index.
 * @param pDataWeight              Pointer to the data weight.
 * @param pRandom                  Pointer to the random data.
 * @param pSparseTransposedEnd     Pointer to the end of the transposed sparse matrix.
 * @param pSparseTransposedIndex   Pointer to the transposed sparse matrix index.
 * @param pSparseTransposedData    Pointer to the transposed sparse matrix data.
 */
void kCalculateIndexedSparseTransposedDenoisedMatrix(uint32_t position, uint32_t batch, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, Float* pDataWeight, Float* pRandom, uint32_t* pSparseTransposedEnd, uint32_t* pSparseTransposedIndex, Float* pSparseTransposedData);

/**
 * Calculates the sparse transposed weight gradient.
 *
 * @param alpha                    The alpha value.
 * @param beta                     The beta value.
 * @param m                        The m value.
 * @param n                        The n value.
 * @param pSparseTransposedStart   Pointer to the start of the transposed sparse matrix.
 * @param pSparseTransposedEnd     Pointer to the end of the transposed sparse matrix.
 * @param pSparseTransposedIndex   Pointer to the transposed sparse matrix index.
 * @param pDelta                   Pointer to the delta.
 * @param pWeightGradient          Pointer to the weight gradient.
 */
void kCalculateSparseTransposedWeightGradient(Float alpha, Float beta, uint32_t m, uint32_t n, uint32_t* pSparseTransposedStart, uint32_t* pSparseTransposedEnd, uint32_t* pSparseTransposedIndex, Float* pDelta, Float* pWeightGradient);

/**
 * Calculates the sparse transposed analog matrix.
 *
 * @param position                 The position.
 * @param batch                    The batch.
 * @param pSparseStart             Pointer to the start of the sparse matrix.
 * @param pSparseEnd               Pointer to the end of the sparse matrix.
 * @param pSparseIndex             Pointer to the sparse matrix index.
 * @param pDataWeight              Pointer to the data weight.
 * @param pSparseData              Pointer to the sparse matrix data.
 * @param pSparseTransposedEnd     Pointer to the end of the transposed sparse matrix.
 * @param pSparseTransposedIndex   Pointer to the transposed sparse matrix index.
 * @param pSparseTransposedData    Pointer to the transposed sparse matrix data.
 */
template<typename T>
void kCalculateSparseTransposedAnalogMatrix(uint32_t position, uint32_t batch, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, Float* pDataWeight, T* pSparseData, uint32_t* pSparseTransposedEnd, uint32_t* pSparseTransposedIndex, Float* pSparseTransposedData);

/**
 * Calculates the indexed sparse transposed analog matrix.
 *
 * @tparam T The data type.
 * @param position The position value.
 * @param batch The batch value.
 * @param pIndex Pointer to the index array.
 * @param pSparseStart Pointer to the sparse start array.
 * @param pSparseEnd Pointer to the sparse end array.
 * @param pSparseIndex Pointer to the sparse index array.
 * @param pDataWeight Pointer to the data weight array.
 * @param pSparseData Pointer to the sparse data array.
 * @param pSparseTransposedEnd Pointer to the sparse transposed end array.
 * @param pSparseTransposedIndex Pointer to the sparse transposed index array.
 * @param pSparseTransposedData Pointer to the sparse transposed data array.
 */
template<typename T>
void kCalculateIndexedSparseTransposedAnalogMatrix(uint32_t position, uint32_t batch, uint32_t* pIndex,
                                                  uint64_t* pSparseStart, uint64_t* pSparseEnd,
                                                  uint32_t* pSparseIndex, Float* pDataWeight,
                                                  T* pSparseData, uint32_t* pSparseTransposedEnd,
                                                  uint32_t* pSparseTransposedIndex,
                                                  Float* pSparseTransposedData);

/**
 * Calculates the sparse transposed analog denoised matrix.
 *
 * @tparam T The data type.
 * @param position The position value.
 * @param batch The batch value.
 * @param pSparseStart Pointer to the sparse start array.
 * @param pSparseEnd Pointer to the sparse end array.
 * @param pSparseIndex Pointer to the sparse index array.
 * @param pDataWeight Pointer to the data weight array.
 * @param pSparseData Pointer to the sparse data array.
 * @param pRandom Pointer to the random array.
 * @param pSparseTransposedEnd Pointer to the sparse transposed end array.
 * @param pSparseTransposedIndex Pointer to the sparse transposed index array.
 * @param pSparseTransposedData Pointer to the sparse transposed data array.
 */
template<typename T>
void kCalculateSparseTransposedAnalogDenoisedMatrix(uint32_t position, uint32_t batch,
                                                    uint64_t* pSparseStart, uint64_t* pSparseEnd,
                                                    uint32_t* pSparseIndex, Float* pDataWeight,
                                                    T* pSparseData, Float* pRandom,
                                                    uint32_t* pSparseTransposedEnd,
                                                    uint32_t* pSparseTransposedIndex,
                                                    Float* pSparseTransposedData);

/**
 * Calculates the indexed sparse transposed analog denoised matrix.
 *
 * @tparam T The data type.
 * @param position The position value.
 * @param batch The batch value.
 * @param pIndex Pointer to the index array.
 * @param pSparseStart Pointer to the sparse start array.
 * @param pSparseEnd Pointer to the sparse end array.
 * @param pSparseIndex Pointer to the sparse index array.
 * @param pDataWeight Pointer to the data weight array.
 * @param pSparseData Pointer to the sparse data array.
 * @param pRandom Pointer to the random array.
 * @param pSparseTransposedEnd Pointer to the sparse transposed end array.
 * @param pSparseTransposedIndex Pointer to the sparse transposed index array.
 * @param pSparseTransposedData Pointer to the sparse transposed data array.
 */
template<typename T>
void kCalculateIndexedSparseTransposedAnalogDenoisedMatrix(uint32_t position, uint32_t batch,
                                                           uint32_t* pIndex, uint64_t* pSparseStart,
                                                           uint64_t* pSparseEnd, uint32_t* pSparseIndex,
                                                           Float* pDataWeight, T* pSparseData,
                                                           Float* pRandom, uint32_t* pSparseTransposedEnd,
                                                           uint32_t* pSparseTransposedIndex,
                                                           Float* pSparseTransposedData);

/**
 * Calculates the sparse transposed analog weight gradient.
 *
 * @param alpha The alpha value.
 * @param beta The beta value.
 * @param m The m value.
 * @param n The n value.
 * @param pSparseTransposedStart Pointer to the sparse transposed start array.
 * @param pSparseTransposedEnd Pointer to the sparse transposed end array.
 * @param pSparseTransposedIndex Pointer to the sparse transposed index array.
 * @param pSparseTransposedData Pointer to the sparse transposed data array.
 * @param pDelta Pointer to the delta array.
 * @param pWeightGradient Pointer to the weight gradient array.
 */
void kCalculateSparseTransposedAnalogWeightGradient(Float alpha, Float beta, uint32_t m, uint32_t n,
                                                    uint32_t* pSparseTransposedStart,
                                                    uint32_t* pSparseTransposedEnd,
                                                    uint32_t* pSparseTransposedIndex,
                                                    Float* pSparseTransposedData, Float* pDelta,
                                                    Float* pWeightGradient);

/**
 * Calculates the L1 error.
 *
 * @tparam T The data type.
 * @param position The position value.
 * @param batch The batch value.
 * @param stride The stride value.
 * @param pUnit Pointer to the unit array.
 * @param pData Pointer to the data array.
 * @param pDataWeight Pointer to the data weight array.
 * @return The calculated L1 error.
 */
template<typename T>
Float kCalculateL1Error(uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit,
                        T* pData, Float* pDataWeight);

/**
 * Calculates the indexed L1 error.
 *
 * @tparam T The data type.
 * @param position The position value.
 * @param batch The batch value.
 * @param stride The stride value.
 * @param pUnit Pointer to the unit array.
 * @param pIndex Pointer to the index array.
 * @param pData Pointer to the data array.
 * @param pDataWeight Pointer to the data weight array.
 * @return The calculated indexed L1 error.
 */
template<typename T>
Float kCalculateIndexedL1Error(uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit,
                               uint32_t* pIndex, T* pData, Float* pDataWeight);

/**
 * Calculates the L2 error.
 *
 * @tparam T The data type.
 * @param position The position value.
 * @param batch The batch value.
 * @param stride The stride value.
 * @param pUnit Pointer to the unit array.
 * @param pData Pointer to the data array.
 * @param pDataWeight Pointer to the data weight array.
 * @return The calculated L2 error.
 */
template<typename T>
Float kCalculateL2Error(uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit,
                        T* pData, Float* pDataWeight);

/**
 * Calculates the indexed L2 error.
 *
 * @tparam T The data type.
 * @param position The position value.
 * @param batch The batch value.
 * @param stride The stride value.
 * @param pUnit Pointer to the unit array.
 * @param pIndex Pointer to the index array.
 * @param pData Pointer to the data array.
 * @param pDataWeight Pointer to the data weight array.
 * @return The calculated indexed L2 error.
 */
template<typename T>
Float kCalculateIndexedL2Error(uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit,
                               uint32_t* pIndex, T* pData, Float* pDataWeight);

/**
 * Calculates the L2 hinge error.
 *
 * @tparam T The data type.
 * @param position The position value.
 * @param batch The batch value.
 * @param stride The stride value.
 * @param pUnit Pointer to the unit array.
 * @param pData Pointer to the data array.
 * @param pDataWeight Pointer to the data weight array.
 * @return The calculated L2 hinge error.
 */
template<typename T>
Float kCalculateL2HingeError(uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit,
                            T* pData, Float* pDataWeight);

/**
 * Calculates the indexed L2 hinge error.
 *
 * @tparam T The data type.
 * @param position The position value.
 * @param batch The batch value.
 * @param stride The stride value.
 * @param pUnit Pointer to the unit array.
 * @param pIndex Pointer to the index array.
 * @param pData Pointer to the data array.
 * @param pDataWeight Pointer to the data weight array.
 * @return The calculated indexed L2 hinge error.
 */
template<typename T>
Float kCalculateIndexedL2HingeError(uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit,
                                   uint32_t* pIndex, T* pData, Float* pDataWeight);

/**
 * Calculates the cross-entropy error.
 *
 * @tparam T The data type.
 * @param position The position value.
 * @param batch The batch value.
 * @param stride The stride value.
 * @param pUnit Pointer to the unit array.
 * @param pData Pointer to the data array.
 * @param pDataWeight Pointer to the data weight array.
 * @return The calculated cross-entropy error.
 */
template<typename T>
Float kCalculateCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit,
                                  T* pData, Float* pDataWeight);

/**
 * Calculates the indexed cross-entropy error.
 *
 * @tparam T The data type.
 * @param position The position value.
 * @param batch The batch value.
 * @param stride The stride value.
 * @param pUnit Pointer to the unit array.
 * @param pIndex Pointer to the index array.
 * @param pData Pointer to the data array.
 * @param pDataWeight Pointer to the data weight array.
 * @return The calculated indexed cross-entropy error.
 */
template<typename T>
Float kCalculateIndexedCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit,
                                         uint32_t* pIndex, T* pData, Float* pDataWeight);

/**
 * Calculates the scaled marginal cross-entropy error.
 *
 * @tparam T The data type.
 * @param position The position value.
 * @param batch The batch value.
 * @param stride The stride value.
 * @param pUnit Pointer to the unit array.
 * @param pData Pointer to the data array.
 * @param pDataWeight Pointer to the data weight array.
 * @return The calculated scaled marginal cross-entropy error.
 */
template<typename T>
Float kCalculateScaledMarginalCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride,
                                                Float* pUnit, T* pData, Float* pDataWeight);

/**
 * Calculates the indexed scaled marginal cross-entropy error.
 *
 * @tparam T The data type.
 * @param position The position value.
 * @param batch The batch value.
 * @param stride The stride value.
 * @param pUnit Pointer to the unit array.
 * @param pIndex Pointer to the index array.
 * @param pData Pointer to the data array.
 * @param pDataWeight Pointer to the data weight array.
 * @return The calculated indexed scaled marginal cross-entropy error.
 */
template<typename T>
Float kCalculateIndexedScaledMarginalCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride,
                                                       Float* pUnit, uint32_t* pIndex, T* pData,
                                                       Float* pDataWeight);

/**
 * Calculates the multinomial cross-entropy error.
 *
 * @tparam T The data type.
 * @param position The position value.
 * @param batch The batch value.
 * @param stride The stride value.
 * @param pUnit Pointer to the unit array.
 * @param pData Pointer to the data array.
 * @param pDataWeight Pointer to the data weight array.
 * @return The calculated multinomial cross-entropy error.
 */
template<typename T>
Float kCalculateMultinomialCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride,
                                             Float* pUnit, T* pData, Float* pDataWeight);

/**
 * Calculates the indexed multinomial cross-entropy error.
 *
 * @tparam T The data type.
 * @param position The position value.
 * @param batch The batch value.
 * @param stride The stride value.
 * @param pUnit Pointer to the unit array.
 * @param pIndex Pointer to the index array.
 * @param pData Pointer to the data array.
 * @param pDataWeight Pointer to the data weight array.
 * @return The calculated indexed multinomial cross-entropy error.
 */
template<typename T>
Float kCalculateIndexedMultinomialCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride,
                                                    Float* pUnit, uint32_t* pIndex, T* pData,
                                                    Float* pDataWeight);

/**
 * Calculates the multinomial scaled marginal cross-entropy error.
 *
 * @tparam T The data type.
 * @param position The position value.
 * @param batch The batch value.
 * @param stride The stride value.
 * @param pUnit Pointer to the unit array.
 * @param pData Pointer to the data array.
 * @param pDataWeight Pointer to the data weight array.
 * @return The calculated multinomial scaled marginal cross-entropy error.
 */
template<typename T>
Float kCalculateMultinomialScaledMarginalCrossEntropyError(uint32_t position, uint32_t batch,
                                                           uint32_t stride, Float* pUnit,
                                                           T* pData, Float* pDataWeight);

/**
 * Calculates the indexed multinomial scaled marginal cross-entropy error.
 *
 * @tparam T The data type.
 * @param position The position value.
 * @param batch The batch value.
 * @param stride The stride value.
 * @param pUnit Pointer to the unit array.
 * @param pIndex Pointer to the index array.
 * @param pData Pointer to the data array.
 * @param pDataWeight Pointer to the data weight array.
 * @return The calculated indexed multinomial scaled marginal cross-entropy error.
 */
template<typename T>
Float kCalculateIndexedMultinomialScaledMarginalCrossEntropyError(uint32_t position, uint32_t batch,
                                                                  uint32_t stride, Float* pUnit,
                                                                  uint32_t* pIndex, T* pData,
                                                                  Float* pDataWeight);

/**
 * Calculates the hinge error.
 *
 * @tparam T The data type.
 * @param position The position value.
 * @param batch The batch value.
 * @param stride The stride value.
 * @param pUnit Pointer to the unit array.
 * @param pData Pointer to the data array.
 * @param pDataWeight Pointer to the data weight array.
 * @return The calculated hinge error.
 */
template<typename T>
Float kCalculateHingeError(uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit,
                           T* pData, Float* pDataWeight);

/**
 * Calculates the indexed hinge error.
 *
 * @tparam T The data type.
 * @param position The position value.
 * @param batch The batch value.
 * @param stride The stride value.
 * @param pUnit Pointer to the unit array.
 * @param pIndex Pointer to the index array.
 * @param pData Pointer to the data array.
 * @param pDataWeight Pointer to the data weight array.
 * @return The calculated indexed hinge error.
 */
template<typename T>
Float kCalculateIndexedHingeError(uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit,
                                  uint32_t* pIndex, T* pData, Float* pDataWeight);


/**
 * Calculates the sparse L1 error.
 *
 * @param position The position.
 * @param batch The batch size.
 * @param stride The stride.
 * @param pUnit Pointer to the unit.
 * @param pSparseStart Pointer to the start of sparse data.
 * @param pSparseEnd Pointer to the end of sparse data.
 * @param pSparseIndex Pointer to the sparse index.
 * @param pDataWeight Pointer to the data weight.
 * @param bSparseIgnoreZero Boolean value indicating whether to ignore zero values in sparse data.
 * @return The sparse L1 error.
 */
Float kCalculateSparseL1Error(uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, Float* pDataWeight, bool bSparseIgnoreZero);

/**
 * Calculates the indexed sparse L1 error.
 *
 * @param position The position.
 * @param batch The batch size.
 * @param stride The stride.
 * @param pUnit Pointer to the unit.
 * @param pIndex Pointer to the index.
 * @param pSparseStart Pointer to the start of sparse data.
 * @param pSparseEnd Pointer to the end of sparse data.
 * @param pSparseIndex Pointer to the sparse index.
 * @param pDataWeight Pointer to the data weight.
 * @param bSparseIgnoreZero Boolean value indicating whether to ignore zero values in sparse data.
 * @return The indexed sparse L1 error.
 */
Float kCalculateIndexedSparseL1Error(uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, Float* pDataWeight, bool bSparseIgnoreZero);

/**
 * Calculates the sparse L2 error.
 *
 * @param position The position.
 * @param batch The batch size.
 * @param stride The stride.
 * @param pUnit Pointer to the unit.
 * @param pSparseStart Pointer to the start of sparse data.
 * @param pSparseEnd Pointer to the end of sparse data.
 * @param pSparseIndex Pointer to the sparse index.
 * @param pDataWeight Pointer to the data weight.
 * @param bSparseIgnoreZero Boolean value indicating whether to ignore zero values in sparse data.
 * @return The sparse L2 error.
 */
Float kCalculateSparseL2Error(uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, Float* pDataWeight, bool bSparseIgnoreZero);

/**
 * Calculates the indexed sparse L2 error.
 *
 * @param position The position.
 * @param batch The batch size.
 * @param stride The stride.
 * @param pUnit Pointer to the unit.
 * @param pIndex Pointer to the index.
 * @param pSparseStart Pointer to the start of sparse data.
 * @param pSparseEnd Pointer to the end of sparse data.
 * @param pSparseIndex Pointer to the sparse index.
 * @param pDataWeight Pointer to the data weight.
 * @param bSparseIgnoreZero Boolean value indicating whether to ignore zero values in sparse data.
 * @return The indexed sparse L2 error.
 */
Float kCalculateIndexedSparseL2Error(uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, Float* pDataWeight, bool bSparseIgnoreZero);

/**
 * Calculates the sparse L2 hinge error.
 *
 * @param position The position.
 * @param batch The batch size.
 * @param stride The stride.
 * @param pUnit Pointer to the unit.
 * @param pSparseStart Pointer to the start of sparse data.
 * @param pSparseEnd Pointer to the end of sparse data.
 * @param pSparseIndex Pointer to the sparse index.
 * @param pDataWeight Pointer to the data weight.
 * @param bSparseIgnoreZero Boolean value indicating whether to ignore zero values in sparse data.
 * @return The sparse L2 hinge error.
 */
Float kCalculateSparseL2HingeError(uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, Float* pDataWeight, bool bSparseIgnoreZero);

/**
 * Calculates the indexed sparse L2 hinge error.
 *
 * @param position The position.
 * @param batch The batch size.
 * @param stride The stride.
 * @param pUnit Pointer to the unit.
 * @param pIndex Pointer to the index.
 * @param pSparseStart Pointer to the start of sparse data.
 * @param pSparseEnd Pointer to the end of sparse data.
 * @param pSparseIndex Pointer to the sparse index.
 * @param pDataWeight Pointer to the data weight.
 * @param bSparseIgnoreZero Boolean value indicating whether to ignore zero values in sparse data.
 * @return The indexed sparse L2 hinge error.
 */
Float kCalculateIndexedSparseL2HingeError(uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, Float* pDataWeight, bool bSparseIgnoreZero);

/**
 * Calculates the sparse cross-entropy error.
 *
 * @param position The position.
 * @param batch The batch size.
 * @param stride The stride.
 * @param pUnit Pointer to the unit.
 * @param pSparseStart Pointer to the start of sparse data.
 * @param pSparseEnd Pointer to the end of sparse data.
 * @param pSparseIndex Pointer to the sparse index.
 * @param pDataWeight Pointer to the data weight.
 * @param bSparseIgnoreZero Boolean value indicating whether to ignore zero values in sparse data.
 * @return The sparse cross-entropy error.
 */
Float kCalculateSparseCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, Float* pDataWeight, bool bSparseIgnoreZero);

/**
 * Calculates the indexed sparse cross-entropy error.
 *
 * @param position The position.
 * @param batch The batch size.
 * @param stride The stride.
 * @param pUnit Pointer to the unit.
 * @param pIndex Pointer to the index.
 * @param pSparseStart Pointer to the start of sparse data.
 * @param pSparseEnd Pointer to the end of sparse data.
 * @param pSparseIndex Pointer to the sparse index.
 * @param pDataWeight Pointer to the data weight.
 * @param bSparseIgnoreZero Boolean value indicating whether to ignore zero values in sparse data.
 * @return The indexed sparse cross-entropy error.
 */
Float kCalculateIndexedSparseCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, Float* pDataWeight, bool bSparseIgnoreZero);

/**
 * Calculates the sparse scaled marginal cross-entropy error.
 *
 * @param position The position.
 * @param batch The batch size.
 * @param stride The stride.
 * @param pUnit Pointer to the unit.
 * @param pSparseStart Pointer to the start of sparse data.
 * @param pSparseEnd Pointer to the end of sparse data.
 * @param pSparseIndex Pointer to the sparse index.
 * @param pDataWeight Pointer to the data weight.
 * @param bSparseIgnoreZero Boolean value indicating whether to ignore zero values in sparse data.
 * @return The sparse scaled marginal cross-entropy error.
 */
Float kCalculateSparseScaledMarginalCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, Float* pDataWeight, bool bSparseIgnoreZero);

/**
 * Calculates the indexed sparse scaled marginal cross-entropy error.
 *
 * @param position The position.
 * @param batch The batch size.
 * @param stride The stride.
 * @param pUnit Pointer to the unit.
 * @param pIndex Pointer to the index.
 * @param pSparseStart Pointer to the start of sparse data.
 * @param pSparseEnd Pointer to the end of sparse data.
 * @param pSparseIndex Pointer to the sparse index.
 * @param pDataWeight Pointer to the data weight.
 * @param bSparseIgnoreZero Boolean value indicating whether to ignore zero values in sparse data.
 * @return The indexed sparse scaled marginal cross-entropy error.
 */
Float kCalculateIndexedSparseScaledMarginalCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, Float* pDataWeight, bool bSparseIgnoreZero);

/**
 * Calculates the sparse multinomial cross-entropy error.
 *
 * @param position The position.
 * @param batch The batch size.
 * @param stride The stride.
 * @param pUnit Pointer to the unit.
 * @param pSparseStart Pointer to the start of sparse data.
 * @param pSparseEnd Pointer to the end of sparse data.
 * @param pSparseIndex Pointer to the sparse index.
 * @param pDataWeight Pointer to the data weight.
 * @return The sparse multinomial cross-entropy error.
 */
Float kCalculateSparseMultinomialCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, Float* pDataWeight);

/**
 * Calculates the indexed sparse multinomial cross-entropy error.
 *
 * @param position The position.
 * @param batch The batch size.
 * @param stride The stride.
 * @param pUnit Pointer to the unit.
 * @param pIndex Pointer to the index.
 * @param pSparseStart Pointer to the start of sparse data.
 * @param pSparseEnd Pointer to the end of sparse data.
 * @param pSparseIndex Pointer to the sparse index.
 * @param pDataWeight Pointer to the data weight.
 * @return The indexed sparse multinomial cross-entropy error.
 */
Float kCalculateIndexedSparseMultinomialCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, Float* pDataWeight);

/**
 * Calculates the sparse multinomial scaled marginal cross-entropy error.
 *
 * @param position The position.
 * @param batch The batch size.
 * @param stride The stride.
 * @param pUnit Pointer to the unit.
 * @param pSparseStart Pointer to the start of sparse data.
 * @param pSparseEnd Pointer to the end of sparse data.
 * @param pSparseIndex Pointer to the sparse index.
 * @param pDataWeight Pointer to the data weight.
 * @return The sparse multinomial scaled marginal cross-entropy error.
 */
Float kCalculateSparseMultinomialScaledMarginalCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, Float* pDataWeight);

/**
 * Calculates the indexed sparse multinomial scaled marginal cross-entropy error.
 *
 * @param position The position.
 * @param batch The batch size.
 * @param stride The stride.
 * @param pUnit Pointer to the unit.
 * @param pIndex Pointer to the index.
 * @param pSparseStart Pointer to the start of sparse data.
 * @param pSparseEnd Pointer to the end of sparse data.
 * @param pSparseIndex Pointer to the sparse index.
 * @param pDataWeight Pointer to the data weight.
 * @return The indexed sparse multinomial scaled marginal cross-entropy error.
 */
Float kCalculateIndexedSparseMultinomialScaledMarginalCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, Float* pDataWeight);

/**
 * Calculates the sparse analog L1 error.
 *
 * @tparam T The data type.
 * @param position The position.
 * @param batch The batch size.
 * @param stride The stride.
 * @param pUnit Pointer to the unit.
 * @param pSparseStart Pointer to the start of sparse data.
 * @param pSparseEnd Pointer to the end of sparse data.
 * @param pSparseIndex Pointer to the sparse index.
 * @param pDataWeight Pointer to the data weight.
 * @param pSparseData Pointer to the sparse data.
 * @param bSparseIgnoreZero Boolean value indicating whether to ignore zero values in sparse data.
 * @return The sparse analog L1 error.
 */
template<typename T>
Float kCalculateSparseAnalogL1Error(uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, Float* pDataWeight, T* pSparseData, bool bSparseIgnoreZero);

/**
 * Calculates the indexed sparse analog L1 error.
 *
 * @tparam T The data type.
 * @param position The position.
 * @param batch The batch size.
 * @param stride The stride.
 * @param pUnit Pointer to the unit.
 * @param pIndex Pointer to the index.
 * @param pSparseStart Pointer to the start of sparse data.
 * @param pSparseEnd Pointer to the end of sparse data.
 * @param pSparseIndex Pointer to the sparse index.
 * @param pDataWeight Pointer to the data weight.
 * @param pSparseData Pointer to the sparse data.
 * @param bSparseIgnoreZero Boolean value indicating whether to ignore zero values in sparse data.
 * @return The indexed sparse analog L1 error.
 */
template<typename T>
Float kCalculateIndexedSparseAnalogL1Error(uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, Float* pDataWeight, T* pSparseData, bool bSparseIgnoreZero);

/**
 * Calculates the sparse analog L2 error.
 *
 * @tparam T The data type.
 * @param position The position.
 * @param batch The batch size.
 * @param stride The stride.
 * @param pUnit Pointer to the unit.
 * @param pSparseStart Pointer to the start of sparse data.
 * @param pSparseEnd Pointer to the end of sparse data.
 * @param pSparseIndex Pointer to the sparse index.
 * @param pDataWeight Pointer to the data weight.
 * @param pSparseData Pointer to the sparse data.
 * @param bSparseIgnoreZero Boolean value indicating whether to ignore zero values in sparse data.
 * @return The sparse analog L2 error.
 */
template<typename T>
Float kCalculateSparseAnalogL2Error(uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, Float* pDataWeight, T* pSparseData, bool bSparseIgnoreZero);

/**
 * Calculates the indexed sparse analog L2 error.
 *
 * @tparam T The data type.
 * @param position The position.
 * @param batch The batch size.
 * @param stride The stride.
 * @param pUnit Pointer to the unit.
 * @param pIndex Pointer to the index.
 * @param pSparseStart Pointer to the start of sparse data.
 * @param pSparseEnd Pointer to the end of sparse data.
 * @param pSparseIndex Pointer to the sparse index.
 * @param pDataWeight Pointer to the data weight.
 * @param pSparseData Pointer to the sparse data.
 * @param bSparseIgnoreZero Boolean value indicating whether to ignore zero values in sparse data.
 * @return The indexed sparse analog L2 error.
 */
template<typename T>
Float kCalculateIndexedSparseAnalogL2Error(uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, Float* pDataWeight, T* pSparseData, bool bSparseIgnoreZero);

/**
 * Calculates the sparse analog L2 hinge error.
 *
 * @tparam T The data type.
 * @param position The position.
 * @param batch The batch size.
 * @param stride The stride.
 * @param pUnit Pointer to the unit.
 * @param pSparseStart Pointer to the start of sparse data.
 * @param pSparseEnd Pointer to the end of sparse data.
 * @param pSparseIndex Pointer to the sparse index.
 * @param pDataWeight Pointer to the data weight.
 * @param pSparseData Pointer to the sparse data.
 * @param bSparseIgnoreZero Boolean value indicating whether to ignore zero values in sparse data.
 * @return The sparse analog L2 hinge error.
 */
template<typename T>
Float kCalculateSparseAnalogL2HingeError(uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, Float* pDataWeight, T* pSparseData, bool bSparseIgnoreZero);

/**
 * Calculates the indexed sparse analog L2 hinge error.
 *
 * @tparam T The data type.
 * @param position The position.
 * @param batch The batch size.
 * @param stride The stride.
 * @param pUnit Pointer to the unit.
 * @param pIndex Pointer to the index.
 * @param pSparseStart Pointer to the start of sparse data.
 * @param pSparseEnd Pointer to the end of sparse data.
 * @param pSparseIndex Pointer to the sparse index.
 * @param pDataWeight Pointer to the data weight.
 * @param pSparseData Pointer to the sparse data.
 * @param bSparseIgnoreZero Boolean value indicating whether to ignore zero values in sparse data.
 * @return The indexed sparse analog L2 hinge error.
 */
template<typename T>
Float kCalculateIndexedSparseAnalogL2HingeError(uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, Float* pDataWeight, T* pSparseData, bool bSparseIgnoreZero);

/**
 * Calculates the sparse analog cross-entropy error.
 *
 * @tparam T The data type.
 * @param position The position.
 * @param batch The batch size.
 * @param stride The stride.
 * @param pUnit Pointer to the unit.
 * @param pSparseStart Pointer to the start of sparse data.
 * @param pSparseEnd Pointer to the end of sparse data.
 * @param pSparseIndex Pointer to the sparse index.
 * @param pDataWeight Pointer to the data weight.
 * @param pSparseData Pointer to the sparse data.
 * @param bSparseIgnoreZero Boolean value indicating whether to ignore zero values in sparse data.
 * @return The sparse analog cross-entropy error.
 */
template<typename T>
Float kCalculateSparseAnalogCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, Float* pDataWeight, T* pSparseData, bool bSparseIgnoreZero);

/**
 * Calculates the indexed sparse analog cross-entropy error.
 *
 * @tparam T The data type.
 * @param position The position.
 * @param batch The batch size.
 * @param stride The stride.
 * @param pUnit Pointer to the unit.
 * @param pIndex Pointer to the index.
 * @param pSparseStart Pointer to the start of sparse data.
 * @param pSparseEnd Pointer to the end of sparse data.
 * @param pSparseIndex Pointer to the sparse index.
 * @param pDataWeight Pointer to the data weight.
 * @param pSparseData Pointer to the sparse data.
 * @param bSparseIgnoreZero Boolean value indicating whether to ignore zero values in sparse data.
 * @return The indexed sparse analog cross-entropy error.
 */
template<typename T>
Float kCalculateIndexedSparseAnalogCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, Float* pDataWeight, T* pSparseData, bool bSparseIgnoreZero);

/**
 * Calculates the sparse analog scaled marginal cross-entropy error.
 *
 * @tparam T The data type.
 * @param position The position.
 * @param batch The batch size.
 * @param stride The stride.
 * @param pUnit Pointer to the unit.
 * @param pSparseStart Pointer to the start of sparse data.
 * @param pSparseEnd Pointer to the end of sparse data.
 * @param pSparseIndex Pointer to the sparse index.
 * @param pDataWeight Pointer to the data weight.
 * @param pSparseData Pointer to the sparse data.
 * @param bSparseIgnoreZero Boolean value indicating whether to ignore zero values in sparse data.
 * @return The sparse analog scaled marginal cross-entropy error.
 */
template<typename T>
Float kCalculateSparseAnalogScaledMarginalCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, Float* pDataWeight, T* pSparseData, bool bSparseIgnoreZero);

/**
 * Calculates the indexed sparse analog scaled marginal cross-entropy error.
 *
 * @tparam T The data type.
 * @param position The position.
 * @param batch The batch size.
 * @param stride The stride.
 * @param pUnit Pointer to the unit.
 * @param pIndex Pointer to the index.
 * @param pSparseStart Pointer to the start of sparse data.
 * @param pSparseEnd Pointer to the end of sparse data.
 * @param pSparseIndex Pointer to the sparse index.
 * @param pDataWeight Pointer to the data weight.
 * @param pSparseData Pointer to the sparse data.
 * @param bSparseIgnoreZero Boolean value indicating whether to ignore zero values in sparse data.
 * @return The indexed sparse analog scaled marginal cross-entropy error.
 */
template<typename T>
Float kCalculateIndexedSparseAnalogScaledMarginalCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, Float* pDataWeight, T* pSparseData, bool bSparseIgnoreZero);

/**
 * Calculates the sparse analog multinomial cross-entropy error.
 *
 * @tparam T The data type.
 * @param position The position.
 * @param batch The batch size.
 * @param stride The stride.
 * @param pUnit Pointer to the unit.
 * @param pSparseStart Pointer to the start of sparse data.
 * @param pSparseEnd Pointer to the end of sparse data.
 * @param pSparseIndex Pointer to the sparse index.
 * @param pDataWeight Pointer to the data weight.
 * @param pSparseData Pointer to the sparse data.
 * @return The sparse analog multinomial cross-entropy error.
 */
template<typename T>
Float kCalculateSparseAnalogMultinomialCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, Float* pDataWeight, T* pSparseData);

/**
 * Calculates the indexed sparse analog multinomial cross-entropy error.
 *
 * @tparam T The data type.
 * @param position The position.
 * @param batch The batch size.
 * @param stride The stride.
 * @param pUnit Pointer to the unit.
 * @param pIndex Pointer to the index.
 * @param pSparseStart Pointer to the start of sparse data.
 * @param pSparseEnd Pointer to the end of sparse data.
 * @param pSparseIndex Pointer to the sparse index.
 * @param pDataWeight Pointer to the data weight.
 * @param pSparseData Pointer to the sparse data.
 * @return The indexed sparse analog multinomial cross-entropy error.
 */
template<typename T>
Float kCalculateIndexedSparseAnalogMultinomialCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, Float* pDataWeight, T* pSparseData);

/**
 * Calculates the sparse analog multinomial scaled marginal cross-entropy error.
 *
 * @tparam T The data type.
 * @param position The position.
 * @param batch The batch size.
 * @param stride The stride.
 * @param pUnit Pointer to the unit.
 * @param pSparseStart Pointer to the start of sparse data.
 * @param pSparseEnd Pointer to the end of sparse data.
 * @param pSparseIndex Pointer to the sparse index.
 * @param pDataWeight Pointer to the data weight.
 * @param pSparseData Pointer to the sparse data.
 * @return The sparse analog multinomial scaled marginal cross-entropy error.
 */
template<typename T>
Float kCalculateSparseAnalogMultinomialScaledMarginalCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, Float* pDataWeight, T* pSparseData);

/**
 * Calculates the indexed sparse analog multinomial scaled marginal cross-entropy error.
 *
 * @tparam T The data type.
 * @param position The position.
 * @param batch The batch size.
 * @param stride The stride.
 * @param pUnit Pointer to the unit.
 * @param pIndex Pointer to the index.
 * @param pSparseStart Pointer to the start of sparse data.
 * @param pSparseEnd Pointer to the end of sparse data.
 * @param pSparseIndex Pointer to the sparse index.
 * @param pDataWeight Pointer to the data weight.
 * @param pSparseData Pointer to the sparse data.
 * @return The indexed sparse analog multinomial scaled marginal cross-entropy error.
 */
template<typename T>
Float kCalculateIndexedSparseAnalogMultinomialScaledMarginalCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, Float* pDataWeight, T* pSparseData);

/**
 * Calculates the sparse data scaled marginal cross-entropy error.
 *
 * @tparam T The data type.
 * @param position The position.
 * @param batch The batch size.
 * @param stride The stride.
 * @param pUnit Pointer to the unit.
 * @param pSparseStart Pointer to the start of sparse data.
 * @param pSparseEnd Pointer to the end of sparse data.
 * @param pSparseIndex Pointer to the sparse index.
 * @param pSparseData Pointer to the sparse data.
 * @param bSparseIgnoreZero Boolean value indicating whether to ignore zero values in sparse data.
 * @return The sparse data scaled marginal cross-entropy error.
 */
template<typename T>
Float kCalculateSparseDataScaledMarginalCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, T* pSparseData, bool bSparseIgnoreZero);

/**
 * Calculates the indexed sparse data scaled marginal cross-entropy error.
 *
 * @tparam T The data type.
 * @param position The position.
 * @param batch The batch size.
 * @param stride The stride.
 * @param pUnit Pointer to the unit.
 * @param pIndex Pointer to the index.
 * @param pSparseStart Pointer to the start of sparse data.
 * @param pSparseEnd Pointer to the end of sparse data.
 * @param pSparseIndex Pointer to the sparse index.
 * @param pSparseData Pointer to the sparse data.
 * @param bSparseIgnoreZero Boolean value indicating whether to ignore zero values in sparse data.
 * @return The indexed sparse data scaled marginal cross-entropy error.
 */
template<typename T>
Float kCalculateIndexedSparseDataScaledMarginalCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, T* pSparseData, bool bSparseIgnoreZero);


/**
 * @brief Calculates the regularization error.
 *
 * @param lambda The regularization parameter.
 * @param lambda1 The L1 regularization parameter.
 * @param pWeight Pointer to the weight array.
 * @param size The size of the weight array.
 * @return The regularization error.
 */
Float kCalculateRegularizationError(Float lambda, Float lambda1, Float* pWeight, uint64_t size);

/**
 * @brief Normalizes the weights.
 *
 * @param norm The normalization factor.
 * @param outputStride The output stride.
 * @param inputStride The input stride.
 * @param pWeight Pointer to the weight array.
 */
void kNormalizeWeights(Float norm, uint32_t outputStride, uint32_t inputStride, Float* pWeight);

/**
 * @brief Calculates the magnitudes of the weights.
 *
 * @param outputStride The output stride.
 * @param inputStride The input stride.
 * @param pWeight Pointer to the weight array.
 * @param pMagnitude Pointer to store the magnitudes.
 */
void kCalculateWeightMagnitudes(uint32_t outputStride, uint32_t inputStride, Float* pWeight, Float* pMagnitude);

/**
 * @brief Normalizes the magnitudes of the weights.
 *
 * @param norm The normalization factor.
 * @param outputStride The output stride.
 * @param inputStride The input stride.
 * @param pWeight Pointer to the weight array.
 * @param pMagnitude Pointer to the magnitude array.
 */
void kNormalizeWeightMagnitudes(Float norm, uint32_t outputStride, uint32_t inputStride, Float* pWeight, Float* pMagnitude);

/**
 * @brief Normalizes the deltas.
 *
 * @param norm The normalization factor.
 * @param batch The batch size.
 * @param stride The stride.
 * @param pDelta Pointer to the delta array.
 */
void kNormalizeDeltas(Float norm, uint32_t batch, uint32_t stride, Float* pDelta);

/**
 * @brief Calculates the magnitudes of the deltas.
 *
 * @param batch The batch size.
 * @param stride The stride.
 * @param pDelta Pointer to the delta array.
 * @param pMagnitude Pointer to store the magnitudes.
 */
void kCalculateDeltaMagnitudes(uint32_t batch, uint32_t stride, Float* pDelta, Float* pMagnitude);

/**
 * @brief Normalizes the magnitudes of the deltas.
 *
 * @param norm The normalization factor.
 * @param batch The batch size.
 * @param stride The stride.
 * @param pDelta Pointer to the delta array.
 * @param pMagnitude Pointer to the magnitude array.
 */
void kNormalizeDeltaMagnitudes(Float norm, uint32_t batch, uint32_t stride, Float* pDelta, Float* pMagnitude);

/**
 * @brief Calculates scaled and biased dropout.
 *
 * @param pUnit Pointer to the unit array.
 * @param pRandom Pointer to the random array.
 * @param batch The batch size.
 * @param stride The stride.
 * @param p The dropout probability.
 * @param target The target value.
 * @param a The scaling factor.
 * @param b The bias factor.
 */
void kCalculateScaledBiasedDropout(Float* pUnit, Float* pRandom, uint32_t batch, uint32_t stride, Float p, Float target, Float a, Float b);

/**
 * @brief Calculates dropout.
 *
 * @param pUnit Pointer to the unit array.
 * @param pRandom Pointer to the random array.
 * @param batch The batch size.
 * @param stride The stride.
 * @param p The dropout probability.
 * @param target The target value.
 */
void kCalculateDropout(Float* pUnit, Float* pRandom, uint32_t batch, uint32_t stride, Float p, Float target);


/**
 * Calculates the L1 output delta.
 *
 * @tparam T The type of data.
 * @param activation The activation function.
 * @param position The position.
 * @param batch The batch size.
 * @param stride The stride.
 * @param pUnit Pointer to the unit data.
 * @param pDelta Pointer to the delta data.
 * @param pData Pointer to the data.
 * @param pDataWeight Pointer to the data weight.
 * @param slope The slope value.
 * @param alpha The alpha value.
 * @param lambda The lambda value.
 */
template<typename T>
void kCalculateL1OutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit, Float* pDelta, T* pData, Float* pDataWeight, Float slope, Float alpha, Float lambda);

/**
 * Calculates the indexed L1 output delta.
 *
 * @tparam T The type of data.
 * @param activation The activation function.
 * @param position The position.
 * @param batch The batch size.
 * @param stride The stride.
 * @param pUnit Pointer to the unit data.
 * @param pDelta Pointer to the delta data.
 * @param pIndex Pointer to the index data.
 * @param pData Pointer to the data.
 * @param pDataWeight Pointer to the data weight.
 * @param slope The slope value.
 * @param alpha The alpha value.
 * @param lambda The lambda value.
 */
template<typename T>
void kCalculateIndexedL1OutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit, Float* pDelta, uint32_t* pIndex, T* pData, Float* pDataWeight, Float slope, Float alpha, Float lambda);

/**
 * Calculates the cross entropy output delta.
 *
 * @tparam T The type of data.
 * @param activation The activation function.
 * @param position The position.
 * @param batch The batch size.
 * @param stride The stride.
 * @param pUnit Pointer to the unit data.
 * @param pDelta Pointer to the delta data.
 * @param pData Pointer to the data.
 * @param pDataWeight Pointer to the data weight.
 */
template<typename T>
void kCalculateCrossEntropyOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit, Float* pDelta, T* pData, Float* pDataWeight);

/**
 * Calculates the indexed cross entropy output delta.
 *
 * @tparam T The type of data.
 * @param activation The activation function.
 * @param position The position.
 * @param batch The batch size.
 * @param stride The stride.
 * @param pUnit Pointer to the unit data.
 * @param pDelta Pointer to the delta data.
 * @param pIndex Pointer to the index data.
 * @param pData Pointer to the data.
 * @param pDataWeight Pointer to the data weight.
 */
template<typename T>
void kCalculateIndexedCrossEntropyOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit, Float* pDelta, uint32_t* pIndex, T* pData, Float* pDataWeight);

/**
 * Calculates the scaled marginal cross entropy output delta.
 *
 * @tparam T The type of data.
 * @param activation The activation function.
 * @param position The position.
 * @param batch The batch size.
 * @param stride The stride.
 * @param pUnit Pointer to the unit data.
 * @param pDelta Pointer to the delta data.
 * @param pData Pointer to the data.
 * @param pDataWeight Pointer to the data weight.
 */
template<typename T>
void kCalculateScaledMarginalCrossEntropyOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit, Float* pDelta, T* pData, Float* pDataWeight);

/**
 * Calculates the indexed scaled marginal cross entropy output delta.
 *
 * @tparam T The type of data.
 * @param activation The activation function.
 * @param position The position.
 * @param batch The batch size.
 * @param stride The stride.
 * @param pUnit Pointer to the unit data.
 * @param pDelta Pointer to the delta data.
 * @param pIndex Pointer to the index data.
 * @param pData Pointer to the data.
 * @param pDataWeight Pointer to the data weight.
 */
template<typename T>
void kCalculateIndexedScaledMarginalCrossEntropyOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit, Float* pDelta, uint32_t* pIndex, T* pData, Float* pDataWeight);

/**
 * Calculates the output delta.
 *
 * @tparam T The type of data.
 * @param activation The activation function.
 * @param position The position.
 * @param batch The batch size.
 * @param stride The stride.
 * @param pUnit Pointer to the unit data.
 * @param pDelta Pointer to the delta data.
 * @param pData Pointer to the data.
 * @param pDataWeight Pointer to the data weight.
 * @param slope The slope value.
 * @param alpha The alpha value.
 * @param lambda The lambda value.
 */
template<typename T>
void kCalculateOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit, Float* pDelta, T* pData, Float* pDataWeight, Float slope, Float alpha, Float lambda);

/**
 * Calculates the indexed output delta.
 *
 * @tparam T The type of data.
 * @param activation The activation function.
 * @param position The position.
 * @param batch The batch size.
 * @param stride The stride.
 * @param pUnit Pointer to the unit data.
 * @param pDelta Pointer to the delta data.
 * @param pIndex Pointer to the index data.
 * @param pData Pointer to the data.
 * @param pDataWeight Pointer to the data weight.
 * @param slope The slope value.
 * @param alpha The alpha value.
 * @param lambda The lambda value.
 */
template<typename T>
void kCalculateIndexedOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit, Float* pDelta, uint32_t* pIndex, T* pData, Float* pDataWeight, Float slope, Float alpha, Float lambda);

/**
 * Calculates the L2 hinge output delta.
 *
 * @tparam T The type of data.
 * @param activation The activation function.
 * @param position The position.
 * @param batch The batch size.
 * @param stride The stride.
 * @param pUnit Pointer to the unit data.
 * @param pDelta Pointer to the delta data.
 * @param pData Pointer to the data.
 * @param pDataWeight Pointer to the data weight.
 * @param slope The slope value.
 * @param alpha The alpha value.
 * @param lambda The lambda value.
 */
template<typename T>
void kCalculateL2HingeOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit, Float* pDelta, T* pData, Float* pDataWeight, Float slope, Float alpha, Float lambda);

/**
 * Calculates the indexed L2 hinge output delta.
 *
 * @tparam T The type of data.
 * @param activation The activation function.
 * @param position The position.
 * @param batch The batch size.
 * @param stride The stride.
 * @param pUnit Pointer to the unit data.
 * @param pDelta Pointer to the delta data.
 * @param pIndex Pointer to the index data.
 * @param pData Pointer to the data.
 * @param pDataWeight Pointer to the data weight.
 * @param slope The slope value.
 * @param alpha The alpha value.
 * @param lambda The lambda value.
 */
template<typename T>
void kCalculateIndexedL2HingeOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit, Float* pDelta, uint32_t* pIndex, T* pData, Float* pDataWeight, Float slope, Float alpha, Float lambda);

/**
 * Calculates the hinge output delta.
 *
 * @tparam T The type of data.
 * @param activation The activation function.
 * @param position The position.
 * @param batch The batch size.
 * @param stride The stride.
 * @param pUnit Pointer to the unit data.
 * @param pDelta Pointer to the delta data.
 * @param pData Pointer to the data.
 * @param pDataWeight Pointer to the data weight.
 */
template<typename T>
void kCalculateHingeOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit, Float* pDelta, T* pData, Float* pDataWeight);

/**
 * Calculates the indexed hinge output delta.
 *
 * @tparam T The type of data.
 * @param activation The activation function.
 * @param position The position.
 * @param batch The batch size.
 * @param stride The stride.
 * @param pUnit Pointer to the unit data.
 * @param pDelta Pointer to the delta data.
 * @param pIndex Pointer to the index data.
 * @param pData Pointer to the data.
 * @param pDataWeight Pointer to the data weight.
 */
template<typename T>
void kCalculateIndexedHingeOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit, Float* pDelta, uint32_t* pIndex, T* pData, Float* pDataWeight);

/**
 * Calculates the output delta for sparse L1 regularization.
 *
 * @param activation The activation function.
 * @param position The position within the layer.
 * @param batch The batch size.
 * @param stride The stride value.
 * @param pUnit The pointer to the unit values.
 * @param pDelta The pointer to the delta values.
 * @param pSparseStart The pointer to the start indices of sparse data.
 * @param pSparseEnd The pointer to the end indices of sparse data.
 * @param pSparseIndex The pointer to the indices of sparse data.
 * @param pDataWeight The pointer to the weights of data.
 * @param bSparseIgnoreZero Specifies whether to ignore zero values in sparse data.
 * @param slope The slope value.
 * @param alpha The alpha value.
 * @param lambda The lambda value.
 */
void kCalculateSparseL1OutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit, Float* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, Float* pDataWeight, bool bSparseIgnoreZero, Float slope, Float alpha, Float lambda);

/**
 * Calculates the output delta for indexed sparse L1 regularization.
 *
 * @param activation The activation function.
 * @param position The position within the layer.
 * @param batch The batch size.
 * @param stride The stride value.
 * @param pUnit The pointer to the unit values.
 * @param pDelta The pointer to the delta values.
 * @param pIndex The pointer to the indices of sparse data.
 * @param pSparseStart The pointer to the start indices of sparse data.
 * @param pSparseEnd The pointer to the end indices of sparse data.
 * @param pSparseIndex The pointer to the indices of sparse data.
 * @param pDataWeight The pointer to the weights of data.
 * @param bSparseIgnoreZero Specifies whether to ignore zero values in sparse data.
 * @param slope The slope value.
 * @param alpha The alpha value.
 * @param lambda The lambda value.
 */
void kCalculateIndexedSparseL1OutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit, Float* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, Float* pDataWeight, bool bSparseIgnoreZero, Float slope, Float alpha, Float lambda);

/**
 * Calculates the output delta for sparse cross-entropy loss.
 *
 * @param activation The activation function.
 * @param position The position within the layer.
 * @param batch The batch size.
 * @param stride The stride value.
 * @param pUnit The pointer to the unit values.
 * @param pDelta The pointer to the delta values.
 * @param pSparseStart The pointer to the start indices of sparse data.
 * @param pSparseEnd The pointer to the end indices of sparse data.
 * @param pSparseIndex The pointer to the indices of sparse data.
 * @param pDataWeight The pointer to the weights of data.
 * @param bSparseIgnoreZero Specifies whether to ignore zero values in sparse data.
 */
void kCalculateSparseCrossEntropyOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit, Float* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, Float* pDataWeight, bool bSparseIgnoreZero);

/**
 * Calculates the output delta for indexed sparse cross-entropy loss.
 *
 * @param activation The activation function.
 * @param position The position within the layer.
 * @param batch The batch size.
 * @param stride The stride value.
 * @param pUnit The pointer to the unit values.
 * @param pDelta The pointer to the delta values.
 * @param pIndex The pointer to the indices of sparse data.
 * @param pSparseStart The pointer to the start indices of sparse data.
 * @param pSparseEnd The pointer to the end indices of sparse data.
 * @param pSparseIndex The pointer to the indices of sparse data.
 * @param pDataWeight The pointer to the weights of data.
 * @param bSparseIgnoreZero Specifies whether to ignore zero values in sparse data.
 */
void kCalculateIndexedSparseCrossEntropyOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit, Float* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, Float* pDataWeight, bool bSparseIgnoreZero);

/**
 * Calculates the output delta for sparse scaled marginal cross-entropy loss.
 *
 * @param activation The activation function.
 * @param position The position within the layer.
 * @param batch The batch size.
 * @param stride The stride value.
 * @param pUnit The pointer to the unit values.
 * @param pDelta The pointer to the delta values.
 * @param pSparseStart The pointer to the start indices of sparse data.
 * @param pSparseEnd The pointer to the end indices of sparse data.
 * @param pSparseIndex The pointer to the indices of sparse data.
 * @param pDataWeight The pointer to the weights of data.
 * @param bSparseIgnoreZero Specifies whether to ignore zero values in sparse data.
 */
void kCalculateSparseScaledMarginalCrossEntropyOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit, Float* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, Float* pDataWeight, bool bSparseIgnoreZero);

/**
 * Calculates the output delta for indexed sparse scaled marginal cross-entropy loss.
 *
 * @param activation The activation function.
 * @param position The position within the layer.
 * @param batch The batch size.
 * @param stride The stride value.
 * @param pUnit The pointer to the unit values.
 * @param pDelta The pointer to the delta values.
 * @param pIndex The pointer to the indices of sparse data.
 * @param pSparseStart The pointer to the start indices of sparse data.
 * @param pSparseEnd The pointer to the end indices of sparse data.
 * @param pSparseIndex The pointer to the indices of sparse data.
 * @param pDataWeight The pointer to the weights of data.
 * @param bSparseIgnoreZero Specifies whether to ignore zero values in sparse data.
 */
void kCalculateIndexedSparseScaledMarginalCrossEntropyOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit, Float* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, Float* pDataWeight, bool bSparseIgnoreZero);

/**
 * Calculates the output delta for sparse regularization.
 *
 * @param activation The activation function.
 * @param position The position within the layer.
 * @param batch The batch size.
 * @param stride The stride value.
 * @param pUnit The pointer to the unit values.
 * @param pDelta The pointer to the delta values.
 * @param pSparseStart The pointer to the start indices of sparse data.
 * @param pSparseEnd The pointer to the end indices of sparse data.
 * @param pSparseIndex The pointer to the indices of sparse data.
 * @param pDataWeight The pointer to the weights of data.
 * @param bSparseIgnoreZero Specifies whether to ignore zero values in sparse data.
 * @param slope The slope value.
 * @param alpha The alpha value.
 * @param lambda The lambda value.
 */
void kCalculateSparseOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit, Float* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, Float* pDataWeight, bool bSparseIgnoreZero, Float slope, Float alpha, Float lambda);

/**
 * Calculates the output delta for indexed sparse regularization.
 *
 * @param activation The activation function.
 * @param position The position within the layer.
 * @param batch The batch size.
 * @param stride The stride value.
 * @param pUnit The pointer to the unit values.
 * @param pDelta The pointer to the delta values.
 * @param pIndex The pointer to the indices of sparse data.
 * @param pSparseStart The pointer to the start indices of sparse data.
 * @param pSparseEnd The pointer to the end indices of sparse data.
 * @param pSparseIndex The pointer to the indices of sparse data.
 * @param pDataWeight The pointer to the weights of data.
 * @param bSparseIgnoreZero Specifies whether to ignore zero values in sparse data.
 * @param slope The slope value.
 * @param alpha The alpha value.
 * @param lambda The lambda value.
 */
void kCalculateIndexedSparseOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit, Float* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, Float* pDataWeight, bool bSparseIgnoreZero, Float slope, Float alpha, Float lambda);

/**
 * Calculates the output delta for sparse L2 hinge regularization.
 *
 * @param activation The activation function.
 * @param position The position within the layer.
 * @param batch The batch size.
 * @param stride The stride value.
 * @param pUnit The pointer to the unit values.
 * @param pDelta The pointer to the delta values.
 * @param pSparseStart The pointer to the start indices of sparse data.
 * @param pSparseEnd The pointer to the end indices of sparse data.
 * @param pSparseIndex The pointer to the indices of sparse data.
 * @param pDataWeight The pointer to the weights of data.
 * @param bSparseIgnoreZero Specifies whether to ignore zero values in sparse data.
 * @param slope The slope value.
 * @param alpha The alpha value.
 * @param lambda The lambda value.
 */
void kCalculateSparseL2HingeOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit, Float* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, Float* pDataWeight, bool bSparseIgnoreZero, Float slope, Float alpha, Float lambda);

/**
 * Calculates the output delta for indexed sparse L2 hinge regularization.
 *
 * @param activation The activation function.
 * @param position The position within the layer.
 * @param batch The batch size.
 * @param stride The stride value.
 * @param pUnit The pointer to the unit values.
 * @param pDelta The pointer to the delta values.
 * @param pIndex The pointer to the indices of sparse data.
 * @param pSparseStart The pointer to the start indices of sparse data.
 * @param pSparseEnd The pointer to the end indices of sparse data.
 * @param pSparseIndex The pointer to the indices of sparse data.
 * @param pDataWeight The pointer to the weights of data.
 * @param bSparseIgnoreZero Specifies whether to ignore zero values in sparse data.
 * @param slope The slope value.
 * @param alpha The alpha value.
 * @param lambda The lambda value.
 */
void kCalculateIndexedSparseL2HingeOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit, Float* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, Float* pDataWeight, bool bSparseIgnoreZero, Float slope, Float alpha, Float lambda);

/**
 * Calculates the output delta for sparse L1 regularization with typed sparse data.
 *
 * @tparam T The type of sparse data.
 * @param activation The activation function.
 * @param position The position within the layer.
 * @param batch The batch size.
 * @param stride The stride value.
 * @param pUnit The pointer to the unit values.
 * @param pDelta The pointer to the delta values.
 * @param pSparseStart The pointer to the start indices of sparse data.
 * @param pSparseEnd The pointer to the end indices of sparse data.
 * @param pSparseIndex The pointer to the indices of sparse data.
 * @param pDataWeight The pointer to the weights of data.
 * @param pSparseData The pointer to the typed sparse data.
 * @param bSparseIgnoreZero Specifies whether to ignore zero values in sparse data.
 * @param scope The scope value.
 * @param alpha The alpha value.
 * @param lambda The lambda value.
 */
template<typename T>
void kCalculateSparseL1OutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit, Float* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, Float* pDataWeight, T* pSparseData, bool bSparseIgnoreZero, Float scope, Float alpha, Float lambda);

/**
 * Calculates the output delta for indexed sparse L1 regularization with typed sparse data.
 *
 * @tparam T The type of sparse data.
 * @param activation The activation function.
 * @param position The position within the layer.
 * @param batch The batch size.
 * @param stride The stride value.
 * @param pUnit The pointer to the unit values.
 * @param pDelta The pointer to the delta values.
 * @param pIndex The pointer to the indices of sparse data.
 * @param pSparseStart The pointer to the start indices of sparse data.
 * @param pSparseEnd The pointer to the end indices of sparse data.
 * @param pSparseIndex The pointer to the indices of sparse data.
 * @param pDataWeight The pointer to the weights of data.
 * @param pSparseData The pointer to the typed sparse data.
 * @param bSparseIgnoreZero Specifies whether to ignore zero values in sparse data.
 * @param scope The scope value.
 * @param alpha The alpha value.
 * @param lambda The lambda value.
 */
template<typename T>
void kCalculateIndexedSparseL1OutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit, Float* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, Float* pDataWeight, T* pSparseData, bool bSparseIgnoreZero, Float scope, Float alpha, Float lambda);

/**
 * Calculates the output delta for sparse cross-entropy loss with typed sparse data.
 *
 * @tparam T The type of sparse data.
 * @param activation The activation function.
 * @param position The position within the layer.
 * @param batch The batch size.
 * @param stride The stride value.
 * @param pUnit The pointer to the unit values.
 * @param pDelta The pointer to the delta values.
 * @param pSparseStart The pointer to the start indices of sparse data.
 * @param pSparseEnd The pointer to the end indices of sparse data.
 * @param pSparseIndex The pointer to the indices of sparse data.
 * @param pDataWeight The pointer to the weights of data.
 * @param pSparseData The pointer to the typed sparse data.
 * @param bSparseIgnoreZero Specifies whether to ignore zero values in sparse data.
 */
template<typename T>
void kCalculateSparseCrossEntropyOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit, Float* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, Float* pDataWeight, T* pSparseData, bool bSparseIgnoreZero);

/**
 * Calculates the output delta for indexed sparse cross-entropy loss with typed sparse data.
 *
 * @tparam T The type of sparse data.
 * @param activation The activation function.
 * @param position The position within the layer.
 * @param batch The batch size.
 * @param stride The stride value.
 * @param pUnit The pointer to the unit values.
 * @param pDelta The pointer to the delta values.
 * @param pIndex The pointer to the indices of sparse data.
 * @param pSparseStart The pointer to the start indices of sparse data.
 * @param pSparseEnd The pointer to the end indices of sparse data.
 * @param pSparseIndex The pointer to the indices of sparse data.
 * @param pDataWeight The pointer to the weights of data.
 * @param pSparseData The pointer to the typed sparse data.
 * @param bSparseIgnoreZero Specifies whether to ignore zero values in sparse data.
 */
template<typename T>
void kCalculateIndexedSparseCrossEntropyOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit, Float* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, Float* pDataWeight, T* pSparseData, bool bSparseIgnoreZero);

/**
 * Calculates the output delta for sparse scaled marginal cross-entropy loss with typed sparse data.
 *
 * @tparam T The type of sparse data.
 * @param activation The activation function.
 * @param position The position within the layer.
 * @param batch The batch size.
 * @param stride The stride value.
 * @param pUnit The pointer to the unit values.
 * @param pDelta The pointer to the delta values.
 * @param pSparseStart The pointer to the start indices of sparse data.
 * @param pSparseEnd The pointer to the end indices of sparse data.
 * @param pSparseIndex The pointer to the indices of sparse data.
 * @param pDataWeight The pointer to the weights of data.
 * @param pSparseData The pointer to the typed sparse data.
 * @param bSparseIgnoreZero Specifies whether to ignore zero values in sparse data.
 */
template<typename T>
void kCalculateSparseScaledMarginalCrossEntropyOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit, Float* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, Float* pDataWeight, T* pSparseData, bool bSparseIgnoreZero);

/**
 * Calculates the output delta for indexed sparse scaled marginal cross-entropy loss with typed sparse data.
 *
 * @tparam T The type of sparse data.
 * @param activation The activation function.
 * @param position The position within the layer.
 * @param batch The batch size.
 * @param stride The stride value.
 * @param pUnit The pointer to the unit values.
 * @param pDelta The pointer to the delta values.
 * @param pIndex The pointer to the indices of sparse data.
 * @param pSparseStart The pointer to the start indices of sparse data.
 * @param pSparseEnd The pointer to the end indices of sparse data.
 * @param pSparseIndex The pointer to the indices of sparse data.
 * @param pDataWeight The pointer to the weights of data.
 * @param pSparseData The pointer to the typed sparse data.
 * @param bSparseIgnoreZero Specifies whether to ignore zero values in sparse data.
 */
template<typename T>
void kCalculateIndexedSparseScaledMarginalCrossEntropyOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit, Float* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, Float* pDataWeight, T* pSparseData, bool bSparseIgnoreZero);

/**
 * Calculates the output delta for sparse regularization with typed sparse data.
 *
 * @tparam T The type of sparse data.
 * @param activation The activation function.
 * @param position The position within the layer.
 * @param batch The batch size.
 * @param stride The stride value.
 * @param pUnit The pointer to the unit values.
 * @param pDelta The pointer to the delta values.
 * @param pSparseStart The pointer to the start indices of sparse data.
 * @param pSparseEnd The pointer to the end indices of sparse data.
 * @param pSparseIndex The pointer to the indices of sparse data.
 * @param pDataWeight The pointer to the weights of data.
 * @param pSparseData The pointer to the typed sparse data.
 * @param bSparseIgnoreZero Specifies whether to ignore zero values in sparse data.
 * @param slope The slope value.
 * @param alpha The alpha value.
 * @param lambda The lambda value.
 */
template<typename T>
void kCalculateSparseOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit, Float* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, Float* pDataWeight, T* pSparseData, bool bSparseIgnoreZero, Float slope, Float alpha, Float lambda);

/**
 * Calculates the output delta for indexed sparse regularization with typed sparse data.
 *
 * @tparam T The type of sparse data.
 * @param activation The activation function.
 * @param position The position within the layer.
 * @param batch The batch size.
 * @param stride The stride value.
 * @param pUnit The pointer to the unit values.
 * @param pDelta The pointer to the delta values.
 * @param pIndex The pointer to the indices of sparse data.
 * @param pSparseStart The pointer to the start indices of sparse data.
 * @param pSparseEnd The pointer to the end indices of sparse data.
 * @param pSparseIndex The pointer to the indices of sparse data.
 * @param pDataWeight The pointer to the weights of data.
 * @param pSparseData The pointer to the typed sparse data.
 * @param bSparseIgnoreZero Specifies whether to ignore zero values in sparse data.
 * @param slope The slope value.
 * @param alpha The alpha value.
 * @param lambda The lambda value.
 */
template<typename T>
void kCalculateIndexedSparseOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit, Float* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, Float* pDataWeight, T* pSparseData, bool bSparseIgnoreZero, Float slope, Float alpha, Float lambda);

/**
 * Calculates the output delta for sparse data scaled marginal cross-entropy loss with typed sparse data.
 *
 * @tparam T The type of sparse data.
 * @param activation The activation function.
 * @param position The position within the layer.
 * @param batch The batch size.
 * @param stride The stride value.
 * @param pUnit The pointer to the unit values.
 * @param pDelta The pointer to the delta values.
 * @param pSparseStart The pointer to the start indices of sparse data.
 * @param pSparseEnd The pointer to the end indices of sparse data.
 * @param pSparseIndex The pointer to the indices of sparse data.
 * @param pSparseData The pointer to the typed sparse data.
 * @param bSparseIgnoreZero Specifies whether to ignore zero values in sparse data.
 */
template<typename T>
void kCalculateSparseDataScaledMarginalCrossEntropyOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit, Float* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, T* pSparseData, bool bSparseIgnoreZero);

/**
 * Calculates the output delta for indexed sparse data scaled marginal cross-entropy loss with typed sparse data.
 *
 * @tparam T The type of sparse data.
 * @param activation The activation function.
 * @param position The position within the layer.
 * @param batch The batch size.
 * @param stride The stride value.
 * @param pUnit The pointer to the unit values.
 * @param pDelta The pointer to the delta values.
 * @param pIndex The pointer to the indices of sparse data.
 * @param pSparseStart The pointer to the start indices of sparse data.
 * @param pSparseEnd The pointer to the end indices of sparse data.
 * @param pSparseIndex The pointer to the indices of sparse data.
 * @param pSparseData The pointer to the typed sparse data.
 * @param bSparseIgnoreZero Specifies whether to ignore zero values in sparse data.
 */
template<typename T>
void kCalculateIndexedSparseDataScaledMarginalCrossEntropyOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit, Float* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, T* pSparseData, bool bSparseIgnoreZero);

/**
 * Calculates the output delta for sparse analog regularization with typed sparse data.
 *
 * @tparam T The type of sparse data.
 * @param activation The activation function.
 * @param position The position within the layer.
 * @param batch The batch size.
 * @param stride The stride value.
 * @param pUnit The pointer to the unit values.
 * @param pDelta The pointer to the delta values.
 * @param pSparseStart The pointer to the start indices of sparse data.
 * @param pSparseEnd The pointer to the end indices of sparse data.
 * @param pSparseIndex The pointer to the indices of sparse data.
 * @param pDataWeight The pointer to the weights of data.
 * @param pSparseData The pointer to the typed sparse data.
 * @param bSparseIgnoreZero Specifies whether to ignore zero values in sparse data.
 * @param slope The slope value.
 * @param alpha The alpha value.
 * @param lambda The lambda value.
 */
template<typename T>
void kCalculateSparseAnalogOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit, Float* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, Float* pDataWeight, T* pSparseData, bool bSparseIgnoreZero, Float slope, Float alpha, Float lambda);

/**
 * Calculates the output delta for indexed sparse analog regularization with typed sparse data.
 *
 * @tparam T The type of sparse data.
 * @param activation The activation function.
 * @param position The position within the layer.
 * @param batch The batch size.
 * @param stride The stride value.
 * @param pUnit The pointer to the unit values.
 * @param pDelta The pointer to the delta values.
 * @param pIndex The pointer to the indices of sparse data.
 * @param pSparseStart The pointer to the start indices of sparse data.
 * @param pSparseEnd The pointer to the end indices of sparse data.
 * @param pSparseIndex The pointer to the indices of sparse data.
 * @param pDataWeight The pointer to the weights of data.
 * @param pSparseData The pointer to the typed sparse data.
 * @param bSparseIgnoreZero Specifies whether to ignore zero values in sparse data.
 * @param slope The slope value.
 * @param alpha The alpha value.
 * @param lambda The lambda value.
 */
template<typename T>
void kCalculateIndexedSparseAnalogOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit, Float* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, Float* pDataWeight, T* pSparseData, bool bSparseIgnoreZero, Float slope, Float alpha, Float lambda);

/**
 * Calculates the output delta for sparse analog L2 hinge regularization with typed sparse data.
 *
 * @tparam T The type of sparse data.
 * @param activation The activation function.
 * @param position The position within the layer.
 * @param batch The batch size.
 * @param stride The stride value.
 * @param pUnit The pointer to the unit values.
 * @param pDelta The pointer to the delta values.
 * @param pSparseStart The pointer to the start indices of sparse data.
 * @param pSparseEnd The pointer to the end indices of sparse data.
 * @param pSparseIndex The pointer to the indices of sparse data.
 * @param pDataWeight The pointer to the weights of data.
 * @param pSparseData The pointer to the typed sparse data.
 * @param bSparseIgnoreZero Specifies whether to ignore zero values in sparse data.
 * @param slope The slope value.
 * @param alpha The alpha value.
 * @param lambda The lambda value.
 */
template<typename T>
void kCalculateSparseAnalogL2HingeOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit, Float* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, Float* pDataWeight, T* pSparseData, bool bSparseIgnoreZero, Float slope, Float alpha, Float lambda);

/**
 * Calculates the output delta for indexed sparse analog L2 hinge regularization with typed sparse data.
 *
 * @tparam T The type of sparse data.
 * @param activation The activation function.
 * @param position The position within the layer.
 * @param batch The batch size.
 * @param stride The stride value.
 * @param pUnit The pointer to the unit values.
 * @param pDelta The pointer to the delta values.
 * @param pIndex The pointer to the indices of sparse data.
 * @param pSparseStart The pointer to the start indices of sparse data.
 * @param pSparseEnd The pointer to the end indices of sparse data.
 * @param pSparseIndex The pointer to the indices of sparse data.
 * @param pDataWeight The pointer to the weights of data.
 * @param pSparseData The pointer to the typed sparse data.
 * @param bSparseIgnoreZero Specifies whether to ignore zero values in sparse data.
 * @param slope The slope value.
 * @param alpha The alpha value.
 * @param lambda The lambda value.
 */
template<typename T>
void kCalculateIndexedSparseAnalogL2HingeOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, Float* pUnit, Float* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, Float* pDataWeight, T* pSparseData, bool bSparseIgnoreZero, Float slope, Float alpha, Float lambda);



/**
 * Calculates the sparseness penalty.
 *
 * @param batch    The batch size.
 * @param stride   The stride.
 * @param pUnit    Pointer to the unit data.
 * @param pDelta   Pointer to the delta data.
 * @param p        The p value.
 * @param beta     The beta value.
 */
void kCalculateSparsenessPenalty(uint32_t batch, uint32_t stride, Float* pUnit, Float* pDelta, Float p, Float beta);

/**
 * Calculates the Hadamard product.
 *
 * @param activation The activation type.
 * @param size       The size of the data.
 * @param scale      The scale value.
 * @param pUnit      Pointer to the unit data.
 * @param pDelta     Pointer to the delta data.
 * @param slope      The slope value.
 * @param alpha      The alpha value.
 * @param lambda     The lambda value.
 */
void kCalculateHadamardProduct(Activation activation, uint64_t size, Float scale, Float* pUnit, Float* pDelta, Float slope, Float alpha, Float lambda);

/**
 * Calculates the sigmoid activation.
 *
 * @param pData Pointer to the data.
 * @param size  The size of the data.
 */
void kCalculateSigmoidActivation(Float* pData, uint64_t size);

/**
 * Calculates the hyperbolic tangent (tanh) activation.
 *
 * @param pData Pointer to the data.
 * @param size  The size of the data.
 */
void kCalculateTanhActivation(Float* pData, uint64_t size);

/**
 * Calculates the rectified linear unit (ReLU) activation.
 *
 * @param pData Pointer to the data.
 * @param size  The size of the data.
 */
void kCalculateRELUActivation(Float* pData, uint64_t size);

/**
 * Calculates the exponential linear unit (ELU) activation.
 *
 * @param pData  Pointer to the data.
 * @param size   The size of the data.
 * @param alpha  The alpha value.
 */
void kCalculateELUActivation(Float* pData, uint64_t size, Float alpha);

/**
 * Calculates the scaled exponential linear unit (SELU) activation.
 *
 * @param pData   Pointer to the data.
 * @param size    The size of the data.
 * @param alpha   The alpha value.
 * @param lambda  The lambda value.
 */
void kCalculateSELUActivation(Float* pData, uint64_t size, Float alpha, Float lambda);

/**
 * Calculates the leaky rectified linear unit (LReLU) activation.
 *
 * @param pData  Pointer to the data.
 * @param size   The size of the data.
 * @param slope  The slope value.
 */
void kCalculateLRELUActivation(Float* pData, uint64_t size, Float slope);

/**
 * Calculates the softmax activation.
 *
 * @param pData   Pointer to the data.
 * @param batch   The batch size.
 * @param stride  The stride.
 */
void kCalculateSoftMaxActivation(Float* pData, uint32_t batch, uint32_t stride);

/**
 * Updates the weights using stochastic gradient descent (SGD).
 *
 * @param alpha            The learning rate.
 * @param lambda           The weight decay.
 * @param lambda1          The L1 regularization.
 * @param size             The size of the weights.
 * @param pWeightGradient  Pointer to the weight gradients.
 * @param pWeight          Pointer to the weights.
 */
void kSGDUpdateWeights(Float alpha, Float lambda, Float lambda1, uint64_t size, Float* pWeightGradient, Float* pWeight);

/**
 * Updates the biases using stochastic gradient descent (SGD).
 *
 * @param alpha   The learning rate.
 * @param batch   The batch size.
 * @param width   The width of the biases.
 * @param pDelta  Pointer to the deltas.
 * @param pBias   Pointer to the biases.
 */
void kSGDUpdateBiases(Float alpha, uint32_t batch, uint32_t width, Float* pDelta, Float* pBias);

/**
 * Updates the weights using momentum.
 *
 * @param alpha            The learning rate.
 * @param lambda           The weight decay.
 * @param lambda1          The L1 regularization.
 * @param mu               The momentum.
 * @param size             The size of the weights.
 * @param pWeightVelocity  Pointer to the weight velocities.
 * @param pWeightGradient  Pointer to the weight gradients.
 * @param pWeight          Pointer to the weights.
 */
void kMomentumUpdateWeights(Float alpha, Float lambda, Float lambda1, Float mu, uint64_t size, Float* pWeightVelocity, Float* pWeightGradient, Float* pWeight);

/**
 * Updates the biases using momentum.
 *
 * @param alpha        The learning rate.
 * @param mu           The momentum.
 * @param batch        The batch size.
 * @param width        The width of the biases.
 * @param pDelta       Pointer to the deltas.
 * @param pBiasVelocity  Pointer to the bias velocities.
 * @param pBias        Pointer to the biases.
 */
void kMomentumUpdateBiases(Float alpha, Float mu, uint32_t batch, uint32_t width, Float* pDelta, Float* pBiasVelocity, Float* pBias);

/**
 * Updates the weights using AdaGrad.
 *
 * @param alpha            The learning rate.
 * @param lambda           The weight decay.
 * @param lambda1          The L1 regularization.
 * @param size             The size of the weights.
 * @param pWeightVelocity  Pointer to the weight velocities.
 * @param pWeightGradient  Pointer to the weight gradients.
 * @param pWeight          Pointer to the weights.
 */
void kAdaGradUpdateWeights(Float alpha, Float lambda, Float lambda1, uint64_t size, Float* pWeightVelocity, Float* pWeightGradient, Float* pWeight);

/**
 * Updates the biases using AdaGrad.
 *
 * @param alpha    The learning rate.
 * @param batch    The batch size.
 * @param width    The width of the biases.
 * @param pDelta   Pointer to the deltas.
 * @param pBiasVelocity  Pointer to the bias velocities.
 * @param pBias    Pointer to the biases.
 */
void kAdaGradUpdateBiases(Float alpha, uint32_t batch, uint32_t width, Float* pDelta, Float* pBiasVelocity, Float* pBias);

/**
 * Shifts the weights using Nesterov momentum.
 *
 * @param mu       The momentum.
 * @param size     The size of the weights.
 * @param pWeightVelocity  Pointer to the weight velocities.
 * @param pWeight  Pointer to the weights.
 */
void kNesterovShiftWeights(Float mu, uint64_t size, Float* pWeightVelocity, Float* pWeight);

/**
 * Shifts the biases using Nesterov momentum.
 *
 * @param mu       The momentum.
 * @param width    The width of the biases.
 * @param pBiasVelocity  Pointer to the bias velocities.
 * @param pBias    Pointer to the biases.
 */
void kNesterovShiftBiases(Float mu, uint32_t width, Float* pBiasVelocity, Float* pBias);

/**
 * Updates the weights using Nesterov accelerated gradient (NAG).
 *
 * @param alpha            The learning rate.
 * @param lambda           The weight decay.
 * @param lambda1          The L1 regularization.
 * @param mu               The momentum.
 * @param size             The size of the weights.
 * @param pWeightVelocity  Pointer to the weight velocities.
 * @param pWeightGradient  Pointer to the weight gradients.
 * @param pWeight          Pointer to the weights.
 */
void kNesterovUpdateWeights(Float alpha, Float lambda, Float lambda1, Float mu, uint64_t size, Float* pWeightVelocity, Float* pWeightGradient, Float* pWeight);

/**
 * Updates the biases using Nesterov accelerated gradient (NAG).
 *
 * @param alpha            The learning rate.
 * @param mu               The momentum.
 * @param batch            The batch size.
 * @param width            The width of the biases.
 * @param pDelta           Pointer to the deltas.
 * @param pBiasVelocity    Pointer to the bias velocities.
 * @param pBias            Pointer to the biases.
 */
void kNesterovUpdateBiases(Float alpha, Float mu, uint32_t batch, uint32_t width, Float* pDelta, Float* pBiasVelocity, Float* pBias);

/**
 * Updates the weights using RMSProp.
 *
 * @param alpha            The learning rate.
 * @param lambda           The weight decay.
 * @param lambda1          The L1 regularization.
 * @param mu               The momentum.
 * @param size             The size of the weights.
 * @param pWeightVelocity  Pointer to the weight velocities.
 * @param pWeightGradient  Pointer to the weight gradients.
 * @param pWeight          Pointer to the weights.
 */
void kRMSPropUpdateWeights(Float alpha, Float lambda, Float lambda1, Float mu, uint64_t size, Float* pWeightVelocity, Float* pWeightGradient, Float* pWeight);

/**
 * Updates the biases using RMSProp.
 *
 * @param alpha        The learning rate.
 * @param mu           The momentum.
 * @param batch        The batch size.
 * @param width        The width of the biases.
 * @param pDelta       Pointer to the deltas.
 * @param pBiasVelocity  Pointer to the bias velocities.
 * @param pBias        Pointer to the biases.
 */
void kRMSPropUpdateBiases(Float alpha, Float mu, uint32_t batch, uint32_t width, Float* pDelta, Float* pBiasVelocity, Float* pBias);

/**
 * Updates the weights using AdaDelta.
 *
 * @param lambda                   The weight decay.
 * @param lambda1                  The L1 regularization.
 * @param mu                       The momentum.
 * @param size                     The size of the weights.
 * @param pWeightVelocity          Pointer to the weight velocities.
 * @param pWeightGradient          Pointer to the weight gradients.
 * @param pWeightGradientVelocity  Pointer to the weight gradient velocities.
 * @param pWeight                  Pointer to the weights.
 */
void kAdaDeltaUpdateWeights(Float lambda, Float lambda1, Float mu, uint64_t size, Float* pWeightVelocity, Float* pWeightGradient, Float* pWeightGradientVelocity, Float* pWeight);

/**
 * Updates the biases using AdaDelta.
 *
 * @param mu                       The momentum.
 * @param batch                    The batch size.
 * @param width                    The width of the biases.
 * @param pDelta                   Pointer to the deltas.
 * @param pBiasVelocity            Pointer to the bias velocities.
 * @param pBiasGradientVelocity    Pointer to the bias gradient velocities.
 * @param pBias                    Pointer to the biases.
 */
void kAdaDeltaUpdateBiases(Float mu, uint32_t batch, uint32_t width, Float* pDelta, Float* pBiasVelocity, Float* pBiasGradientVelocity, Float* pBias);

/**
 * Updates the weights using Adam.
 *
 * @param alpha                    The learning rate.
 * @param lambda                   The weight decay.
 * @param lambda1                  The L1 regularization.
 * @param mu                       The momentum.
 * @param mu1                      The momentum1.
 * @param t                        The time step.
 * @param size                     The size of the weights.
 * @param pWeightVelocity          Pointer to the weight velocities.
 * @param pWeightGradient          Pointer to the weight gradients.
 * @param pWeightGradientVelocity  Pointer to the weight gradient velocities.
 * @param pWeight                  Pointer to the weights.
 */
void kAdamUpdateWeights(Float alpha, Float lambda, Float lambda1, Float mu, Float mu1, Float t, uint64_t size, Float* pWeightVelocity, Float* pWeightGradient, Float* pWeightGradientVelocity, Float* pWeight);

/**
 * Updates the biases using Adam.
 *
 * @param alpha                    The learning rate.
 * @param mu                       The momentum.
 * @param mu1                      The momentum1.
 * @param t                        The time step.
 * @param batch                    The batch size.
 * @param width                    The width of the biases.
 * @param pDelta                   Pointer to the deltas.
 * @param pBiasVelocity            Pointer to the bias velocities.
 * @param pBiasGradientVelocity    Pointer to the bias gradient velocities.
 * @param pBias                    Pointer to the biases.
 */
void kAdamUpdateBiases(Float alpha, Float mu, Float mu1, Float t, uint32_t batch, uint32_t width, Float* pDelta, Float* pBiasVelocity, Float* pBiasGradientVelocity, Float* pBias);

/**
 * Calculates the maxout activation.
 *
 * @param pSrc  Pointer to the source data.
 * @param size  The size of the data.
 * @param pDst  Pointer to the destination data.
 */
void kCalculateMaxout(Float* pSrc, size_t size, Float* pDst);

/**
 * Calculates the cosine similarity.
 *
 * @param p0Vector   Pointer to the first vector.
 * @param pVector    Pointer to the second vector.
 * @param batch      The batch size.
 * @param stride     The stride.
 * @param pDPOut     Pointer to the dot product output.
 * @param pAOut      Pointer to the A output.
 * @param pBOut      Pointer to the B output.
 * @param outStride  The output stride.
 */
void kCalculateCosine(Float* p0Vector, Float* pVector, uint32_t batch, uint32_t stride, Float* pDPOut, Float* pAOut, Float* pBOut, uint32_t outStride);

/**
 * Calculates the dot product.
 *
 * @param p0Vector      Pointer to the first vector.
 * @param pVector       Pointer to the second vector.
 * @param batch         The batch size.
 * @param stride        The stride.
 * @param pDPOut        Pointer to the dot product output.
 * @param outStride     The output stride.
 */
void kCalculateDotProduct(Float* p0Vector, Float* pVector, uint32_t batch, uint32_t stride, Float* pDPOut, uint32_t outStride);

/**
 * Calculates the delta for the maxout activation.
 *
 * @param pSrc         Pointer to the source data.
 * @param pSrcDelta    Pointer to the source delta data.
 * @param size         The size of the data.
 * @param beta         The beta value.
 * @param pDst         Pointer to the destination data.
 * @param pDstDelta    Pointer to the destination delta data.
 */
void kCalculateMaxoutDelta(Float* pSrc, Float* pSrcDelta, size_t size, Float beta, Float* pDst, Float* pDstDelta);

/**
 * Calculates the delta for the dot product.
 *
 * @param pDPDelta       Pointer to the dot product delta.
 * @param p0Vector       Pointer to the first vector.
 * @param pVector        Pointer to the second vector.
 * @param batch          The batch size.
 * @param stride         The stride.
 * @param pDelta0        Pointer to the delta for the first vector.
 * @param beta0          The beta value for the first vector.
 * @param pDelta         Pointer to the delta for the second vector.
 * @param beta           The beta value for the second vector.
 * @param inputStride    The input stride.
 */
void kCalculateDotProductDelta(Float* pDPDelta, Float* p0Vector, Float* pVector, uint32_t batch, uint32_t stride, Float* pDelta0, Float beta0, Float* pDelta, Float beta, uint32_t inputStride);

/**
 * Calculates the delta for the cosine similarity.
 *
 * @param pDPDelta       Pointer to the dot product delta.
 * @param pDP            Pointer to the dot product.
 * @param pA             Pointer to the A data.
 * @param pB             Pointer to the B data.
 * @param p0Vector       Pointer to the first vector.
 * @param pVector        Pointer to the second vector.
 * @param batch          The batch size.
 * @param stride         The stride.
 * @param pDelta0        Pointer to the delta for the first vector.
 * @param beta0          The beta value for the first vector.
 * @param pDelta         Pointer to the delta for the second vector.
 * @param beta           The beta value for the second vector.
 * @param inputStride    The input stride.
 */
void kCalculateCosineDelta(Float* pDPDelta, Float* pDP, Float* pA, Float* pB, Float* p0Vector, Float* pVector, uint32_t batch, uint32_t stride, Float* pDelta0, Float beta0, Float* pDelta, Float beta, uint32_t inputStride);



#ifdef __NVCC__
__device__ inline uint64_t llitoulli(int64_t l)
{
    uint64_t u;
    asm("mov.b64    %0, %1;" : "=l"(u) : "l"(l));
    return u;
}

__device__ inline int64_t ullitolli(uint64_t u)
{
    int64_t l;
    asm("mov.b64    %0, %1;" : "=l"(l) : "l"(u));
    return l;
}

#if (CUDA_VERSION >= 9000)
#define SHFL(x, lane) __shfl_sync(0xffffffff, (x), (lane))
#define BALLOT(predicate) __ballot_sync(0xffffffff, (predicate))
#define ANY(predicate) __any_sync(0xffffffff, (predicate))
#else
#define SHFL(x, lane) __shfl((x), (lane))
#define BALLOT(predicate) __ballot(predicate)
#define ANY(predicate) __any(predicate)
#endif


#define REDUCEERROR(error) \
    if (ANY(error != (Float)0.0)) \
    { \
        uint32_t tgx            = threadIdx.x & cData._warpMask; \
        error                  += SHFL(error, tgx ^ 1); \
        error                  += SHFL(error, tgx ^ 2); \
        error                  += SHFL(error, tgx ^ 4); \
        error                  += SHFL(error, tgx ^ 8); \
        error                  += SHFL(error, tgx ^ 16); \
        if (tgx == 0) \
        { \
            atomicAdd(cData._pAccumulator, llitoulli(llrintf(ERRORSCALEF * error))); \
        } \
    } 


#define REDUCE(a) \
    if (ANY((a) != (Float)0.0)) \
    { \
        uint32_t tgx            = threadIdx.x & cData._warpMask; \
        a                      += SHFL((a), tgx ^ 1); \
        a                      += SHFL((a), tgx ^ 2); \
        a                      += SHFL((a), tgx ^ 4); \
        a                      += SHFL((a), tgx ^ 8); \
        a                      += SHFL((a), tgx ^ 16); \
    } 


#endif

#endif
