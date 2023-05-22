#include "../engine/bitonic.h"
#include "Output.h"
#include <limits>

/**
 * @brief CUDA kernel for calculating the output.
 *
 * @param pOutputBuffer Pointer to the output buffer.
 * @param pKeyBuffer Pointer to the key buffer.
 * @param pValueBuffer Pointer to the value buffer.
 * @param batch The number of batches.
 * @param width The width of the data.
 * @param widthPadding The width padding.
 * @param k The value of k.
 */
__global__ void kCalculateOutput_kernel(float* pOutputBuffer, float* pKeyBuffer, unsigned int* pValueBuffer,
                                        unsigned int batch, unsigned int width, unsigned int widthPadding,
                                        unsigned int k)
{
    __shared__ volatile float sKey[160 * 4];
    __shared__ volatile unsigned int sValue[160 * 4];

    unsigned int dataWidth = width - widthPadding;
    unsigned int pos = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    unsigned int tgx = threadIdx.x & 31;

    if (pos < batch)
    {
        float* pOutput = pOutputBuffer + pos * width;
        unsigned int offset = threadIdx.x >> 5;
        volatile float* psKey = &sKey[160 * offset];
        volatile unsigned int* psValue = &sValue[160 * offset];

        float k0 = -std::numeric_limits<float>::max();
        float k1 = -std::numeric_limits<float>::max();
        float k2 = -std::numeric_limits<float>::max();
        float k3 = -std::numeric_limits<float>::max();
        unsigned int v0 = 0;
        unsigned int v1 = 0;
        unsigned int v2 = 0;
        unsigned int v3 = 0;

        unsigned int wpos = tgx;
        if (wpos < dataWidth)
        {
            k0 = pOutput[wpos];
            v0 = wpos;
        }
        wpos += 32;
        if (wpos < dataWidth)
        {
            k1 = pOutput[wpos];
            v1 = wpos;
        }
        wpos += 32;
        if (wpos < dataWidth)
        {
            k2 = pOutput[wpos];
            v2 = wpos;
        }
        wpos += 32;
        if (wpos < dataWidth)
        {
            k3 = pOutput[wpos];
            v3 = wpos;
        }

        float minValue = -std::numeric_limits<float>::max();
        unsigned int rpos = 128;
        unsigned int bufferSize = 0;

        while (rpos < dataWidth)
        {
            unsigned int wpos = rpos + tgx;
            float key = -std::numeric_limits<float>::max();
            unsigned int value = wpos;
            if (wpos < dataWidth)
            {
                key = pOutput[wpos];
            }

            unsigned int count = __ballot_sync(0xffffffff, key > minValue);

            if (key > minValue)
            {
                unsigned int mask = 0xffffffff >> (32 - tgx);
                unsigned int offset = __popc_sync(0xffffffff, count & mask) + bufferSize;
                psKey[offset] = key;
                psValue[offset] = value;
            }

            bufferSize += __popc_sync(0xffffffff, count);

            if (bufferSize >= 128)
            {
                k2 = psKey[tgx + 2 * 32];
                v2 = psValue[tgx + 2 * 32];
                k3 = psKey[tgx + 3 * 32];
                v3 = psValue[tgx + 3 * 32];

                BITONICSORT256_256();

                minValue = __shfl_sync(0xffffffff, k3, 31);

                bufferSize -= 128;
                if (tgx < bufferSize)
                {
                    psKey[tgx] = psKey[tgx + 128];
                    psValue[tgx] = psValue[tgx + 128];
                }
            }

            rpos += 32;
        }

        if ((bufferSize > 0) || (dataWidth <= 128))
        {
            k2 = -std::numeric_limits<float>::max();
            k3 = -std::numeric_limits<float>::max();
            v2 = 0;
            v3 = 0;

            if (tgx < bufferSize)
            {
                k2 = psKey[tgx];
                v2 = psValue[tgx];
            }
            if (tgx + 32 < bufferSize)
            {
                k3 = psKey[tgx + 32];
                v3 = psValue[tgx + 32];
            }

            BITONICSORT256_256();
        }

        float* pKey = pKeyBuffer + pos * k;
        unsigned int* pValue = pValueBuffer + pos * k;
        wpos = tgx;
        if (wpos < k)
        {
            pKey[wpos] = k0;
            pValue[wpos] = v0;
        }
        wpos += 32;
        if (wpos < k)
        {
            pKey[wpos] = k1;
            pValue[wpos] = v1;
        }
        wpos += 32;
        if (wpos < k)
        {
            pKey[wpos] = k2;
            pValue[wpos] = v2;
        }
        wpos += 32;
        if (wpos < k)
        {
            pKey[wpos] = k3;
            pValue[wpos] = v3;
        }
    }
}

/**
 * @brief Calculates the output using CUDA.
 *
 * @param pOutput Pointer to the output.
 * @param pKey Pointer to the key.
 * @param pValue Pointer to the value.
 * @param batch The number of batches.
 * @param width The width of the data.
 * @param widthPadding The width padding.
 * @param k The value of k.
 */
void kCalculateOutput(float* pOutput, float* pKey, unsigned int* pValue, unsigned int batch, unsigned int width,
                      unsigned int widthPadding, unsigned int k)
{
    unsigned int blocks = (batch + 3) / 4;
    kCalculateOutput_kernel<<<blocks, 128>>>(pOutput, pKey, pValue, batch, width, widthPadding, k);
    LAUNCHERROR("kCalculateOutput_kernel");
}

