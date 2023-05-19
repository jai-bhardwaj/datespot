#ifndef BITONIC_H
#define BITONIC_H

#define BITONICWARPEXCHANGE_64(mask) \
    for (int i = 0; i < 2; i++) { \
        int idx1 = i; \
        int idx2 = i + 1; \
        \
        key1 = k[idx1]; \
        value1 = v[idx1]; \
        otgx = tgx ^ mask; \
        key2 = SHFL(k[idx2], otgx); \
        value2 = SHFL(v[idx2], otgx); \
        flag = ((key1 > key2) ^ (tgx > otgx)) && (key1 != key2); \
        k[idx1] = flag ? key1 : key2; \
        v[idx1] = flag ? value1 : value2; \
    }

#define BITONICSORT32_64() \
    BITONICWARPEXCHANGE_64(1) \
    BITONICWARPEXCHANGE_64(3) \
    BITONICWARPEXCHANGE_64(1) \
    BITONICWARPEXCHANGE_64(7) \
    BITONICWARPEXCHANGE_64(2) \
    BITONICWARPEXCHANGE_64(1) \
    BITONICWARPEXCHANGE_64(15) \
    BITONICWARPEXCHANGE_64(4) \
    BITONICWARPEXCHANGE_64(2) \
    BITONICWARPEXCHANGE_64(1) \
    BITONICWARPEXCHANGE_64(31) \
    BITONICWARPEXCHANGE_64(8) \
    BITONICWARPEXCHANGE_64(4) \
    BITONICWARPEXCHANGE_64(2) \
    BITONICWARPEXCHANGE_64(1) 


#define BITONICMERGE64_64() \
    for (int i = 0; i < 2; i++) { \
        int idx1 = i; \
        int idx2 = i + 2; \
        \
        key1 = k[idx1]; \
        value1 = v[idx1]; \
        otgx = 31 - tgx; \
        key2 = SHFL(k[idx2], otgx); \
        value2 = SHFL(v[idx2], otgx); \
        flag = (key1 > key2); \
        k[idx1] = flag ? key1 : key2; \
        v[idx1] = flag ? value1 : value2; \
        \
        key1 = k[idx1 + 2]; \
        value1 = v[idx1 + 2]; \
        key2 = SHFL(k[idx2 + 2], otgx); \
        value2 = SHFL(v[idx2 + 2], otgx); \
        flag = (key1 > key2); \
        k[idx1 + 2] = flag ? key1 : key2; \
        v[idx1 + 2] = flag ? value1 : value2; \
    }

#define BITONICSORT64_64() \
    BITONICSORT32_64() \
    BITONICMERGE64_64() \
    BITONICWARPEXCHANGE_64(16) \
    BITONICWARPEXCHANGE_64(8) \
    BITONICWARPEXCHANGE_64(4) \
    BITONICWARPEXCHANGE_64(2) \
    BITONICWARPEXCHANGE_64(1)
    
#define BITONICWARPEXCHANGE_128(mask) \
    for (int i = 0; i < 4; i++) { \
        int idx1 = i; \
        int idx2 = i + 4; \
        \
        key1 = k[idx1]; \
        value1 = v[idx1]; \
        otgx = tgx ^ mask; \
        key2 = SHFL(k[idx2], otgx); \
        value2 = SHFL(v[idx2], otgx); \
        flag = ((key1 > key2) ^ (tgx > otgx)) && (key1 != key2); \
        k[idx1] = flag ? key1 : key2; \
        v[idx1] = flag ? value1 : value2; \
        \
        key1 = k[idx1 + 4]; \
        value1 = v[idx1 + 4]; \
        key2 = SHFL(k[idx2 + 4], otgx); \
        value2 = SHFL(v[idx2 + 4], otgx); \
        flag = ((key1 > key2) ^ (tgx > otgx)) && (key1 != key2); \
        k[idx1 + 4] = flag ? key1 : key2; \
        v[idx1 + 4] = flag ? value1 : value2; \
    }

#define BITONICSORT32_128() \
    BITONICWARPEXCHANGE_128(1) \
    BITONICWARPEXCHANGE_128(3) \
    BITONICWARPEXCHANGE_128(1) \
    BITONICWARPEXCHANGE_128(7) \
    BITONICWARPEXCHANGE_128(2) \
    BITONICWARPEXCHANGE_128(1) \
    BITONICWARPEXCHANGE_128(15) \
    BITONICWARPEXCHANGE_128(4) \
    BITONICWARPEXCHANGE_128(2) \
    BITONICWARPEXCHANGE_128(1) \
    BITONICWARPEXCHANGE_128(31) \
    BITONICWARPEXCHANGE_128(8) \
    BITONICWARPEXCHANGE_128(4) \
    BITONICWARPEXCHANGE_128(2) \
    BITONICWARPEXCHANGE_128(1) 



#define BITONICMERGE64_128() \
    for (int i = 0; i < 2; i++) { \
        int idx1 = i; \
        int idx2 = i + 2; \
        \
        key1 = k[idx1]; \
        value1 = v[idx1]; \
        key2 = SHFL(k[idx2], otgx); \
        value2 = SHFL(v[idx2], otgx); \
        flag = (key1 > key2); \
        k[idx1] = flag ? key1 : key2; \
        v[idx1] = flag ? value1 : value2; \
        key1 = flag ? key2 : key1; \
        value1 = flag ? value2 : value1; \
        k[idx2] = SHFL(key1, otgx); \
        v[idx2] = SHFL(value1, otgx); \
    }

#define BITONICSORT64_128() \
    BITONICSORT32_128() \
    BITONICMERGE64_128() \
    BITONICWARPEXCHANGE_128(16) \
    BITONICWARPEXCHANGE_128(8) \
    BITONICWARPEXCHANGE_128(4) \
    BITONICWARPEXCHANGE_128(2) \
    BITONICWARPEXCHANGE_128(1)

#define BITONICMERGE128_128() \
    for (int i = 0; i < 2; i++) { \
        int idx1 = i; \
        int idx2 = i + 2; \
        \
        key1 = k[idx1]; \
        value1 = v[idx1]; \
        key2 = SHFL(k[idx2], otgx); \
        value2 = SHFL(v[idx2], otgx); \
        flag = (key1 > key2); \
        k[idx1] = flag ? key1 : key2; \
        v[idx1] = flag ? value1 : value2; \
        key1 = flag ? key2 : key1; \
        value1 = flag ? value2 : value1; \
        k[idx2] = SHFL(key1, otgx); \
        v[idx2] = SHFL(value1, otgx); \
    }


#define BITONICEXCHANGE32_128() \
    for (int i = 0; i < 2; i++) { \
        int idx1 = i; \
        int idx2 = i + 2; \
        \
        if (k[idx1] < k[idx2]) { \
            key1 = k[idx1]; \
            value1 = v[idx1]; \
            k[idx1] = k[idx2]; \
            v[idx1] = v[idx2]; \
            k[idx2] = key1; \
            v[idx2] = value1; \
        } \
    }


#define BITONICEXCHANGE64_128() \
    for (int i = 0; i < 2; i++) { \
        int idx1 = i; \
        int idx2 = i + 2; \
        \
        if (k[idx1] < k[idx2]) { \
            key1 = k[idx1]; \
            value1 = v[idx1]; \
            k[idx1] = k[idx2]; \
            v[idx1] = v[idx2]; \
            k[idx2] = key1; \
            v[idx2] = value1; \
        } \
    }

#define BITONICSORT128_128() \
    BITONICSORT64_128() \
    BITONICMERGE128_128() \
    BITONICEXCHANGE32_128() \
    BITONICWARPEXCHANGE_128(16) \
    BITONICWARPEXCHANGE_128(8) \
    BITONICWARPEXCHANGE_128(4) \
    BITONICWARPEXCHANGE_128(2) \
    BITONICWARPEXCHANGE_128(1)





#define BITONICWARPEXCHANGE_256(mask) \
    for (int i = 0; i < 8; i++) { \
        int idx1 = i; \
        int idx2 = i ^ mask; \
        \
        key1 = k[idx1]; \
        value1 = v[idx1]; \
        \
        key2 = SHFL(k[idx2], otgx); \
        value2 = SHFL(v[idx2], otgx); \
        \
        flag = ((key1 > key2) ^ (tgx > otgx)) && (key1 != key2); \
        \
        k[idx1] = flag ? key1 : key2; \
        v[idx1] = flag ? value1 : value2; \
    }

#define BITONICSORT32_256() \
    BITONICWARPEXCHANGE_256(1) \
    BITONICWARPEXCHANGE_256(3) \
    BITONICWARPEXCHANGE_256(1) \
    BITONICWARPEXCHANGE_256(7) \
    BITONICWARPEXCHANGE_256(2) \
    BITONICWARPEXCHANGE_256(1) \
    BITONICWARPEXCHANGE_256(15) \
    BITONICWARPEXCHANGE_256(4) \
    BITONICWARPEXCHANGE_256(2) \
    BITONICWARPEXCHANGE_256(1) \
    BITONICWARPEXCHANGE_256(31) \
    BITONICWARPEXCHANGE_256(8) \
    BITONICWARPEXCHANGE_256(4) \
    BITONICWARPEXCHANGE_256(2) \
    BITONICWARPEXCHANGE_256(1) 

#define BITONICMERGE64_256() \
    for (int i = 0; i < 4; i++) { \
        int idx1 = i; \
        int idx2 = i + 4; \
        \
        otgx = 31 - tgx; \
        key1 = k[idx1]; \
        value1 = v[idx1]; \
        key2 = SHFL(k[idx2], otgx); \
        value2 = SHFL(v[idx2], otgx); \
        \
        flag = (key1 > key2); \
        \
        k[idx1] = flag ? key1 : key2; \
        v[idx1] = flag ? value1 : value2; \
        \
        key1 = flag ? key2 : key1; \
        value1 = flag ? value2 : value1; \
        \
        k[idx2] = SHFL(key1, otgx); \
        v[idx2] = SHFL(value1, otgx); \
    }
    
#define BITONICSORT64_256() \
    BITONICSORT32_256() \
    BITONICMERGE64_256() \
    BITONICWARPEXCHANGE_256(16) \
    BITONICWARPEXCHANGE_256(8) \
    BITONICWARPEXCHANGE_256(4) \
    BITONICWARPEXCHANGE_256(2) \
    BITONICWARPEXCHANGE_256(1)

#define BITONICMERGE128_256() \
    for (int i = 0; i < 4; i++) { \
        int idx1 = i; \
        int idx2 = i + 4; \
        \
        otgx = 31 - tgx; \
        key1 = k[idx1]; \
        value1 = v[idx1]; \
        key2 = SHFL(k[idx2], otgx); \
        value2 = SHFL(v[idx2], otgx); \
        \
        flag = (key1 > key2); \
        \
        k[idx1] = flag ? key1 : key2; \
        v[idx1] = flag ? value1 : value2; \
        \
        key1 = flag ? key2 : key1; \
        value1 = flag ? value2 : value1; \
        \
        k[idx2] = SHFL(key1, otgx); \
        v[idx2] = SHFL(value1, otgx); \
    }

#define BITONICMERGE256_256() \
    for (int i = 0; i < 4; i++) { \
        int idx1 = i; \
        int idx2 = i + 4; \
        \
        otgx = 31 - tgx; \
        key1 = k[idx1]; \
        value1 = v[idx1]; \
        key2 = SHFL(k[idx2], otgx); \
        value2 = SHFL(v[idx2], otgx); \
        \
        flag = (key1 > key2); \
        \
        k[idx1] = flag ? key1 : key2; \
        v[idx1] = flag ? value1 : value2; \
        \
        key1 = flag ? key2 : key1; \
        value1 = flag ? value2 : value1; \
        \
        k[idx2] = SHFL(key1, otgx); \
        v[idx2] = SHFL(value1, otgx); \
    }

#define BITONICEXCHANGE32_256() \
    for (int i = 0; i < 4; i++) { \
        int idx1 = i * 2; \
        int idx2 = idx1 + 1; \
        \
        if (k[idx1] < k[idx2]) { \
            int key1 = k[idx1]; \
            int value1 = v[idx1]; \
            \
            k[idx1] = k[idx2]; \
            v[idx1] = v[idx2]; \
            \
            k[idx2] = key1; \
            v[idx2] = value1; \
        } \
    }

#define BITONICEXCHANGE64_256() \
    for (int i = 0; i < 4; i++) { \
        int idx1 = i * 2; \
        int idx2 = idx1 + 1; \
        \
        if (k[idx1] < k[idx2]) { \
            int key1 = k[idx1]; \
            int value1 = v[idx1]; \
            \
            k[idx1] = k[idx2]; \
            v[idx1] = v[idx2]; \
            \
            k[idx2] = key1; \
            v[idx2] = value1; \
        } \
    }


#define BITONICSORT128_256() \
    BITONICSORT64_256() \
    BITONICMERGE128_256() \
    BITONICEXCHANGE32_256() \
    BITONICWARPEXCHANGE_256(16) \
    BITONICWARPEXCHANGE_256(8) \
    BITONICWARPEXCHANGE_256(4) \
    BITONICWARPEXCHANGE_256(2) \
    BITONICWARPEXCHANGE_256(1)

#define BITONICSORT256_256() \
    BITONICSORT128_256() \
    BITONICMERGE256_256() \
    BITONICEXCHANGE64_256() \
    BITONICEXCHANGE32_256() \
    BITONICWARPEXCHANGE_256(16) \
    BITONICWARPEXCHANGE_256(8) \
    BITONICWARPEXCHANGE_256(4) \
    BITONICWARPEXCHANGE_256(2) \
    BITONICWARPEXCHANGE_256(1)

#define BITONICWARPEXCHANGE_512(mask) \
    for (int i = 0; i < 16; i++) { \
        int idx = i; \
        int otgx = tgx ^ mask; \
        \
        int key1 = k[idx]; \
        int value1 = v[idx]; \
        int key2 = SHFL(k[idx], otgx); \
        int value2 = SHFL(v[idx], otgx); \
        \
        bool flag = ((key1 > key2) ^ (tgx > otgx)) && (key1 != key2); \
        \
        k[idx] = flag ? key1 : key2; \
        v[idx] = flag ? value1 : value2; \
    }  

#define BITONICSORT32_512() \
    BITONICWARPEXCHANGE_512(1) \
    BITONICWARPEXCHANGE_512(3) \
    BITONICWARPEXCHANGE_512(1) \
    BITONICWARPEXCHANGE_512(7) \
    BITONICWARPEXCHANGE_512(2) \
    BITONICWARPEXCHANGE_512(1) \
    BITONICWARPEXCHANGE_512(15) \
    BITONICWARPEXCHANGE_512(4) \
    BITONICWARPEXCHANGE_512(2) \
    BITONICWARPEXCHANGE_512(1) \
    BITONICWARPEXCHANGE_512(31) \
    BITONICWARPEXCHANGE_512(8) \
    BITONICWARPEXCHANGE_512(4) \
    BITONICWARPEXCHANGE_512(2) \
    BITONICWARPEXCHANGE_512(1) 



#define BITONICMERGE64_512() \
    for (int i = 0; i < 2; i++) { \
        int idx = i * 32 + tgx; \
        int otgx = 31 - tgx; \
        \
        int key1 = k[i]; \
        int value1 = v[i]; \
        int key2 = SHFL(k[i + 1], otgx); \
        int value2 = SHFL(v[i + 1], otgx); \
        \
        bool flag = (key1 > key2); \
        \
        k[i] = flag ? key1 : key2; \
        v[i] = flag ? value1 : value2; \
        \
        key1 = flag ? key2 : key1; \
        value1 = flag ? value2 : value1; \
        \
        k[i + 1] = SHFL(key1, otgx); \
        v[i + 1] = SHFL(value1, otgx); \
        \
        key1 = k[i + 2]; \
        value1 = v[i + 2]; \
        key2 = SHFL(k[i + 3], otgx); \
        value2 = SHFL(v[i + 3], otgx); \
        \
        flag = (key1 > key2); \
        \
        k[i + 2] = flag ? key1 : key2; \
        v[i + 2] = flag ? value1 : value2; \
        \
        key1 = flag ? key2 : key1; \
        value1 = flag ? value2 : value1; \
        \
        k[i + 3] = SHFL(key1, otgx); \
        v[i + 3] = SHFL(value1, otgx); \
        \
        key1 = k[i + 4]; \
        value1 = v[i + 4]; \
        key2 = SHFL(k[i + 5], otgx); \
        value2 = SHFL(v[i + 5], otgx); \
        \
        flag = (key1 > key2); \
        \
        k[i + 4] = flag ? key1 : key2; \
        v[i + 4] = flag ? value1 : value2; \
        \
        key1 = flag ? key2 : key1; \
        value1 = flag ? value2 : value1; \
        \
        k[i + 5] = SHFL(key1, otgx); \
        v[i + 5] = SHFL(value1, otgx); \
        \
        key1 = k[i + 6]; \
        value1 = v[i + 6]; \
        key2 = SHFL(k[i + 7], otgx); \
        value2 = SHFL(v[i + 7], otgx); \
        \
        flag = (key1 > key2); \
        \
        k[i + 6] = flag ? key1 : key2; \
        v[i + 6] = flag ? value1 : value2; \
        \
        key1 = flag ? key2 : key1; \
        value1 = flag ? value2 : value1; \
        \
        k[i + 7] = SHFL(key1, otgx); \
        v[i + 7] = SHFL(value1, otgx); \
        \
        key1 = k[i + 8]; \
        value1 = v[i + 8]; \
        key2 = SHFL(k[i + 9], otgx); \
        value2 = SHFL(v[i + 9], otgx); \
        \
        flag = (key1 > key2); \
        \
        k[i + 8] = flag ? key1 : key2; \
        v[i + 8] = flag ? value1 : value2; \
        \
        key1 = flag ? key2 : key1; \
        value1 = flag ? value2 : value1; \
        \
        k[i + 9] = SHFL(key1, otgx); \
        v[i + 9] = SHFL(value1, otgx); \
        \
        key1 = k[i + 10]; \
        value1 = v[i + 10]; \
        key2 = SHFL(k[i + 11], otgx); \
        value2 = SHFL(v[i + 11], otgx); \
        \
        flag = (key1 > key2); \
        \
        k[i + 10] = flag ? key1 : key2; \
        v[i + 10] = flag ? value1 : value2; \
        \
        key1 = flag ? key2 : key1; \
        value1 = flag ? value2 : value1; \
        \
        k[i + 11] = SHFL(key1, otgx); \
        v[i + 11] = SHFL(value1, otgx); \
        \
        key1 = k[i + 12]; \
        value1 = v[i + 12]; \
        key2 = SHFL(k[i + 13], otgx); \
        value2 = SHFL(v[i + 13], otgx); \
        \
        flag = (key1 > key2); \
        \
        k[i + 12] = flag ? key1 : key2; \
        v[i + 12] = flag ? value1 : value2; \
        \
        key1 = flag ? key2 : key1; \
        value1 = flag ? value2 : value1; \
        \
        k[i + 13] = SHFL(key1, otgx); \
        v[i + 13] = SHFL(value1, otgx); \
        \
        key1 = k[i + 14]; \
        value1 = v[i + 14]; \
        key2 = SHFL(k[i + 15], otgx); \
        value2 = SHFL(v[i + 15], otgx); \
        \
        flag = (key1 > key2); \
        \
        k[i + 14] = flag ? key1 : key2; \
        v[i + 14] = flag ? value1 : value2; \
        \
        key1 = flag ? key2 : key1; \
        value1 = flag ? value2 : value1; \
        \
        k[i + 15] = SHFL(key1, otgx); \
        v[i + 15] = SHFL(value1, otgx); \
    }       

    
#define BITONICSORT64_512() \
    BITONICSORT32_512() \
    BITONICMERGE64_512() \
    BITONICWARPEXCHANGE_512(16) \
    BITONICWARPEXCHANGE_512(8) \
    BITONICWARPEXCHANGE_512(4) \
    BITONICWARPEXCHANGE_512(2) \
    BITONICWARPEXCHANGE_512(1)

#define BITONICMERGE128_512() \
    for (int i = 0; i < 4; i++) { \
        int idx = i * 32 + tgx; \
        int otgx = 31 - tgx; \
        \
        int key1 = k[i]; \
        int value1 = v[i]; \
        int key2 = SHFL(k[i + 3], otgx); \
        int value2 = SHFL(v[i + 3], otgx); \
        \
        bool flag = (key1 > key2); \
        \
        k[i] = flag ? key1 : key2; \
        v[i] = flag ? value1 : value2; \
        \
        key1 = flag ? key2 : key1; \
        value1 = flag ? value2 : value1; \
        \
        k[i + 3] = SHFL(key1, otgx); \
        v[i + 3] = SHFL(value1, otgx); \
        \
        key1 = k[i + 1]; \
        value1 = v[i + 1]; \
        key2 = SHFL(k[i + 2], otgx); \
        value2 = SHFL(v[i + 2], otgx); \
        \
        flag = (key1 > key2); \
        \
        k[i + 1] = flag ? key1 : key2; \
        v[i + 1] = flag ? value1 : value2; \
        \
        key1 = flag ? key2 : key1; \
        value1 = flag ? value2 : value1; \
        \
        k[i + 2] = SHFL(key1, otgx); \
        v[i + 2] = SHFL(value1, otgx); \
        \
        key1 = k[i + 4]; \
        value1 = v[i + 4]; \
        key2 = SHFL(k[i + 7], otgx); \
        value2 = SHFL(v[i + 7], otgx); \
        \
        flag = (key1 > key2); \
        \
        k[i + 4] = flag ? key1 : key2; \
        v[i + 4] = flag ? value1 : value2; \
        \
        key1 = flag ? key2 : key1; \
        value1 = flag ? value2 : value1; \
        \
        k[i + 7] = SHFL(key1, otgx); \
        v[i + 7] = SHFL(value1, otgx); \
        \
        key1 = k[i + 5]; \
        value1 = v[i + 5]; \
        key2 = SHFL(k[i + 6], otgx); \
        value2 = SHFL(v[i + 6], otgx); \
        \
        flag = (key1 > key2); \
        \
        k[i + 5] = flag ? key1 : key2; \
        v[i + 5] = flag ? value1 : value2; \
        \
        key1 = flag ? key2 : key1; \
        value1 = flag ? value2 : value1; \
        \
        k[i + 6] = SHFL(key1, otgx); \
        v[i + 6] = SHFL(value1, otgx); \
        \
        key1 = k[i + 8]; \
        value1 = v[i + 8]; \
        key2 = SHFL(k[i + 11], otgx); \
        value2 = SHFL(v[i + 11], otgx); \
        \
        flag = (key1 > key2); \
        \
        k[i + 8] = flag ? key1 : key2; \
        v[i + 8] = flag ? value1 : value2; \
        \
        key1 = flag ? key2 : key1; \
        value1 = flag ? value2 : value1; \
        \
        k[i + 11] = SHFL(key1, otgx); \
        v[i + 11] = SHFL(value1, otgx); \
        \
        key1 = k[i + 9]; \
        value1 = v[i + 9]; \
        key2 = SHFL(k[i + 10], otgx); \
        value2 = SHFL(v[i + 10], otgx); \
        \
        flag = (key1 > key2); \
        \
        k[i + 9] = flag ? key1 : key2; \
        v[i + 9] = flag ? value1 : value2; \
        \
        key1 = flag ? key2 : key1; \
        value1 = flag ? value2 : value1; \
        \
        k[i + 10] = SHFL(key1, otgx); \
        v[i + 10] = SHFL(value1, otgx); \
        \
        key1 = k[i + 12]; \
        value1 = v[i + 12]; \
        key2 = SHFL(k[i + 15], otgx); \
        value2 = SHFL(v[i + 15], otgx); \
        \
        flag = (key1 > key2); \
        \
        k[i + 12] = flag ? key1 : key2; \
        v[i + 12] = flag ? value1 : value2; \
        \
        key1 = flag ? key2 : key1; \
        value1 = flag ? value2 : value1; \
        \
        k[i + 15] = SHFL(key1, otgx); \
        v[i + 15] = SHFL(value1, otgx); \
        \
        key1 = k[i + 13]; \
        value1 = v[i + 13]; \
        key2 = SHFL(k[i + 14], otgx); \
        value2 = SHFL(v[i + 14], otgx); \
        \
        flag = (key1 > key2); \
        \
        k[i + 13] = flag ? key1 : key2; \
        v[i + 13] = flag ? value1 : value2; \
        \
        key1 = flag ? key2 : key1; \
        value1 = flag ? value2 : value1; \
        \
        k[i + 14] = SHFL(key1, otgx); \
        v[i + 14] = SHFL(value1, otgx); \
    }

#define BITONICMERGE256_512() \
    for (int i = 0; i < 8; i++) { \
        int idx = i * 32 + tgx; \
        int otgx = 31 - tgx; \
        \
        int key1 = k[i]; \
        int value1 = v[i]; \
        int key2 = SHFL(k[i + 7], otgx); \
        int value2 = SHFL(v[i + 7], otgx); \
        \
        bool flag = (key1 > key2); \
        \
        k[i] = flag ? key1 : key2; \
        v[i] = flag ? value1 : value2; \
        \
        key1 = flag ? key2 : key1; \
        value1 = flag ? value2 : value1; \
        \
        k[i + 7] = SHFL(key1, otgx); \
        v[i + 7] = SHFL(value1, otgx); \
        \
        key1 = k[i + 1]; \
        value1 = v[i + 1]; \
        key2 = SHFL(k[i + 6], otgx); \
        value2 = SHFL(v[i + 6], otgx); \
        \
        flag = (key1 > key2); \
        \
        k[i + 1] = flag ? key1 : key2; \
        v[i + 1] = flag ? value1 : value2; \
        \
        key1 = flag ? key2 : key1; \
        value1 = flag ? value2 : value1; \
        \
        k[i + 6] = SHFL(key1, otgx); \
        v[i + 6] = SHFL(value1, otgx); \
        \
        key1 = k[i + 2]; \
        value1 = v[i + 2]; \
        key2 = SHFL(k[i + 5], otgx); \
        value2 = SHFL(v[i + 5], otgx); \
        \
        flag = (key1 > key2); \
        \
        k[i + 2] = flag ? key1 : key2; \
        v[i + 2] = flag ? value1 : value2; \
        \
        key1 = flag ? key2 : key1; \
        value1 = flag ? value2 : value1; \
        \
        k[i + 5] = SHFL(key1, otgx); \
        v[i + 5] = SHFL(value1, otgx); \
        \
        key1 = k[i + 3]; \
        value1 = v[i + 3]; \
        key2 = SHFL(k[i + 4], otgx); \
        value2 = SHFL(v[i + 4], otgx); \
        \
        flag = (key1 > key2); \
        \
        k[i + 3] = flag ? key1 : key2; \
        v[i + 3] = flag ? value1 : value2; \
        \
        key1 = flag ? key2 : key1; \
        value1 = flag ? value2 : value1; \
        \
        k[i + 4] = SHFL(key1, otgx); \
        v[i + 4] = SHFL(value1, otgx); \
    }

#define BITONICMERGE512_512() \
    otgx = 31 - tgx; \
    \
    for (int i = 0; i < 16; i++) { \
        int idx1 = i; \
        int idx2 = 31 - i; \
        \
        key1 = k[idx1]; \
        value1 = v[idx1]; \
        key2 = SHFL(k[idx2], otgx); \
        value2 = SHFL(v[idx2], otgx); \
        flag = (key1 > key2); \
        k[idx1] = flag ? key1 : key2; \
        v[idx1] = flag ? value1 : value2; \
        k[idx2] = flag ? key2 : key1; \
        v[idx2] = flag ? value2 : value1; \
    } 

#define BITONICEXCHANGE32_512() \
    for (int i = 0; i < 16; i += 2) { \
        int idx1 = i; \
        int idx2 = i + 1; \
        \
        if (k[idx1] < k[idx2]) { \
            key1 = k[idx1]; \
            value1 = v[idx1]; \
            k[idx1] = k[idx2]; \
            v[idx1] = v[idx2]; \
            k[idx2] = key1; \
            v[idx2] = value1; \
        } \
    }    

#define BITONICEXCHANGE64_512() \
    for (int i = 0; i < 8; i++) { \
        int idx1 = i * 2; \
        int idx2 = i * 2 + 1; \
        \
        if (k[idx1] < k[idx2]) { \
            key1 = k[idx1]; \
            value1 = v[idx1]; \
            k[idx1] = k[idx2]; \
            v[idx1] = v[idx2]; \
            k[idx2] = key1; \
            v[idx2] = value1; \
        } \
    }


#define BITONICEXCHANGE128_512() \
    for (int i = 0; i < 8; i++) { \
        int idx1 = i * 2; \
        int idx2 = i * 2 + 1; \
        \
        if (k[idx1] < k[idx2]) { \
            key1 = k[idx1]; \
            value1 = v[idx1]; \
            k[idx1] = k[idx2]; \
            v[idx1] = v[idx2]; \
            k[idx2] = key1; \
            v[idx2] = value1; \
        } \
    }

#define BITONICSORT128_512() \
    BITONICSORT64_512() \
    BITONICMERGE128_512() \
    BITONICEXCHANGE32_512() \
    BITONICWARPEXCHANGE_512(16) \
    BITONICWARPEXCHANGE_512(8) \
    BITONICWARPEXCHANGE_512(4) \
    BITONICWARPEXCHANGE_512(2) \
    BITONICWARPEXCHANGE_512(1)

#define BITONICSORT256_512() \
    BITONICSORT128_512() \
    BITONICMERGE256_512() \
    BITONICEXCHANGE64_512() \
    BITONICEXCHANGE32_512() \
    BITONICWARPEXCHANGE_512(16) \
    BITONICWARPEXCHANGE_512(8) \
    BITONICWARPEXCHANGE_512(4) \
    BITONICWARPEXCHANGE_512(2) \
    BITONICWARPEXCHANGE_512(1)

#define BITONICSORT512_512() \
    BITONICSORT256_512() \
    BITONICMERGE512_512() \
    BITONICEXCHANGE128_512() \
    BITONICEXCHANGE64_512() \
    BITONICEXCHANGE32_512() \
    BITONICWARPEXCHANGE_512(16) \
    BITONICWARPEXCHANGE_512(8) \
    BITONICWARPEXCHANGE_512(4) \
    BITONICWARPEXCHANGE_512(2) \
    BITONICWARPEXCHANGE_512(1)    

#endif
