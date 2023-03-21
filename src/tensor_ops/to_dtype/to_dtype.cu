#include <cstdint>

#define CONVERSION_OP_INTO(RTYPE1, RTYPE2, CTYPE1, CTYPE2) \
extern "C" __global__ void RTYPE1##_to_##RTYPE2( \
    const size_t numel, \
    const CTYPE1 *inp, \
    CTYPE2 *out \
) { \
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; \
    if (i >= numel) { \
        return; \
    } \
    out[i] = inp[i]; \
}

#define CONVERSION_OP_FROM(RTYPE1, CTYPE1) \
    CONVERSION_OP_INTO(RTYPE1, f32, CTYPE1, float); \
    CONVERSION_OP_INTO(RTYPE1, f64, CTYPE1, double); \
    \
    CONVERSION_OP_INTO(RTYPE1, u8, CTYPE1, uint8_t); \
    CONVERSION_OP_INTO(RTYPE1, u16, CTYPE1, uint16_t); \
    CONVERSION_OP_INTO(RTYPE1, u32, CTYPE1, uint32_t); \
    CONVERSION_OP_INTO(RTYPE1, u64, CTYPE1, uint64_t); \
    CONVERSION_OP_INTO(RTYPE1, usize, CTYPE1, uintptr_t); \
    \
    CONVERSION_OP_INTO(RTYPE1, i8, CTYPE1, int8_t); \
    CONVERSION_OP_INTO(RTYPE1, i16, CTYPE1, int16_t); \
    CONVERSION_OP_INTO(RTYPE1, i32, CTYPE1, int32_t); \
    CONVERSION_OP_INTO(RTYPE1, i64, CTYPE1, int64_t); \
    CONVERSION_OP_INTO(RTYPE1, isize, CTYPE1, intptr_t); \

CONVERSION_OP_FROM(f32, float);
CONVERSION_OP_FROM(f64, double);

CONVERSION_OP_FROM(u8, uint8_t);
CONVERSION_OP_FROM(u16, uint16_t);
CONVERSION_OP_FROM(u32, uint32_t);
CONVERSION_OP_FROM(u64, uint64_t);
CONVERSION_OP_FROM(usize, uintptr_t);

CONVERSION_OP_FROM(i8, int8_t);
CONVERSION_OP_FROM(i16, int16_t);
CONVERSION_OP_FROM(i32, int32_t);
CONVERSION_OP_FROM(i64, int64_t);
CONVERSION_OP_FROM(isize, intptr_t);

CONVERSION_OP_FROM(bool, bool);
