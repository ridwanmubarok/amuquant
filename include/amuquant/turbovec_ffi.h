/**
 * @file turbovec_ffi.h
 * @brief C-ABI Foreign Function Interface for TurboQuant vector index.
 */

#ifndef AMUQUANT_TURBOVEC_FFI_H
#define AMUQUANT_TURBOVEC_FFI_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    TURBOVEC_STATUS_OK = 0,
    TURBOVEC_ERR_NULL_POINTER = -1,
    TURBOVEC_ERR_INVALID_DIMENSION = -2,
    TURBOVEC_ERR_UNSUPPORTED_BIT_WIDTH = -3,
    TURBOVEC_ERR_OUT_OF_MEMORY = -4,
    TURBOVEC_ERR_PANIC_CAUGHT = -5
} TurbovecStatus;

typedef struct TurbovecIndexHandle TurbovecIndexHandle;

/**
 * @brief Allocates and initializes an in-process TurboQuant vector index.
 */
TurbovecStatus turbovec_init(
    uint32_t dimension,
    uint8_t bit_width,
    TurbovecIndexHandle** out_handle
);

/**
 * @brief Inserts a contiguous raw float vector into the quantized index.
 */
TurbovecStatus turbovec_insert_zero_copy(
    TurbovecIndexHandle* handle,
    uint64_t chunk_id,
    const float* float_ptr,
    size_t len
);

/**
 * @brief Removes a previously inserted vector by its chunk identifier.
 * Provides transactional rollback capabilities when an eviction batch fails midway.
 */
TurbovecStatus turbovec_remove(
    TurbovecIndexHandle* handle,
    uint64_t chunk_id
);

/**
 * @brief Removes a batch of vectors by their chunk identifiers.
 */
TurbovecStatus turbovec_remove_batch(
    TurbovecIndexHandle* handle,
    const uint64_t* chunk_ids,
    size_t count
);

/**
 * @brief Queries Top-K cosine similarity matches via SIMD dot-product acceleration.
 */
TurbovecStatus turbovec_search_simd(
    const TurbovecIndexHandle* handle,
    const float* query_vector,
    uint32_t top_k,
    uint64_t* out_ids,
    float* out_scores,
    uint32_t* out_count
);

/**
 * @brief Requantizes indexed vectors from 4-bit to 2-bit representations in-place.
 */
TurbovecStatus turbovec_requantize_to_2bit(
    TurbovecIndexHandle* handle
);

/**
 * @brief Returns total heap allocation allocated by indexed vectors in bytes.
 */
TurbovecStatus turbovec_get_memory_usage(
    const TurbovecIndexHandle* handle,
    size_t* out_bytes
);

/**
 * @brief Frees the vector index instance and releases heap allocations.
 */
TurbovecStatus turbovec_free(
    TurbovecIndexHandle* handle
);

#ifdef __cplusplus
}
#endif

#endif // AMUQUANT_TURBOVEC_FFI_H
