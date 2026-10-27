#include "amuquant/turbovec_ffi.h"
#include "amuquant/turboquant.hpp"

#include <new>

struct TurbovecIndexHandle {
    amuquant::TurboQuantEngine engine;

    TurbovecIndexHandle(uint32_t dim, uint8_t bit_width)
        : engine(dim, bit_width) {}
};

extern "C" {

TurbovecStatus turbovec_init(
    uint32_t dimension,
    uint8_t bit_width,
    TurbovecIndexHandle** out_handle
) {
    if (out_handle == nullptr) {
        return TURBOVEC_ERR_NULL_POINTER;
    }
    if (dimension == 0) {
        return TURBOVEC_ERR_INVALID_DIMENSION;
    }
    if (bit_width != 4 && bit_width != 2) {
        return TURBOVEC_ERR_UNSUPPORTED_BIT_WIDTH;
    }

    try {
        *out_handle = new TurbovecIndexHandle(dimension, bit_width);
        return TURBOVEC_STATUS_OK;
    } catch (const std::bad_alloc&) {
        return TURBOVEC_ERR_OUT_OF_MEMORY;
    } catch (...) {
        return TURBOVEC_ERR_PANIC_CAUGHT;
    }
}

TurbovecStatus turbovec_insert_zero_copy(
    TurbovecIndexHandle* handle,
    uint64_t chunk_id,
    const float* float_ptr,
    size_t len
) {
    if (handle == nullptr || float_ptr == nullptr) {
        return TURBOVEC_ERR_NULL_POINTER;
    }

    try {
        handle->engine.insert(chunk_id, float_ptr, len);
        return TURBOVEC_STATUS_OK;
    } catch (const std::bad_alloc&) {
        return TURBOVEC_ERR_OUT_OF_MEMORY;
    } catch (...) {
        return TURBOVEC_ERR_PANIC_CAUGHT;
    }
}

TurbovecStatus turbovec_remove(
    TurbovecIndexHandle* handle,
    uint64_t chunk_id
) {
    if (handle == nullptr) {
        return TURBOVEC_ERR_NULL_POINTER;
    }

    try {
        handle->engine.remove(chunk_id);
        return TURBOVEC_STATUS_OK;
    } catch (...) {
        return TURBOVEC_ERR_PANIC_CAUGHT;
    }
}

TurbovecStatus turbovec_remove_batch(
    TurbovecIndexHandle* handle,
    const uint64_t* chunk_ids,
    size_t count
) {
    if (handle == nullptr) {
        return TURBOVEC_ERR_NULL_POINTER;
    }
    if (chunk_ids == nullptr && count > 0) {
        return TURBOVEC_ERR_NULL_POINTER;
    }

    try {
        handle->engine.remove_batch(chunk_ids, count);
        return TURBOVEC_STATUS_OK;
    } catch (...) {
        return TURBOVEC_ERR_PANIC_CAUGHT;
    }
}

TurbovecStatus turbovec_search_simd(
    const TurbovecIndexHandle* handle,
    const float* query_ptr,
    uint32_t top_k,
    uint64_t* out_ids,
    float* out_scores,
    uint32_t* out_count
) {
    if (handle == nullptr || query_ptr == nullptr || out_ids == nullptr || out_scores == nullptr || out_count == nullptr) {
        return TURBOVEC_ERR_NULL_POINTER;
    }

    try {
        auto results = handle->engine.search_simd(query_ptr, handle->engine.dimension(), top_k);
        uint32_t count = static_cast<uint32_t>(results.size());
        for (uint32_t i = 0; i < count; ++i) {
            out_ids[i] = results[i].id;
            out_scores[i] = results[i].score;
        }
        *out_count = count;
        return TURBOVEC_STATUS_OK;
    } catch (...) {
        return TURBOVEC_ERR_PANIC_CAUGHT;
    }
}

TurbovecStatus turbovec_requantize_to_2bit(
    TurbovecIndexHandle* handle
) {
    if (handle == nullptr) {
        return TURBOVEC_ERR_NULL_POINTER;
    }

    try {
        handle->engine.requantize_to_2bit();
        return TURBOVEC_STATUS_OK;
    } catch (...) {
        return TURBOVEC_ERR_PANIC_CAUGHT;
    }
}

TurbovecStatus turbovec_get_memory_usage(
    const TurbovecIndexHandle* handle,
    size_t* out_bytes_used
) {
    if (handle == nullptr || out_bytes_used == nullptr) {
        return TURBOVEC_ERR_NULL_POINTER;
    }

    try {
        *out_bytes_used = handle->engine.get_memory_bytes();
        return TURBOVEC_STATUS_OK;
    } catch (...) {
        return TURBOVEC_ERR_PANIC_CAUGHT;
    }
}

TurbovecStatus turbovec_free(
    TurbovecIndexHandle* handle
) {
    if (handle == nullptr) {
        return TURBOVEC_ERR_NULL_POINTER;
    }

    try {
        delete handle;
        return TURBOVEC_STATUS_OK;
    } catch (...) {
        return TURBOVEC_ERR_PANIC_CAUGHT;
    }
}

} // extern "C"
