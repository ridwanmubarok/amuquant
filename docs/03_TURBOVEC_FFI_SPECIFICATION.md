# Foreign Function Interface Specification (turbovec-ffi)

This document specifies the technical design, C-ABI signatures, and safety guarantees of **`turbovec-ffi`**, the zero-overhead foreign function bridge connecting C++ (`llama.cpp`) with the Rust `turbovec` vector compression engine.

---

## 1. Algorithmic Foundation: TurboQuant (Google Research, ICLR 2026)

The **TurboQuant** algorithm ([Zandieh et al., ICLR 2026](https://arxiv.org/abs/2504.19874)) provides training-free, in-process vector quantization with near-optimal distortion bounds under the Shannon rate-distortion limit:

```text
Raw Float Vector (FP16 / FP32, dim = d)
               │
               ▼
   [ Block-Orthogonal Rotation Q ]   ───> Transforms vector entries to i.i.d. Gaussian N(0, ||x||^2 / d)
               │
               ▼
  [ Lloyd-Max Scalar Quantizer ]    ───> Maps entries to optimal discrete centroids
               │
       ┌───────┴───────┐
       ▼               ▼
 4-bit Mode (16 levels) 2-bit Mode (4 levels)
 (Compression ~7.8x)    (Compression ~15.5x)
```

### Mathematical Pipeline:
1. **Random Block-Orthogonal Rotation:**  
   The input embedding $\mathbf{x} \in \mathbb{R}^d$ is partitioned into uniform blocks of size $B=64$. Each block is rotated by a precomputed deterministic orthogonal matrix $\mathbf{Q} \in \mathbb{R}^{B \times B}$ satisfying $\mathbf{Q}^T \mathbf{Q} = \mathbf{I}$:
   $$\mathbf{y} = \mathbf{Q} \mathbf{x}$$
   By the Central Limit Theorem, the rotated coordinates $y_i$ become approximately independent, zero-mean Gaussian random variables:
   $$y_i \sim \mathcal{N}\left(0, \frac{\|\mathbf{x}\|_2^2}{d}\right)$$
2. **Zero-Training Lloyd-Max Quantization:**  
   Because the distribution is analytically Gaussian, optimal quantization centroids and partition thresholds are precomputed once:
   * **4-bit Mode:** 16 centroids, consuming 0.5 bytes per component.
   * **2-bit Mode:** 4 centroids, consuming 0.25 bytes per component.
3. **SIMD-Accelerated Vector Dot Products:**  
   During context retrieval, query vectors compute cosine similarity against stored vectors using parallel register unpack instructions (ARM NEON or x86-64 AVX2/AVX-512).

---

## 2. C-ABI Interface Specification (`turbovec_ffi.h`)

The public C-ABI header is structured as follows:

```c
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
 * @brief Inserts a contiguous raw float vector into the index without intermediate copies.
 */
TurbovecStatus turbovec_insert_zero_copy(
    TurbovecIndexHandle* handle,
    uint64_t chunk_id,
    const float* float_ptr,
    size_t len
);

/**
 * @brief Performs SIMD-accelerated Top-K cosine similarity search.
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
 * @brief Requantizes all stored vectors from 4-bit to 2-bit representations in-place.
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
```

---

## 3. Rust Implementation (`ffi.rs`)

The Rust crate exports functions with `#[no_mangle]` and `extern "C"` linkage:

```rust
use std::panic::catch_unwind;
use std::slice;
use std::sync::RwLock;

pub struct TurbovecIndexHandle {
    pub inner: RwLock<TurboIndex>,
}

#[no_mangle]
pub unsafe extern "C" fn turbovec_insert_zero_copy(
    handle: *mut TurbovecIndexHandle,
    chunk_id: u64,
    float_ptr: *const f32,
    len: usize,
) -> TurbovecStatus {
    if handle.is_null() || float_ptr.is_null() {
        return TurbovecStatus::ErrNullPointer;
    }

    let result = catch_unwind(|| {
        let handle_ref = &*handle;
        // Non-allocating slice view over C++ memory
        let slice_view = slice::from_raw_parts(float_ptr, len);
        let mut index = handle_ref.inner.write().unwrap();
        index.insert_slice(chunk_id, slice_view);
    });

    match result {
        Ok(_) => TurbovecStatus::Ok,
        Err(_) => TurbovecStatus::ErrPanicCaught,
    }
}
```

---

## 4. In-Place 4-bit to 2-bit Requantization Mechanics

Under emergency memory pressure (Regime 3), AMU-QUANT condenses all archived 4-bit vectors to 2-bit representations in-place without heap reallocations.

Because each 4-bit nibble represents an index $q_4 \in [0, 15]$ into the 16 Lloyd-Max centroids, and the 4 centroids of the 2-bit quantizer span the exact same Gaussian quartile partitions, the transformation maps directly:

$$q_2 = \left\lfloor \frac{q_4}{4} \right\rfloor \in \{0, 1, 2, 3\}$$

Bit-level condensation packs four 4-bit nibbles (2 bytes) into four 2-bit dibits (1 byte):

```text
Original Byte 0: [  q4_1 (4b)  |  q4_0 (4b)  ]
Original Byte 1: [  q4_3 (4b)  |  q4_2 (4b)  ]
                          │
                          ▼
Compressed Byte: [ q2_3 (2b) | q2_2 (2b) | q2_1 (2b) | q2_0 (2b) ]
```

This reduces the physical RAM footprint of all archived context chunks by an additional $50\%$ in a single sequential sweep.
