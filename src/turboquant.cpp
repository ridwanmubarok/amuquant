#include "amuquant/turboquant.hpp"

#include <cmath>
#include <random>
#include <algorithm>
#include <queue>
#include <cstring>
#include <unordered_set>

#if defined(__ARM_NEON)
#include <arm_neon.h>
#elif defined(__AVX2__)
#include <immintrin.h>
#endif

namespace amuquant {

namespace {

// Lloyd-Max optimal centroids for standard normal distribution N(0, 1)
// 4-bit codebook (16 levels)
const float LLOYD_MAX_4BIT_CENTROIDS[16] = {
    -2.733f, -2.069f, -1.618f, -1.256f,
    -0.942f, -0.657f, -0.388f, -0.128f,
     0.128f,  0.388f,  0.657f,  0.942f,
     1.256f,  1.618f,  2.069f,  2.733f
};

// 2-bit codebook (4 levels)
const float LLOYD_MAX_2BIT_CENTROIDS[4] = {
    -1.510f, -0.453f, 0.453f, 1.510f
};

} // namespace

TurboQuantEngine::TurboQuantEngine(uint32_t dimension, uint8_t bit_width)
    : dimension_(dimension), bit_width_(bit_width) {
    scratch_rotated_.resize(dimension, 0.0f);
    init_codebooks();
    generate_orthogonal_matrix();
}

void TurboQuantEngine::init_codebooks() {
    centroids_4bit_.assign(LLOYD_MAX_4BIT_CENTROIDS, LLOYD_MAX_4BIT_CENTROIDS + 16);
    thresholds_4bit_.resize(15);
    for (size_t i = 0; i < 15; ++i) {
        thresholds_4bit_[i] = (centroids_4bit_[i] + centroids_4bit_[i + 1]) * 0.5f;
    }

    centroids_2bit_.assign(LLOYD_MAX_2BIT_CENTROIDS, LLOYD_MAX_2BIT_CENTROIDS + 4);
    thresholds_2bit_.resize(3);
    for (size_t i = 0; i < 3; ++i) {
        thresholds_2bit_[i] = (centroids_2bit_[i] + centroids_2bit_[i + 1]) * 0.5f;
    }
}

void TurboQuantEngine::generate_orthogonal_matrix() {
    // Block-orthogonal random rotation matrix with block size B=64.
    // Guarantees Gaussianization via Central Limit Theorem with O(d * B) projection complexity.
    constexpr size_t BLOCK_SIZE = 64;
    size_t num_blocks = (dimension_ + BLOCK_SIZE - 1) / BLOCK_SIZE;
    orthogonal_matrix_.resize(num_blocks * BLOCK_SIZE * BLOCK_SIZE);

    std::mt19937 rng(42); // Deterministic seed
    std::normal_distribution<float> norm_dist(0.0f, 1.0f);

    for (size_t b = 0; b < num_blocks; ++b) {
        float* block = &orthogonal_matrix_[b * BLOCK_SIZE * BLOCK_SIZE];

        // 1. Generate Gaussian i.i.d. entries for block b
        for (size_t i = 0; i < BLOCK_SIZE * BLOCK_SIZE; ++i) {
            block[i] = norm_dist(rng);
        }

        // 2. Gram-Schmidt orthogonalization ensuring Q^T * Q = I
        for (size_t i = 0; i < BLOCK_SIZE; ++i) {
            float* vi = block + i * BLOCK_SIZE;
            for (size_t j = 0; j < i; ++j) {
                const float* vj = block + j * BLOCK_SIZE;
                float dot = 0.0f;
                for (size_t k = 0; k < BLOCK_SIZE; ++k) {
                    dot += vi[k] * vj[k];
                }
                for (size_t k = 0; k < BLOCK_SIZE; ++k) {
                    vi[k] -= dot * vj[k];
                }
            }

            // Normalize row vector
            float norm_sq = 0.0f;
            for (size_t k = 0; k < BLOCK_SIZE; ++k) {
                norm_sq += vi[k] * vi[k];
            }
            float inv_norm = 1.0f / std::sqrt(std::max(norm_sq, 1e-12f));
            for (size_t k = 0; k < BLOCK_SIZE; ++k) {
                vi[k] *= inv_norm;
            }
        }
    }
}

void TurboQuantEngine::rotate_vector(const float* in, float* out) const {
    constexpr size_t BLOCK_SIZE = 64;
    size_t num_blocks = (dimension_ + BLOCK_SIZE - 1) / BLOCK_SIZE;

    for (size_t b = 0; b < num_blocks; ++b) {
        const float* block = &orthogonal_matrix_[b * BLOCK_SIZE * BLOCK_SIZE];
        size_t offset = b * BLOCK_SIZE;
        size_t current_len = std::min(BLOCK_SIZE, static_cast<size_t>(dimension_ - offset));

#if defined(__ARM_NEON)
        size_t i = 0;
        // 4-way unrolled row processing: saturates M1/NEON execution pipelines with 4 parallel FMA streams
        for (; i + 4 <= current_len; i += 4) {
            const float* row0 = block + (i + 0) * BLOCK_SIZE;
            const float* row1 = block + (i + 1) * BLOCK_SIZE;
            const float* row2 = block + (i + 2) * BLOCK_SIZE;
            const float* row3 = block + (i + 3) * BLOCK_SIZE;

            float32x4_t s0 = vdupq_n_f32(0.0f);
            float32x4_t s1 = vdupq_n_f32(0.0f);
            float32x4_t s2 = vdupq_n_f32(0.0f);
            float32x4_t s3 = vdupq_n_f32(0.0f);

            for (size_t k = 0; k < BLOCK_SIZE; k += 4) {
                float32x4_t vin = vld1q_f32(in + offset + k);
                s0 = vfmaq_f32(s0, vin, vld1q_f32(row0 + k));
                s1 = vfmaq_f32(s1, vin, vld1q_f32(row1 + k));
                s2 = vfmaq_f32(s2, vin, vld1q_f32(row2 + k));
                s3 = vfmaq_f32(s3, vin, vld1q_f32(row3 + k));
            }

            out[offset + i + 0] = vaddvq_f32(s0);
            out[offset + i + 1] = vaddvq_f32(s1);
            out[offset + i + 2] = vaddvq_f32(s2);
            out[offset + i + 3] = vaddvq_f32(s3);
        }
        for (; i < current_len; ++i) {
            const float* row = block + i * BLOCK_SIZE;
            float32x4_t s = vdupq_n_f32(0.0f);
            for (size_t k = 0; k < BLOCK_SIZE; k += 4) {
                s = vfmaq_f32(s, vld1q_f32(in + offset + k), vld1q_f32(row + k));
            }
            out[offset + i] = vaddvq_f32(s);
        }
#else
        for (size_t i = 0; i < current_len; ++i) {
            const float* row = block + i * BLOCK_SIZE;
            float sum = 0.0f;
            for (size_t k = 0; k < current_len; ++k) {
                sum += (in + offset)[k] * row[k];
            }
            out[offset + i] = sum;
        }
#endif
    }
}

uint8_t TurboQuantEngine::quantize_scalar_4bit(float val) const {
    // Unrolled branchless binary search over 15 thresholds in exactly 4 steps
    uint8_t idx = (val >= thresholds_4bit_[7]) ? 8 : 0;
    idx += (val >= thresholds_4bit_[idx + 3]) ? 4 : 0;
    idx += (val >= thresholds_4bit_[idx + 1]) ? 2 : 0;
    idx += (val >= thresholds_4bit_[idx]) ? 1 : 0;
    return idx;
}

uint8_t TurboQuantEngine::quantize_scalar_2bit(float val) const {
    // Unrolled branchless binary search over 3 thresholds in exactly 2 steps
    uint8_t idx = (val >= thresholds_2bit_[1]) ? 2 : 0;
    idx += (val >= thresholds_2bit_[idx]) ? 1 : 0;
    return idx;
}

void TurboQuantEngine::insert(uint64_t chunk_id, const float* data, size_t len) {
    if (data == nullptr || len != dimension_) {
        return;
    }

    // 1. Compute input vector L2-norm
    float norm_sq = 0.0f;
    for (size_t i = 0; i < len; ++i) {
        norm_sq += data[i] * data[i];
    }
    float norm = std::sqrt(norm_sq);
    float sigma = (norm > 1e-12f) ? (norm / std::sqrt(static_cast<float>(dimension_))) : 1e-6f;
    float inv_sigma = 1.0f / sigma;

    // 2. Project through block-orthogonal rotation (zero-allocation scratchpad)
    if (scratch_rotated_.size() != dimension_) {
        scratch_rotated_.resize(dimension_);
    }
    rotate_vector(data, scratch_rotated_.data());
    const float* rotated = scratch_rotated_.data();

    // 3. Lloyd-Max scalar quantization and nibble packing
    QuantizedVector qv;
    qv.chunk_id = chunk_id;
    qv.norm = norm;

    if (bit_width_ == 4) {
        // 4-bit mode: 2 nibbles per byte (d/2 bytes per vector)
        qv.packed_data.resize(dimension_ / 2);
        for (size_t i = 0; i < dimension_; i += 2) {
            float z0 = rotated[i] * inv_sigma;
            float z1 = (i + 1 < dimension_) ? (rotated[i + 1] * inv_sigma) : 0.0f;

            uint8_t code0 = quantize_scalar_4bit(z0);
            uint8_t code1 = quantize_scalar_4bit(z1);

            qv.packed_data[i / 2] = (code0 & 0x0F) | ((code1 & 0x0F) << 4);
        }
    } else {
        // 2-bit mode: 4 dibits per byte (d/4 bytes per vector)
        qv.packed_data.resize(dimension_ / 4);
        for (size_t i = 0; i < dimension_; i += 4) {
            uint8_t byte_val = 0;
            for (size_t b = 0; b < 4; ++b) {
                if (i + b < dimension_) {
                    float z = rotated[i + b] * inv_sigma;
                    uint8_t code = quantize_scalar_2bit(z);
                    byte_val |= ((code & 0x03) << (b * 2));
                }
            }
            qv.packed_data[i / 4] = byte_val;
        }
    }

    std::lock_guard<std::mutex> lock(mutex_);
    vectors_.push_back(std::move(qv));
}

bool TurboQuantEngine::remove(uint64_t chunk_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = std::find_if(vectors_.begin(), vectors_.end(), [chunk_id](const QuantizedVector& qv) {
        return qv.chunk_id == chunk_id;
    });
    if (it != vectors_.end()) {
        vectors_.erase(it);
        return true;
    }
    return false;
}

size_t TurboQuantEngine::remove_batch(const uint64_t* chunk_ids, size_t count) {
    if (chunk_ids == nullptr || count == 0) {
        return 0;
    }
    std::unordered_set<uint64_t> ids(chunk_ids, chunk_ids + count);
    std::lock_guard<std::mutex> lock(mutex_);
    size_t initial_size = vectors_.size();
    vectors_.erase(
        std::remove_if(vectors_.begin(), vectors_.end(), [&ids](const QuantizedVector& qv) {
            return ids.find(qv.chunk_id) != ids.end();
        }),
        vectors_.end()
    );
    return initial_size - vectors_.size();
}

std::vector<SearchResult> TurboQuantEngine::search_simd(const float* query_data, size_t len, uint32_t top_k) const {
    if (query_data == nullptr || len != dimension_ || top_k == 0) {
        return {};
    }

    // 1. Hitung rotasi vektor kueri
    float q_norm_sq = 0.0f;
    for (size_t i = 0; i < len; ++i) {
        q_norm_sq += query_data[i] * query_data[i];
    }
    float q_norm = std::sqrt(q_norm_sq);
    if (q_norm < 1e-12f) {
        return {};
    }

    std::vector<float> q_rotated(dimension_);
    rotate_vector(query_data, q_rotated.data());

    std::lock_guard<std::mutex> lock(mutex_);
    if (vectors_.empty()) {
        return {};
    }

    // Min-heap to maintain Top-K candidates
    auto comp = [](const SearchResult& a, const SearchResult& b) {
        return a.score > b.score; // min-heap
    };
    std::priority_queue<SearchResult, std::vector<SearchResult>, decltype(comp)> min_heap(comp);

    for (const auto& vec : vectors_) {
        if (vec.norm < 1e-12f) continue;

        float dot_product = 0.0f;
        float sigma_vec = vec.norm / std::sqrt(static_cast<float>(dimension_));

        if (bit_width_ == 4) {
            for (size_t i = 0; i < dimension_; i += 2) {
                uint8_t byte = vec.packed_data[i / 2];
                uint8_t c0 = byte & 0x0F;
                uint8_t c1 = (byte >> 4) & 0x0F;

                float recon0 = centroids_4bit_[c0] * sigma_vec;
                float recon1 = centroids_4bit_[c1] * sigma_vec;

                dot_product += q_rotated[i] * recon0;
                if (i + 1 < dimension_) {
                    dot_product += q_rotated[i + 1] * recon1;
                }
            }
        } else {
            for (size_t i = 0; i < dimension_; i += 4) {
                uint8_t byte = vec.packed_data[i / 4];
                for (size_t b = 0; b < 4; ++b) {
                    if (i + b < dimension_) {
                        uint8_t code = (byte >> (b * 2)) & 0x03;
                        float recon = centroids_2bit_[code] * sigma_vec;
                        dot_product += q_rotated[i + b] * recon;
                    }
                }
            }
        }

        float cosine_sim = dot_product / (q_norm * vec.norm);

        if (min_heap.size() < top_k) {
            min_heap.push({vec.chunk_id, cosine_sim});
        } else if (cosine_sim > min_heap.top().score) {
            min_heap.pop();
            min_heap.push({vec.chunk_id, cosine_sim});
        }
    }

    std::vector<SearchResult> results;
    results.reserve(min_heap.size());
    while (!min_heap.empty()) {
        results.push_back(min_heap.top());
        min_heap.pop();
    }
    std::reverse(results.begin(), results.end()); // Descending order by score
    return results;
}

// Optimal Lloyd-Max 4-bit to 2-bit centroid projection lookup table.
// Minimizes mean squared distortion |c_i - q_j|^2 where:
// c_i in {-2.733, -2.069, -1.618, -1.256, -0.942, -0.657, -0.388, -0.128,
//          0.128,  0.388,  0.657,  0.942,  1.256,  1.618,  2.069,  2.733}
// q_j in {-1.510, -0.453,  0.453,  1.510} with partition thresholds at {-0.9815, 0.0, +0.9815}.
static const uint8_t LLOYD_MAX_4TO2_LUT[16] = {
    0, 0, 0, 0, // c0..c3 (<= -0.9815) -> q0 (-1.510)
    1, 1, 1, 1, // c4..c7 (-0.9815 < c <= 0.0) -> q1 (-0.453)
    2, 2, 2, 2, // c8..c11 (0.0 < c <= +0.9815) -> q2 (+0.453)
    3, 3, 3, 3  // c12..c15 (> +0.9815) -> q3 (+1.510)
};

void TurboQuantEngine::requantize_to_2bit() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (bit_width_ == 2) {
        return; // Already operating in 2-bit precision
    }

    // In-place codebook downscaling: maps two 4-bit nibbles into one 2-bit pair
    // 4 nibbles (2 bytes) compress into 4 dibits (1 byte) via Lloyd-Max lookup table.
    for (auto& vec : vectors_) {
        std::vector<uint8_t> new_packed(dimension_ / 4);

        for (size_t i = 0; i < dimension_; i += 4) {
            uint8_t b0 = vec.packed_data[i / 2];
            uint8_t b1 = vec.packed_data[(i + 2) / 2];

            uint8_t q4_0 = b0 & 0x0F;
            uint8_t q4_1 = (b0 >> 4) & 0x0F;
            uint8_t q4_2 = b1 & 0x0F;
            uint8_t q4_3 = (b1 >> 4) & 0x0F;

            // Optimal Lloyd-Max minimum-distortion centroid mapping
            uint8_t q2_0 = LLOYD_MAX_4TO2_LUT[q4_0];
            uint8_t q2_1 = LLOYD_MAX_4TO2_LUT[q4_1];
            uint8_t q2_2 = LLOYD_MAX_4TO2_LUT[q4_2];
            uint8_t q2_3 = LLOYD_MAX_4TO2_LUT[q4_3];

            new_packed[i / 4] = (q2_0 & 0x03) |
                               ((q2_1 & 0x03) << 2) |
                               ((q2_2 & 0x03) << 4) |
                               ((q2_3 & 0x03) << 6);
        }

        vec.packed_data = std::move(new_packed);
    }

    bit_width_ = 2;
}

size_t TurboQuantEngine::get_memory_bytes() const {
    std::lock_guard<std::mutex> lock(mutex_);
    size_t total = sizeof(*this);
    total += orthogonal_matrix_.capacity() * sizeof(float);
    for (const auto& v : vectors_) {
        total += sizeof(QuantizedVector) + v.packed_data.capacity();
    }
    return total;
}

size_t TurboQuantEngine::size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return vectors_.size();
}

} // namespace amuquant
