#include "amuquant/turboquant.hpp"

#include <cmath>
#include <random>
#include <algorithm>
#include <queue>
#include <cstring>

#if defined(__ARM_NEON)
#include <arm_neon.h>
#elif defined(__AVX2__)
#include <immintrin.h>
#endif

namespace amuquant {

namespace {

// Konstanta Centroid Lloyd-Max untuk Distribusi Normal Standar N(0, 1)
// 4-bit (16 tingkatan)
const float LLOYD_MAX_4BIT_CENTROIDS[16] = {
    -2.733f, -2.069f, -1.618f, -1.256f,
    -0.942f, -0.657f, -0.388f, -0.128f,
     0.128f,  0.388f,  0.657f,  0.942f,
     1.256f,  1.618f,  2.069f,  2.733f
};

// 2-bit (4 tingkatan)
const float LLOYD_MAX_2BIT_CENTROIDS[4] = {
    -1.510f, -0.453f, 0.453f, 1.510f
};

} // namespace

TurboQuantEngine::TurboQuantEngine(uint32_t dimension, uint8_t bit_width)
    : dimension_(dimension), bit_width_(bit_width) {
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
    // Menghasilkan matriks rotasi blok ortogonal deterministik (Blok ukuran B=64)
    // Blok ortogonal menjamin sifat Teorema Batas Pusat (CLT) dengan komputasi O(d * B)
    constexpr size_t BLOCK_SIZE = 64;
    size_t num_blocks = (dimension_ + BLOCK_SIZE - 1) / BLOCK_SIZE;
    orthogonal_matrix_.resize(num_blocks * BLOCK_SIZE * BLOCK_SIZE);

    std::mt19937 rng(42); // Seed deterministik
    std::normal_distribution<float> norm_dist(0.0f, 1.0f);

    for (size_t b = 0; b < num_blocks; ++b) {
        float* block = &orthogonal_matrix_[b * BLOCK_SIZE * BLOCK_SIZE];

        // 1. Inisialisasi matriks Gaussian acak untuk blok b
        for (size_t i = 0; i < BLOCK_SIZE * BLOCK_SIZE; ++i) {
            block[i] = norm_dist(rng);
        }

        // 2. Gram-Schmidt Orthogonalization untuk membuat matriks ortogonal murni (Q^T * Q = I)
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

            // Normalisasi vektor
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

        for (size_t i = 0; i < current_len; ++i) {
            const float* row = block + i * BLOCK_SIZE;
            float sum = 0.0f;

#if defined(__ARM_NEON)
            float32x4_t vsum = vdupq_n_f32(0.0f);
            size_t k = 0;
            for (; k + 4 <= current_len; k += 4) {
                float32x4_t vin = vld1q_f32(in + offset + k);
                float32x4_t vrow = vld1q_f32(row + k);
                vsum = vfmaq_f32(vsum, vin, vrow);
            }
            sum = vaddvq_f32(vsum);
            for (; k < current_len; ++k) {
                sum += (in + offset)[k] * row[k];
            }
#else
            for (size_t k = 0; k < current_len; ++k) {
                sum += (in + offset)[k] * row[k];
            }
#endif
            out[offset + i] = sum;
        }
    }
}

uint8_t TurboQuantEngine::quantize_scalar_4bit(float val) const {
    for (uint8_t i = 0; i < 15; ++i) {
        if (val < thresholds_4bit_[i]) {
            return i;
        }
    }
    return 15;
}

uint8_t TurboQuantEngine::quantize_scalar_2bit(float val) const {
    for (uint8_t i = 0; i < 3; ++i) {
        if (val < thresholds_2bit_[i]) {
            return i;
        }
    }
    return 3;
}

void TurboQuantEngine::insert(uint64_t chunk_id, const float* data, size_t len) {
    if (data == nullptr || len != dimension_) {
        return;
    }

    // 1. Hitung L2-Norm dari vektor masukan
    float norm_sq = 0.0f;
    for (size_t i = 0; i < len; ++i) {
        norm_sq += data[i] * data[i];
    }
    float norm = std::sqrt(norm_sq);
    float sigma = (norm > 1e-12f) ? (norm / std::sqrt(static_cast<float>(dimension_))) : 1e-6f;
    float inv_sigma = 1.0f / sigma;

    // 2. Terapkan rotasi ortogonal acak
    std::vector<float> rotated(dimension_);
    rotate_vector(data, rotated.data());

    // 3. Kuantisasi skalar Lloyd-Max dan pengepakan bit
    QuantizedVector qv;
    qv.chunk_id = chunk_id;
    qv.norm = norm;

    if (bit_width_ == 4) {
        // Mode 4-bit: 2 komponen per byte (d/2 byte)
        qv.packed_data.resize(dimension_ / 2);
        for (size_t i = 0; i < dimension_; i += 2) {
            float z0 = rotated[i] * inv_sigma;
            float z1 = (i + 1 < dimension_) ? (rotated[i + 1] * inv_sigma) : 0.0f;

            uint8_t code0 = quantize_scalar_4bit(z0);
            uint8_t code1 = quantize_scalar_4bit(z1);

            qv.packed_data[i / 2] = (code0 & 0x0F) | ((code1 & 0x0F) << 4);
        }
    } else {
        // Mode 2-bit: 4 komponen per byte (d/4 byte)
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

    // Menggunakan min-heap untuk melacak Top-K hasil terbesar
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
    std::reverse(results.begin(), results.end()); // Urutkan dari skor tertinggi
    return results;
}

void TurboQuantEngine::requantize_to_2bit() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (bit_width_ == 2) {
        return; // Sudah dalam presisi 2-bit
    }

    // Mengonversi secara in-place: setiap dua kode 4-bit diremapping ke satu kode 2-bit
    // 4 kode 4-bit (2 byte) dipadatkan menjadi 4 kode 2-bit (1 byte)
    for (auto& vec : vectors_) {
        std::vector<uint8_t> new_packed(dimension_ / 4);

        for (size_t i = 0; i < dimension_; i += 4) {
            // Ambil 4 kode 4-bit dari packed_data lama
            uint8_t b0 = vec.packed_data[i / 2];
            uint8_t b1 = vec.packed_data[(i + 2) / 2];

            uint8_t q4_0 = b0 & 0x0F;
            uint8_t q4_1 = (b0 >> 4) & 0x0F;
            uint8_t q4_2 = b1 & 0x0F;
            uint8_t q4_3 = (b1 >> 4) & 0x0F;

            // Pemetaan langsung: q2 = q4 / 4
            uint8_t q2_0 = q4_0 / 4;
            uint8_t q2_1 = q4_1 / 4;
            uint8_t q2_2 = q4_2 / 4;
            uint8_t q2_3 = q4_3 / 4;

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
