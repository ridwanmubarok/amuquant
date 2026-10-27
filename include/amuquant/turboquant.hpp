/**
 * @file turboquant.hpp
 * @brief High-performance TurboQuant vector quantization engine.
 */

#ifndef AMUQUANT_TURBOQUANT_HPP
#define AMUQUANT_TURBOQUANT_HPP

#include <vector>
#include <cstdint>
#include <cstddef>
#include <memory>
#include <mutex>

namespace amuquant {

struct QuantizedVector {
    uint64_t chunk_id{0};
    float norm{0.0f};
    std::vector<uint8_t> packed_data;
};

struct SearchResult {
    uint64_t id{0};
    float score{0.0f};
};

class TurboQuantEngine {
public:
    TurboQuantEngine(uint32_t dimension, uint8_t bit_width);
    ~TurboQuantEngine() = default;

    /**
     * @brief Quantizes and inserts a contiguous float vector.
     */
    void insert(uint64_t chunk_id, const float* data, size_t len);

    /**
     * @brief Removes a quantized vector by its chunk identifier.
     * Used for transactional rollback when an eviction batch fails midway.
     * @return true if the chunk was found and removed; false otherwise.
     */
    bool remove(uint64_t chunk_id);

    /**
     * @brief Removes multiple quantized vectors by chunk identifiers in a single pass.
     * @return Number of vectors removed.
     */
    size_t remove_batch(const uint64_t* chunk_ids, size_t count);

    /**
     * @brief Computes Top-K cosine similarity candidates using SIMD-accelerated dot products.
     */
    std::vector<SearchResult> search_simd(const float* query_data, size_t len, uint32_t top_k) const;

    /**
     * @brief Requantizes all stored vectors from 4-bit to 2-bit representations in-place.
     */
    void requantize_to_2bit();

    /**
     * @brief Computes heap allocation footprint for indexed vectors in bytes.
     */
    size_t get_memory_bytes() const;

    uint32_t dimension() const { return dimension_; }
    uint8_t bit_width() const { return bit_width_; }
    size_t size() const;

    uint8_t quantize_scalar_4bit(float val) const;
    uint8_t quantize_scalar_2bit(float val) const;

private:
    uint32_t dimension_;
    uint8_t bit_width_;
    mutable std::mutex mutex_;

    // Block-diagonal random orthogonal rotation matrix Q for SIMD cache locality
    std::vector<float> orthogonal_matrix_;

    // Lloyd-Max quantization codebooks and thresholds for standard normal N(0, 1)
    std::vector<float> centroids_4bit_;
    std::vector<float> thresholds_4bit_;
    std::vector<float> centroids_2bit_;
    std::vector<float> thresholds_2bit_;
    std::vector<QuantizedVector> vectors_;
    mutable std::vector<float> scratch_rotated_;

    void init_codebooks();
    void generate_orthogonal_matrix();
    void rotate_vector(const float* in, float* out) const;
};

} // namespace amuquant

#endif // AMUQUANT_TURBOQUANT_HPP
