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
     * @brief Menyisipkan vektor float tanpa penyalinan perantara (Zero-Copy).
     */
    void insert(uint64_t chunk_id, const float* data, size_t len);

    /**
     * @brief Menjalankan pencarian Top-K kemiripan kosinus berbasis SIMD.
     */
    std::vector<SearchResult> search_simd(const float* query_data, size_t len, uint32_t top_k) const;

    /**
     * @brief Mengompresi ulang seluruh vektor yang tersimpan dari 4-bit ke 2-bit in-place.
     */
    void requantize_to_2bit();

    /**
     * @brief Mendapatkan penggunaan memori heap dalam satuan byte.
     */
    size_t get_memory_bytes() const;

    uint32_t dimension() const { return dimension_; }
    uint8_t bit_width() const { return bit_width_; }
    size_t size() const;

private:
    uint32_t dimension_;
    uint8_t bit_width_; // 4 atau 2
    mutable std::mutex mutex_;

    // Matriks rotasi ortogonal acak Q (tersimpan dalam blok teratur untuk efisiensi SIMD cache)
    std::vector<float> orthogonal_matrix_;

    // Tabel batas (thresholds) dan centroid Lloyd-Max
    std::vector<float> centroids_4bit_;
    std::vector<float> thresholds_4bit_;
    std::vector<float> centroids_2bit_;
    std::vector<float> thresholds_2bit_;

    // Kumpulan vektor terkompresi
    std::vector<QuantizedVector> vectors_;

    void init_codebooks();
    void generate_orthogonal_matrix();
    void rotate_vector(const float* in, float* out) const;
    uint8_t quantize_scalar_4bit(float val) const;
    uint8_t quantize_scalar_2bit(float val) const;
};

} // namespace amuquant

#endif // AMUQUANT_TURBOQUANT_HPP
