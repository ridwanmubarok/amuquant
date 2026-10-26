#ifndef AMUQUANT_TYPES_HPP
#define AMUQUANT_TYPES_HPP

#include <cstdint>
#include <cstddef>
#include <string>

namespace amuquant {

/**
 * @brief Tiga rejim operasional kontroler adaptif AMU-QUANT.
 */
enum class Regime : uint8_t {
    REGIME_1_HIGH_THROUGHPUT = 1,
    REGIME_2_BALANCED_OFFLOAD = 2,
    REGIME_3_ULTRA_LOW_SURVIVAL = 3
};

/**
 * @brief Kode status eksekusi operasi AMU-QUANT.
 */
enum class Status : int32_t {
    OK = 0,
    ERR_INVALID_ARGUMENT = -1,
    ERR_OUT_OF_MEMORY = -2,
    ERR_NOT_INITIALIZED = -3,
    ERR_SYSTEM_CALL_FAILED = -4
};

/**
 * @brief Struktur statistik telemetri memori sistem dan proses.
 */
struct MemoryStats {
    uint64_t total_ram_bytes{0};
    uint64_t available_ram_bytes{0};
    uint64_t process_rss_bytes{0};
    uint64_t swap_used_bytes{0};
    uint64_t major_page_faults{0};
    double timestamp_sec{0.0};
};

/**
 * @brief Parameter konfigurasi kontroler adaptif AMU-QUANT.
 */
struct AMUQuantConfig {
    uint64_t memory_limit_bytes{14ULL * 1024ULL * 1024ULL * 1024ULL}; // Default 14 GB (untuk RAM 16 GB)
    uint64_t os_reserved_bytes{2ULL * 1024ULL * 1024ULL * 1024ULL};   // Cadangan kernel 2 GB
    
    // Parameter Skor Tekanan Prediktif Φ_t
    double alpha{0.65};
    double beta{0.35};
    double tau_horizon_sec{2.0};
    
    // Ambang Batas Transisi Rejim
    double theta_low{0.65};
    double theta_high{0.85};
    double delta_hyst{0.05};
    
    // Interval evaluasi telemetri
    uint32_t eval_interval_tokens{8};
    
    // Parameter Jendela Konteks KV-Cache
    uint32_t attention_sink_tokens{4};
    uint32_t recent_window_tokens{256};
    uint32_t min_recent_window_tokens{64}; // Batas darurat Regime 3
    
    // Parameter Model & Representasi Vektor
    uint32_t num_layers{64};          // Default layer
    uint32_t embedding_dim{4096};     // Dimensi vektor tersembunyi
    uint8_t initial_bit_width{4};     // Kuantisasi awal 4-bit
};

/**
 * @brief Hasil keputusan kontroler pada setiap langkah inferensi token.
 */
struct StepDecision {
    Regime regime{Regime::REGIME_1_HIGH_THROUGHPUT};
    uint32_t prefetch_depth{3};
    bool aggressive_evict{false};
    bool kv_pruned{false};
    bool requantized_to_2bit{false};
    double phi_score{0.0};
};

inline std::string regime_to_string(Regime r) {
    switch (r) {
        case Regime::REGIME_1_HIGH_THROUGHPUT:   return "REGIME_1_HIGH_THROUGHPUT";
        case Regime::REGIME_2_BALANCED_OFFLOAD:  return "REGIME_2_BALANCED_OFFLOAD";
        case Regime::REGIME_3_ULTRA_LOW_SURVIVAL: return "REGIME_3_ULTRA_LOW_SURVIVAL";
        default: return "UNKNOWN";
    }
}

} // namespace amuquant

#endif // AMUQUANT_TYPES_HPP
