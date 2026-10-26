/**
 * @file types.hpp
 * @brief Core types, operational regimes, and telemetry structures for AMU-QUANT.
 */

#ifndef AMUQUANT_TYPES_HPP
#define AMUQUANT_TYPES_HPP

#include <cstdint>
#include <cstddef>
#include <string>

namespace amuquant {

/**
 * @brief Tri-modal operational regimes of the co-adaptive controller.
 */
enum class Regime : uint8_t {
    REGIME_1_HIGH_THROUGHPUT = 1,
    REGIME_2_BALANCED_OFFLOAD = 2,
    REGIME_3_ULTRA_LOW_SURVIVAL = 3
};

/**
 * @brief Execution status codes for AMU-QUANT operations.
 */
enum class Status : int32_t {
    OK = 0,
    ERR_INVALID_ARGUMENT = -1,
    ERR_OUT_OF_MEMORY = -2,
    ERR_NOT_INITIALIZED = -3,
    ERR_SYSTEM_CALL_FAILED = -4
};

/**
 * @brief Host and process memory telemetry snapshot.
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
 * @brief Configuration parameters for the AMU-QUANT runtime engine.
 */
struct AMUQuantConfig {
    uint64_t memory_limit_bytes{14ULL * 1024ULL * 1024ULL * 1024ULL}; // Target working budget
    uint64_t os_reserved_bytes{2ULL * 1024ULL * 1024ULL * 1024ULL};   // Reserved for OS kernel stability
    
    // Predictive pressure score weights: Phi_t = alpha * max(U_proc, U_sys) + beta * P_growth + gamma * P_swap
    double alpha{0.60};
    double beta{0.25};
    double gamma{0.15};
    double tau_horizon_sec{2.0};
    
    // Regime transition thresholds with hysteresis band
    double theta_low{0.65};
    double theta_high{0.85};
    double delta_hyst{0.05};
    
    // Telemetry sampling interval
    uint32_t eval_interval_tokens{8};
    
    // Attention sink and recent context window geometry
    uint32_t attention_sink_tokens{4};
    uint32_t recent_window_tokens{256};
    uint32_t min_recent_window_tokens{64}; // Emergency clamp for Regime 3
    uint32_t min_eviction_batch{1};        // Minimum token slice to trigger batched eviction
    
    // Model architecture parameters
    uint32_t num_layers{64};
    uint32_t embedding_dim{4096};
    uint8_t initial_bit_width{4};
};

/**
 * @brief Controller arbitration decision emitted per token step.
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
