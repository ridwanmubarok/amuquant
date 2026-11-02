/**
 * @file telemetry.hpp
 * @brief Platform-native memory and swap telemetry sampler.
 */

#ifndef AMUQUANT_TELEMETRY_HPP
#define AMUQUANT_TELEMETRY_HPP

#include "amuquant/types.hpp"
#include <memory>

namespace amuquant {

class LayerManager;
class KVSurgeon;

class SystemTelemetry {
public:
    SystemTelemetry();
    ~SystemTelemetry() = default;

    /**
     * @brief Queries instantaneous host OS memory, swap utilization, process RSS, physical footprint,
     *        model page residency via mincore, and KV logical capacity.
     */
    MemoryStats sample();

    /**
     * @brief Monotonic high-resolution timestamp in fractional seconds.
     */
    static double get_time_seconds();

    /**
     * @brief Injects synthetic metrics for deterministic unit testing.
     */
    void set_mock_mode(bool enable);
    void set_mock_stats(const MemoryStats& stats);

    void set_layer_manager(std::shared_ptr<LayerManager> mgr) { layer_mgr_ = mgr; }
    void set_kv_surgeon(std::shared_ptr<KVSurgeon> kv, uint32_t num_layers, uint32_t kv_heads, uint32_t head_dim, uint32_t bytes_per_elem = 2) {
        kv_surgeon_ = kv;
        num_layers_ = num_layers;
        kv_heads_ = kv_heads;
        head_dim_ = head_dim;
        bytes_per_elem_ = bytes_per_elem;
    }

private:
    bool mock_mode_{false};
    MemoryStats mock_stats_{};
    double last_sample_time_{0.0};
    MemoryStats cached_stats_{};
    std::shared_ptr<LayerManager> layer_mgr_;
    std::shared_ptr<KVSurgeon> kv_surgeon_;
    uint32_t num_layers_{0};
    uint32_t kv_heads_{0};
    uint32_t head_dim_{0};
    uint32_t bytes_per_elem_{2};
};

} // namespace amuquant

#endif // AMUQUANT_TELEMETRY_HPP
