/**
 * @file telemetry.hpp
 * @brief Platform-native memory and swap telemetry sampler.
 */

#ifndef AMUQUANT_TELEMETRY_HPP
#define AMUQUANT_TELEMETRY_HPP

#include "amuquant/types.hpp"

namespace amuquant {

class SystemTelemetry {
public:
    SystemTelemetry();
    ~SystemTelemetry() = default;

    /**
     * @brief Queries instantaneous host OS memory, swap utilization, and process RSS.
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

private:
    bool mock_mode_{false};
    MemoryStats mock_stats_{};
};

} // namespace amuquant

#endif // AMUQUANT_TELEMETRY_HPP
