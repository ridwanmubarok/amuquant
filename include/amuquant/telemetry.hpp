#ifndef AMUQUANT_TELEMETRY_HPP
#define AMUQUANT_TELEMETRY_HPP

#include "amuquant/types.hpp"

namespace amuquant {

class SystemTelemetry {
public:
    SystemTelemetry();
    ~SystemTelemetry() = default;

    /**
     * @brief Mengambil sampel metrik memori host OS dan proses secara instan.
     */
    MemoryStats sample();

    /**
     * @brief Mengambil waktu saat ini dalam detik resolusi tinggi.
     */
    static double get_time_seconds();

    /**
     * @brief Mode injeksi pengujian sintetik.
     */
    void set_mock_mode(bool enable);
    void set_mock_stats(const MemoryStats& stats);

private:
    bool mock_mode_{false};
    MemoryStats mock_stats_{};
};

} // namespace amuquant

#endif // AMUQUANT_TELEMETRY_HPP
