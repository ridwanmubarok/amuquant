#include "amuquant/controller.hpp"
#include "amuquant/telemetry.hpp"
#include "amuquant/layer_manager.hpp"
#include "amuquant/kv_surgeon.hpp"
#include "amuquant/turbovec_ffi.h"

#include <iostream>
#include <cassert>
#include <vector>

using namespace amuquant;

int main() {
    std::cout << "[TEST] Starting AMUQuantController & State Machine Validation..." << std::endl;

    AMUQuantConfig cfg;
    cfg.memory_limit_bytes = 16ULL * 1024ULL * 1024ULL * 1024ULL; // 16 GB
    cfg.os_reserved_bytes  = 2ULL * 1024ULL * 1024ULL * 1024ULL;  // 2 GB (Safe limit = 14 GB)
    cfg.alpha = 0.65;
    cfg.beta = 0.35;
    cfg.theta_low = 0.65;
    cfg.theta_high = 0.85;
    cfg.delta_hyst = 0.05;
    cfg.eval_interval_tokens = 1; // Evaluate every step for test precision

    TurbovecIndexHandle* handle = nullptr;
    turbovec_init(cfg.embedding_dim, cfg.initial_bit_width, &handle);

    auto telemetry = std::make_shared<SystemTelemetry>();
    telemetry->set_mock_mode(true);

    auto layer_mgr = std::make_shared<LayerManager>();
    auto kv_surgeon = std::make_shared<KVSurgeon>(handle, cfg.embedding_dim);

    AMUQuantController controller(cfg, telemetry, layer_mgr, kv_surgeon, handle);

    // Mock callback lambdas
    std::vector<float> dummy_vec(cfg.embedding_dim, 0.5f);
    auto extractor = [&](int32_t) -> const float* { return dummy_vec.data(); };
    auto seq_rm = [](int32_t, int32_t) {};
    auto seq_add = [](int32_t, int32_t, int32_t) {};

    // 1. Baseline state (RSS = 4 GB / 14 GB ~ 28% load) -> Expect REGIME_1
    MemoryStats s1{};
    s1.timestamp_sec = 10.0;
    s1.process_rss_bytes = 4ULL * 1024ULL * 1024ULL * 1024ULL;
    telemetry->set_mock_stats(s1);

    auto d1 = controller.on_token_step(0, 100, extractor, seq_rm, seq_add);
    assert(d1.regime == Regime::REGIME_1_HIGH_THROUGHPUT);
    assert(d1.prefetch_depth == 3);
    assert(!d1.aggressive_evict);
    std::cout << "  - Low Load (RSS 4GB) -> Regime 1 (High-Throughput): OK" << std::endl;

    // 2. Moderate pressure increase (RSS = 10 GB / 14 GB ~ 71.4% load) -> Transitions to REGIME_2
    MemoryStats s2{};
    s2.timestamp_sec = 40.0;
    s2.process_rss_bytes = 10ULL * 1024ULL * 1024ULL * 1024ULL;
    telemetry->set_mock_stats(s2);

    auto d2 = controller.on_token_step(1, 500, extractor, seq_rm, seq_add);
    assert(d2.regime == Regime::REGIME_2_BALANCED_OFFLOAD);
    assert(d2.prefetch_depth == 1);
    assert(d2.aggressive_evict);
    std::cout << "  - Moderate Load (RSS 10GB) -> Regime 2 (Balanced Offload): OK" << std::endl;

    // 3. Critical memory pressure (RSS = 13 GB / 14 GB ~ 92.8% load) -> Transitions to REGIME_3
    MemoryStats s3{};
    s3.timestamp_sec = 50.0;
    s3.process_rss_bytes = 13ULL * 1024ULL * 1024ULL * 1024ULL;
    telemetry->set_mock_stats(s3);

    auto d3 = controller.on_token_step(2, 600, extractor, seq_rm, seq_add);
    assert(d3.regime == Regime::REGIME_3_ULTRA_LOW_SURVIVAL);
    assert(d3.prefetch_depth == 0);
    assert(d3.aggressive_evict);
    assert(d3.requantized_to_2bit);
    std::cout << "  - Critical Load (RSS 13GB) -> Regime 3 (Survival Mode 2-bit): OK" << std::endl;

    // 4. Hysteresis stabilization: memory drops to 82% (below theta_high 85%, but above reset floor 80%)
    // State machine must remain in REGIME_3 to prevent thrashing
    MemoryStats s4{};
    s4.timestamp_sec = 60.0;
    s4.process_rss_bytes = static_cast<uint64_t>(14ULL * 1024ULL * 1024ULL * 1024ULL * 0.82);
    telemetry->set_mock_stats(s4);

    auto d4 = controller.on_token_step(3, 650, extractor, seq_rm, seq_add);
    assert(d4.regime == Regime::REGIME_3_ULTRA_LOW_SURVIVAL);
    std::cout << "  - Hysteresis Prevention (82% usage in survival) -> Retained Regime 3: OK" << std::endl;

    // 5. Memory relieved below (theta_high - delta_hyst) -> Demotes back to REGIME_2
    MemoryStats s5{};
    s5.timestamp_sec = 70.0;
    s5.process_rss_bytes = static_cast<uint64_t>(14ULL * 1024ULL * 1024ULL * 1024ULL * 0.72);
    telemetry->set_mock_stats(s5);

    auto d5 = controller.on_token_step(4, 700, extractor, seq_rm, seq_add);
    assert(d5.regime == Regime::REGIME_2_BALANCED_OFFLOAD);
    std::cout << "  - Memory Relieved below threshold -> Returned to Regime 2: OK" << std::endl;

    turbovec_free(handle);
    std::cout << "[PASS] AMUQuantController & State Machine Validation Succeeded.\n" << std::endl;
    return 0;
}
