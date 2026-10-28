#include "amuquant/controller.hpp"
#include "amuquant/telemetry.hpp"
#include "amuquant/layer_manager.hpp"
#include "amuquant/kv_surgeon.hpp"

#include <iostream>
#include <cassert>
#include <vector>

using namespace amuquant;

int main() {
    std::cout << "[TEST] Starting Deterministic FSM State Transition & Hysteresis Validation (Mock Telemetry)..." << std::endl;

    AMUQuantConfig cfg;
    cfg.memory_limit_bytes = 16ULL * 1024ULL * 1024ULL * 1024ULL; // 16 GB
    cfg.os_reserved_bytes  = 2ULL * 1024ULL * 1024ULL * 1024ULL;  // 2 GB (Safe limit = 14 GB)
    cfg.alpha = 0.65;
    cfg.beta = 0.35;
    cfg.theta_low = 0.65;
    cfg.theta_high = 0.85;
    cfg.delta_hyst = 0.05;
    cfg.eval_interval_tokens = 1; // Evaluate every step for test precision
    cfg.policy = ArbitrationPolicy::THRESHOLD_HYSTERESIS;

    {
        auto telemetry = std::make_shared<SystemTelemetry>();
        telemetry->set_mock_mode(true);

        auto layer_mgr = std::make_shared<LayerManager>();
        auto kv_surgeon = std::make_shared<KVSurgeon>();

        AMUQuantController controller(cfg, telemetry, layer_mgr, kv_surgeon);

        // Mock callback lambdas
        auto seq_rm = [](int32_t, int32_t) -> bool { return true; };
        auto seq_add = [](int32_t, int32_t, int32_t) -> bool { return true; };

        // 1. Baseline state (RSS = 4 GB / 14 GB ~ 28% load) -> Expect REGIME_1
        MemoryStats s1{};
        s1.timestamp_sec = 10.0;
        s1.process_rss_bytes = 4ULL * 1024ULL * 1024ULL * 1024ULL;
        telemetry->set_mock_stats(s1);

        auto d1 = controller.on_token_step(0, 100, seq_rm, seq_add);
        assert(d1.regime == Regime::REGIME_1_HIGH_THROUGHPUT);
        assert(d1.prefetch_depth == 3);
        assert(!d1.aggressive_evict);
        std::cout << "  - Low Load (RSS 4GB) -> Regime 1 (High-Throughput): OK" << std::endl;

        // 2. Moderate pressure increase (RSS = 10 GB / 14 GB ~ 71.4% load) -> Transitions to REGIME_2
        MemoryStats s2{};
        s2.timestamp_sec = 40.0;
        s2.process_rss_bytes = 10ULL * 1024ULL * 1024ULL * 1024ULL;
        telemetry->set_mock_stats(s2);

        auto d2 = controller.on_token_step(1, 500, seq_rm, seq_add);
        assert(d2.regime == Regime::REGIME_2_BALANCED_OFFLOAD);
        assert(d2.prefetch_depth == 1);
        assert(d2.aggressive_evict);
        std::cout << "  - Moderate Load (RSS 10GB) -> Regime 2 (Balanced Offload): OK" << std::endl;

        // 3. Critical memory pressure (RSS = 13 GB / 14 GB ~ 92.8% load) -> Transitions to REGIME_3
        MemoryStats s3{};
        s3.timestamp_sec = 50.0;
        s3.process_rss_bytes = 13ULL * 1024ULL * 1024ULL * 1024ULL;
        telemetry->set_mock_stats(s3);

        auto d3 = controller.on_token_step(2, 600, seq_rm, seq_add);
        assert(d3.regime == Regime::REGIME_3_ULTRA_LOW_SURVIVAL);
        assert(d3.prefetch_depth == 0);
        assert(d3.aggressive_evict);
        std::cout << "  - Critical Load (RSS 13GB) -> Regime 3 (Survival Mode): OK" << std::endl;

        // 4. Hysteresis stabilization: memory drops to 82% (below theta_high 85%, but above reset floor 80%)
        MemoryStats s4{};
        s4.timestamp_sec = 60.0;
        s4.process_rss_bytes = static_cast<uint64_t>(14ULL * 1024ULL * 1024ULL * 1024ULL * 0.82);
        telemetry->set_mock_stats(s4);

        auto d4 = controller.on_token_step(3, 650, seq_rm, seq_add);
        assert(d4.regime == Regime::REGIME_3_ULTRA_LOW_SURVIVAL);
        std::cout << "  - Hysteresis Prevention (82% usage in survival) -> Retained Regime 3: OK" << std::endl;

        // 5. Memory relieved below (theta_high - delta_hyst) -> Demotes back to REGIME_2
        MemoryStats s5{};
        s5.timestamp_sec = 70.0;
        s5.process_rss_bytes = static_cast<uint64_t>(14ULL * 1024ULL * 1024ULL * 1024ULL * 0.72);
        telemetry->set_mock_stats(s5);

        auto d5 = controller.on_token_step(4, 700, seq_rm, seq_add);
        assert(d5.regime == Regime::REGIME_2_BALANCED_OFFLOAD);
        std::cout << "  - Memory Relieved below threshold -> Returned to Regime 2: OK" << std::endl;

        // 6. External Host Memory Starvation (Process RSS is low, but available host RAM depleted by other apps)
        controller.reset();
        MemoryStats s6{};
        s6.timestamp_sec = 80.0;
        s6.total_ram_bytes = 16ULL * 1024ULL * 1024ULL * 1024ULL;
        s6.available_ram_bytes = 1ULL * 1024ULL * 1024ULL * 1024ULL; // Breaches 2 GB os_reserved_bytes!
        s6.process_rss_bytes = 3ULL * 1024ULL * 1024ULL * 1024ULL;   // Process itself uses only 3 GB
        s6.swap_used_bytes = 1ULL * 1024ULL * 1024ULL * 1024ULL;     // 1 GB swap thrashing
        telemetry->set_mock_stats(s6);

        auto d6 = controller.on_token_step(5, 750, seq_rm, seq_add);
        assert(d6.regime == Regime::REGIME_3_ULTRA_LOW_SURVIVAL);
        std::cout << "  - External Host Starvation (Low Process RSS, Exhausted System RAM) -> Triggered Regime 3: OK" << std::endl;
    }

    std::cout << "[PASS] Threshold FSM Validation Succeeded.\n" << std::endl;

    // Test 2: Predictive Cost-Benefit Multi-Action Policy
    {
        std::cout << "[TEST] Starting Predictive Cost-Benefit Multi-Action Policy Validation..." << std::endl;

        AMUQuantConfig cfg_opt;
        cfg_opt.memory_limit_bytes = 16ULL * 1024ULL * 1024ULL * 1024ULL; // 16 GB
        cfg_opt.os_reserved_bytes  = 2ULL * 1024ULL * 1024ULL * 1024ULL;  // 2 GB (Safe limit = 14 GB)
        cfg_opt.policy = ArbitrationPolicy::PREDICTIVE_COST_BENEFIT;
        cfg_opt.weight_latency_cost = 0.25;
        cfg_opt.weight_quality_loss = 0.35;
        cfg_opt.risk_oom_penalty = 10.0;
        cfg_opt.tau_horizon_sec = 2.0;
        cfg_opt.eval_interval_tokens = 1;

        auto telemetry = std::make_shared<SystemTelemetry>();
        telemetry->set_mock_mode(true);
        auto layer_mgr = std::make_shared<LayerManager>();
        auto kv_surgeon = std::make_shared<KVSurgeon>();

        AMUQuantController controller(cfg_opt, telemetry, layer_mgr, kv_surgeon);

        auto seq_rm = [](int32_t, int32_t) -> bool { return true; };
        auto seq_add = [](int32_t, int32_t, int32_t) -> bool { return true; };

        // 1. Low load: RSS = 4GB, Available = 10GB -> Controller selects REGIME_1
        MemoryStats s1{};
        s1.timestamp_sec = 10.0;
        s1.total_ram_bytes = 16ULL * 1024ULL * 1024ULL * 1024ULL;
        s1.process_rss_bytes = 4ULL * 1024ULL * 1024ULL * 1024ULL;
        s1.available_ram_bytes = 10ULL * 1024ULL * 1024ULL * 1024ULL;
        telemetry->set_mock_stats(s1);

        auto d1 = controller.on_token_step(0, 100, seq_rm, seq_add);
        assert(d1.regime == Regime::REGIME_1_HIGH_THROUGHPUT);
        assert(d1.policy_used == ArbitrationPolicy::PREDICTIVE_COST_BENEFIT);
        std::cout << "  - Low Load -> Policy chose Regime 1 (Cost: " << d1.selected_utility_cost << "): OK" << std::endl;

        // 2. High memory growth velocity: RSS jumps by 8 GB in 1 sec -> Projected deficit triggers REGIME_3
        MemoryStats s2{};
        s2.timestamp_sec = 11.0;
        s2.total_ram_bytes = 16ULL * 1024ULL * 1024ULL * 1024ULL;
        s2.process_rss_bytes = 13ULL * 1024ULL * 1024ULL * 1024ULL;
        s2.available_ram_bytes = 1ULL * 1024ULL * 1024ULL * 1024ULL;
        telemetry->set_mock_stats(s2);

        auto d2 = controller.on_token_step(1, 400, seq_rm, seq_add);
        assert(d2.regime == Regime::REGIME_3_ULTRA_LOW_SURVIVAL);
        std::cout << "  - High Growth Velocity Spike -> Cost-Benefit selected Regime 3 Survival (Cost: " 
                  << d2.selected_utility_cost << "): OK" << std::endl;
    }

    // 3. Test Dynamic KV Sizing & Online Adaptive Calibration Telemetry
    {
        std::cout << "[TEST] Starting Dynamic KV Sizing & Online Adaptive Calibration Validation..." << std::endl;
        AMUQuantConfig cfg;
        cfg.num_layers = 36;
        cfg.kv_heads = 2;
        cfg.head_dim = 128;
        cfg.embedding_dim = 2048;
        cfg.memory_limit_bytes = 14ULL * 1024ULL * 1024ULL * 1024ULL;
        cfg.os_reserved_bytes  = 2ULL * 1024ULL * 1024ULL * 1024ULL;
        cfg.policy = ArbitrationPolicy::THRESHOLD_HYSTERESIS;
        cfg.theta_low = 0.50;
        cfg.eval_interval_tokens = 1;
        cfg.recent_window_tokens = 32;
        cfg.min_eviction_batch = 16;

        auto telemetry = std::make_shared<SystemTelemetry>();
        telemetry->set_mock_mode(true);
        auto layer_mgr = std::make_shared<LayerManager>();
        auto kv_surgeon = std::make_shared<KVSurgeon>();
        AMUQuantController controller(cfg, telemetry, layer_mgr, kv_surgeon);

        assert(controller.get_adaptive_calibration_factor() == 1.0);
        assert(controller.get_last_estimated_savings() == 0);

        auto seq_rm = [](int32_t, int32_t) -> bool { return true; };
        auto seq_add = [](int32_t, int32_t, int32_t) -> bool { return true; };

        // Trigger transition to Regime 2 (Load = 60% > theta_low)
        MemoryStats s{};
        s.timestamp_sec = 20.0;
        s.process_rss_bytes = 8ULL * 1024ULL * 1024ULL * 1024ULL;
        telemetry->set_mock_stats(s);

        // Sequence length 100 with window 32 -> evicts middle tokens [4..68) = 64 tokens
        StepDecision dec = controller.on_token_step(0, 100, seq_rm, seq_add);
        assert(dec.regime == Regime::REGIME_2_BALANCED_OFFLOAD);
        assert(dec.kv_pruned);
        assert(dec.estimated_savings_bytes > 0);
        assert(dec.actual_reclaimed_bytes > 0);
        assert(controller.get_adaptive_calibration_factor() > 0.0);
        std::cout << "  - Dynamic Sizing & Calibration Telemetry: Estimated = " 
                  << dec.estimated_savings_bytes << " bytes, Actual = " 
                  << dec.actual_reclaimed_bytes << " bytes, Delta = "
                  << dec.estimation_delta_bytes << " bytes, Factor = "
                  << dec.calibration_factor << ": OK" << std::endl;

        // Test manual calibration adjustment
        uint64_t fake_actual = dec.estimated_savings_bytes / 2;
        controller.record_actual_reclaimed(fake_actual);
        assert(controller.get_adaptive_calibration_factor() < 1.0);
        assert(controller.get_last_actual_reclaimed() == fake_actual);
        std::cout << "  - Adaptive Calibration Factor Auto-Adjustment: " 
                  << controller.get_adaptive_calibration_factor() << ": OK" << std::endl;
    }

    // 4. Test Proactive Working-Set Budgeting & Surgery Latency Profiling
    {
        std::cout << "[TEST] Starting Proactive Working-Set Budgeting & Micro-Latency Profiling..." << std::endl;
        AMUQuantConfig cfg;
        cfg.policy = ArbitrationPolicy::PREDICTIVE_COST_BENEFIT;
        cfg.target_working_set_tokens = 64; // Strict O(1) bound
        cfg.attention_sink_tokens = 16;
        cfg.recent_window_tokens = 48;
        cfg.min_eviction_batch = 8;
        cfg.eval_interval_tokens = 1;

        auto telemetry = std::make_shared<SystemTelemetry>();
        telemetry->set_mock_mode(true);
        auto layer_mgr = std::make_shared<LayerManager>();
        auto kv_surgeon = std::make_shared<KVSurgeon>();

        AMUQuantController controller(cfg, telemetry, layer_mgr, kv_surgeon);

        auto seq_rm = [](int32_t, int32_t) -> bool { return true; };
        auto seq_add = [](int32_t, int32_t, int32_t) -> bool { return true; };

        // Even with huge available RAM (12 GB available out of 16 GB),
        // exceeding target_working_set_tokens triggers proactive Regime 2 pruning
        MemoryStats s{};
        s.timestamp_sec = 100.0;
        s.total_ram_bytes = 16ULL * 1024ULL * 1024ULL * 1024ULL;
        s.process_rss_bytes = 2ULL * 1024ULL * 1024ULL * 1024ULL;
        s.available_ram_bytes = 12ULL * 1024ULL * 1024ULL * 1024ULL;
        telemetry->set_mock_stats(s);

        StepDecision dec = controller.on_token_step(0, 150, seq_rm, seq_add);
        assert(dec.regime == Regime::REGIME_2_BALANCED_OFFLOAD);
        assert(dec.kv_pruned);

        auto surgeon_stats = kv_surgeon->get_stats(36, 8, 128);
        assert(surgeon_stats.total_surgery_count > 0);
        assert(surgeon_stats.last_surgery_latency_us >= 0.0);
        std::cout << "  - Proactive Working-Set Budgeting: Target = 64 slots, Sequence = 150 -> Pruned to Regime 2: OK" << std::endl;
        std::cout << "  - Micro-Latency Profiling: Surgery Latency = " << surgeon_stats.last_surgery_latency_us << " us: OK" << std::endl;
    }

    std::cout << "[PASS] All AMUQuantController Policy Validations Succeeded.\n" << std::endl;
    return 0;
}
