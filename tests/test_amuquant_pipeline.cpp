#include "amuquant/engine.hpp"

#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>
#include <cstring>
#include <sys/mman.h>
#include <unistd.h>

using namespace amuquant;

/**
 * @brief Realistic simulated KV-Cache maintaining physical FP16 token slots.
 */
struct RealisticKVCache {
    uint32_t embedding_dim{1024};
    // Each token slot stores 1024 half-precision (FP16) floats = 2048 bytes
    std::vector<std::vector<uint16_t>> token_slots;
    std::vector<int32_t> slot_positions;
    size_t total_reclaimed_bytes{0};

    void append_token() {
        int32_t next_pos = slot_positions.empty() ? 0 : (slot_positions.back() + 1);
        token_slots.push_back(std::vector<uint16_t>(embedding_dim, 0x3C00)); // 1.0 in FP16
        slot_positions.push_back(next_pos);
    }

    void seq_rm(int32_t p0, int32_t p1) {
        std::vector<std::vector<uint16_t>> remaining_slots;
        std::vector<int32_t> remaining_pos;

        for (size_t i = 0; i < slot_positions.size(); ++i) {
            int32_t p = slot_positions[i];
            if (p >= p0 && p < p1) {
                total_reclaimed_bytes += (embedding_dim * sizeof(uint16_t));
            } else {
                remaining_slots.push_back(std::move(token_slots[i]));
                remaining_pos.push_back(p);
            }
        }
        token_slots = std::move(remaining_slots);
        slot_positions = std::move(remaining_pos);
    }

    void seq_add(int32_t p0, int32_t /*p1*/, int32_t delta) {
        for (size_t i = 0; i < slot_positions.size(); ++i) {
            if (slot_positions[i] >= p0) {
                slot_positions[i] += delta;
            }
        }
    }

    size_t active_slots() const {
        return token_slots.size();
    }

    size_t memory_bytes() const {
        return token_slots.size() * embedding_dim * sizeof(uint16_t);
    }
};

static void test_eviction_and_rope_realignment() {
    std::cout << "[TEST] Running Eviction & RoPE Realignment Verification..." << std::endl;
    KVSurgeon surgeon;
    TokenIndexMap& tmap = surgeon.get_token_map();
    tmap.init(30);

    int rm_calls = 0;
    int add_calls = 0;
    auto seq_rm = [&](int32_t, int32_t) -> bool { rm_calls++; return true; };
    auto seq_add = [&](int32_t, int32_t, int32_t) -> bool { add_calls++; return true; };

    // Invalid range test
    EvictionRange invalid_range{-1, -1};
    bool res = surgeon.execute_eviction(invalid_range, seq_rm, seq_add);
    assert(!res);
    assert(rm_calls == 0);
    assert(add_calls == 0);

    // Valid eviction range [4, 20) (16 middle tokens pruned, preserving sink [0..4) and recent window)
    EvictionRange valid_range{4, 20};
    bool success = surgeon.execute_eviction(valid_range, seq_rm, seq_add);
    assert(success);
    assert(rm_calls == 1);
    assert(add_calls == 1);
    assert(tmap.physical_size() == 14); // 30 - 16 = 14

    // Initial sink tokens preserved [0..3]
    for (int i = 0; i < 4; ++i) {
        assert(tmap.get_logical_id(i) == static_cast<uint64_t>(i));
    }
    // Recent tokens shifted: slot 4 corresponds to original token 20
    for (int i = 4; i < 14; ++i) {
        assert(tmap.get_logical_id(i) == static_cast<uint64_t>(i + 16));
    }
    std::cout << "  - Valid Eviction -> Attention Sink Protected & KV Pruned Cleanly: OK" << std::endl;
}

static void test_atomic_rollback_on_failure() {
    std::cout << "[TEST] Running Sequence Removal Failure Rollback Check..." << std::endl;
    KVSurgeon surgeon;
    TokenIndexMap& tmap = surgeon.get_token_map();
    tmap.init(40);

    auto failing_seq_rm = [&](int32_t, int32_t) -> bool { return false; };
    auto seq_add = [&](int32_t, int32_t, int32_t) -> bool { return true; };

    EvictionRange range{4, 20};
    bool success = surgeon.execute_eviction(range, failing_seq_rm, seq_add);
    assert(!success);
    assert(tmap.physical_size() == 40);
    for (int i = 0; i < 40; ++i) {
        assert(tmap.get_logical_id(i) == static_cast<uint64_t>(i));
    }
    std::cout << "  - seq_rm Failure -> Token Map Unaltered (Atomic Safety): OK" << std::endl;
}

static void test_repeated_eviction_invariants() {
    std::cout << "[TEST] Running Repeated Eviction Invariant Checks..." << std::endl;
    KVSurgeon surgeon;
    TokenIndexMap& tmap = surgeon.get_token_map();
    tmap.init(60); // 60 initial tokens

    auto seq_rm = [](int32_t, int32_t) -> bool { return true; };
    auto seq_add = [](int32_t, int32_t, int32_t) -> bool { return true; };

    // Round 1: Evict [4, 24) (20 tokens)
    bool ok1 = surgeon.execute_eviction({4, 24}, seq_rm, seq_add);
    assert(ok1);
    assert(tmap.physical_size() == 40);

    // Round 2: Evict [4, 14) (10 more tokens)
    bool ok2 = surgeon.execute_eviction({4, 14}, seq_rm, seq_add);
    assert(ok2);
    assert(tmap.physical_size() == 30);

    // Verify logical mapping consistency
    for (size_t p = 0; p < tmap.physical_size(); ++p) {
        uint64_t lid = tmap.get_logical_id(static_cast<int32_t>(p));
        assert(lid != UINT64_MAX);
        assert(tmap.get_physical_pos(lid) == static_cast<int32_t>(p));
    }
    std::cout << "  - Multi-Round Consecutive Evictions -> Token Mapping Monotonic: OK" << std::endl;
}

int main() {
    test_eviction_and_rope_realignment();
    test_atomic_rollback_on_failure();
    test_repeated_eviction_invariants();
    std::cout << "[TEST] Starting Realistic End-to-End AMU-QUANT Pipeline Verification..." << std::endl;

    AMUQuantConfig cfg;
    cfg.memory_limit_bytes = 14ULL * 1024ULL * 1024ULL * 1024ULL; // 14 GB Safe Working Limit
    cfg.os_reserved_bytes  = 2ULL * 1024ULL * 1024ULL * 1024ULL;  // 2 GB OS Stability Floor
    cfg.num_layers = 16;
    cfg.embedding_dim = 1024;
    cfg.eval_interval_tokens = 4;
    cfg.recent_window_tokens = 24;
    cfg.min_recent_window_tokens = 12;
    cfg.attention_sink_tokens = 4;

    AMUQuantEngine engine(cfg);
    Status st = engine.initialize();
    assert(st == Status::OK);
    std::cout << "  - Engine initialized: OK" << std::endl;

    // 1. Allocate REAL page-aligned mmap buffers for 16 transformer layers
    constexpr size_t REAL_LAYER_BYTES = 1024 * 1024; // 1 MB physical page allocation per layer
    std::vector<void*> layer_mmap_ptrs(cfg.num_layers, nullptr);

    for (uint32_t l = 0; l < cfg.num_layers; ++l) {
        layer_mmap_ptrs[l] = mmap(nullptr, REAL_LAYER_BYTES, PROT_READ | PROT_WRITE, MAP_ANON | MAP_PRIVATE, -1, 0);
        assert(layer_mmap_ptrs[l] != MAP_FAILED);
        // Touch memory pages to guarantee physical resident page allocation in OS kernel
        std::memset(layer_mmap_ptrs[l], 0xAB, REAL_LAYER_BYTES);
        engine.register_layer(l, layer_mmap_ptrs[l], REAL_LAYER_BYTES);
    }
    std::cout << "  - Registered 16 physical mmap transformer layers (16 MB real resident RAM): OK" << std::endl;

    // Verify mincore(2) kernel page table telemetry
    uint64_t actual_layer_0 = engine.get_layer_manager()->get_actual_resident_bytes(0);
    uint64_t total_actual = engine.get_layer_manager()->get_total_actual_resident_bytes();
    uint64_t total_registered = engine.get_layer_manager()->get_total_registered_bytes();
    uint64_t actual_evicted = engine.get_layer_manager()->get_actual_evicted_bytes();
    assert(actual_layer_0 > 0);
    assert(total_actual >= actual_layer_0);
    assert(total_registered == 16 * REAL_LAYER_BYTES);
    assert(actual_evicted == total_registered - total_actual);

    // Verify kernel synchronization (sync_scheduled_with_actual_residency)
    auto actual_layers = engine.get_layer_manager()->get_actual_resident_layers();
    assert(actual_layers.size() == 16);
    size_t synced_count = engine.get_layer_manager()->sync_scheduled_with_actual_residency();
    assert(synced_count == 16);
    assert(engine.get_layer_manager()->get_scheduled_resident_bytes() == 16 * REAL_LAYER_BYTES);
    std::cout << "  - Kernel Residency Synchronization: Verified 16 resident layers via mincore(2): OK" << std::endl;
    std::cout << "  - mincore(2) Kernel Page Table Verification: Layer 0 resident = " 
              << actual_layer_0 << " bytes, Total Actual = " << total_actual 
              << " bytes, Registered = " << total_registered 
              << " bytes, Evicted = " << actual_evicted << " bytes: OK" << std::endl;

    // 2. Prepare Realistic KV-Cache
    constexpr uint32_t TOTAL_STEPS = 90;
    RealisticKVCache kv_cache;
    kv_cache.embedding_dim = cfg.embedding_dim;

    auto seq_rm = [&](int32_t p0, int32_t p1) -> bool {
        kv_cache.seq_rm(p0, p1);
        return true;
    };

    auto seq_add = [&](int32_t p0, int32_t p1, int32_t delta) -> bool {
        kv_cache.seq_add(p0, p1, delta);
        return true;
    };

    auto kv_surgeon = engine.get_kv_surgeon();
    
    // Record canonical conversation turns in history store
    engine.record_turn("user", "Nama proyeknya Aegis, nomor versinya 90412.", 0, 15);
    engine.record_turn("assistant", "Baik, konfigurasi proyek Aegis telah dicatat.", 15, 25);

    // 3. Autoregressive Simulation with Dynamic Telemetry and Layer madvise Paging
    auto telemetry = engine.get_telemetry();
    telemetry->set_mock_mode(true);

    for (uint32_t t = 0; t < TOTAL_STEPS; ++t) {
        kv_cache.append_token();
        engine.append_token(t);

        // Layer forward pass with POSIX_MADV_WILLNEED and DONTNEED execution
        for (uint32_t l = 0; l < cfg.num_layers; ++l) {
            engine.begin_layer_forward(l);
            engine.end_layer_forward(l);
        }

        // Simulate memory ramp-up:
        // t in [0, 40): Low load -> Regime 1
        // t in [40, 60): Moderate load (RSS = 10 GB) -> Regime 2
        // t in [60, 90): Critical external host starvation (Available RAM depleted to 1.0 GB < 2.0 GB) -> Regime 3
        MemoryStats s{};
        s.timestamp_sec = 20.0 + (t * 0.5);
        s.total_ram_bytes = 16ULL * 1024ULL * 1024ULL * 1024ULL;

        if (t < 40) {
            s.available_ram_bytes = 10ULL * 1024ULL * 1024ULL * 1024ULL;
            s.process_rss_bytes   = 4ULL * 1024ULL * 1024ULL * 1024ULL;
            s.swap_used_bytes     = 0;
        } else if (t < 60) {
            s.available_ram_bytes = 5ULL * 1024ULL * 1024ULL * 1024ULL;
            // Smoothly ramp RSS from 5 GB up to 8.5 GB to avoid huge dM/dt spike
            s.process_rss_bytes   = static_cast<uint64_t>((5.0 + (t - 40) * 0.175) * 1024ULL * 1024ULL * 1024ULL);
            s.swap_used_bytes     = 0;
        } else {
            // External application consumes host RAM! Process RSS is 6 GB, but available RAM is critical!
            s.available_ram_bytes = 1ULL * 1024ULL * 1024ULL * 1024ULL;  // Breaches 2 GB reserve!
            s.process_rss_bytes   = 6ULL * 1024ULL * 1024ULL * 1024ULL;
            s.swap_used_bytes     = 800ULL * 1024ULL * 1024ULL;          // 800 MB swap thrashing
        }
        telemetry->set_mock_stats(s);

        StepDecision dec = engine.step_token(t, static_cast<uint32_t>(kv_cache.active_slots()), seq_rm, seq_add);

        // Verify TokenIndexMap synchronization with physical KV-cache
        assert(kv_surgeon->get_token_map().physical_size() == kv_cache.active_slots());

        if (t == 56) {
            assert(dec.regime == Regime::REGIME_2_BALANCED_OFFLOAD);
            assert(dec.aggressive_evict == true);
        } else if (t == 76) {
            assert(dec.regime == Regime::REGIME_3_ULTRA_LOW_SURVIVAL);
            assert(dec.prefetch_depth == 0);
        }
    }

    std::cout << "  - Dynamic Regime Transitions (Regime 2 Offload & Regime 3 External Starvation): OK" << std::endl;
    std::cout << "  - Physical KV-Cache Reclaimed: " << (kv_cache.total_reclaimed_bytes / 1024) << " KB (OK)" << std::endl;
    assert(kv_cache.total_reclaimed_bytes > 0);

    // 4. Verify Canonical History Store Integrity
    auto history = engine.get_history_store();
    assert(history != nullptr);
    assert(history->size() >= 2);
    const HistoryTurn* turn0 = history->get_turn(0);
    assert(turn0 != nullptr);
    assert(turn0->is_evicted_from_kv == true);
    std::cout << "  - Canonical History Store: Verbatim Turns Retained & Evicted Boundaries Flagged: OK" << std::endl;

    // 5. Clean up physical mmap layers
    for (uint32_t l = 0; l < cfg.num_layers; ++l) {
        if (layer_mmap_ptrs[l] != nullptr && layer_mmap_ptrs[l] != MAP_FAILED) {
            munmap(layer_mmap_ptrs[l], REAL_LAYER_BYTES);
        }
    }
    std::cout << "  - Cleaned up physical mmap memory pages: OK" << std::endl;

    std::cout << "[PASS] Realistic End-to-End Pipeline Verification Succeeded.\n" << std::endl;
    return 0;
}
