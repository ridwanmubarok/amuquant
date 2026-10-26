#include "amuquant/engine.hpp"

#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>
#include <random>
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

static void test_transactional_eviction_rollback() {
    std::cout << "[TEST] Running Transactional Eviction & Rollback Verification..." << std::endl;
    TurbovecIndexHandle* handle = nullptr;
    TurbovecStatus st = turbovec_init(128, 4, &handle);
    assert(st == TURBOVEC_STATUS_OK && handle != nullptr);

    {
        KVSurgeon surgeon(handle, 128, true); // Async archiver enabled
        TokenIndexMap& tmap = surgeon.get_token_map();
        tmap.init(30);

        std::vector<float> sample_vec(128, 0.42f);
        int rm_calls = 0;
        int add_calls = 0;
        auto seq_rm = [&](int32_t, int32_t) { rm_calls++; };
        auto seq_add = [&](int32_t, int32_t, int32_t) { add_calls++; };

        // Test 1: Extractor fails on token pos 10
        auto failing_extractor = [&](int32_t pos) -> const float* {
            if (pos == 10) return nullptr;
            return sample_vec.data();
        };

        EvictionRange range{4, 20}; // 16 middle tokens
        bool success = surgeon.execute_eviction(range, failing_extractor, seq_rm, seq_add);
        assert(!success);
        assert(rm_calls == 0); // seq_rm was NEVER called
        assert(add_calls == 0); // seq_add was NEVER called
        assert(tmap.physical_size() == 30); // token map unchanged
        for (int i = 0; i < 30; ++i) {
            assert(tmap.get_logical_id(i) == static_cast<uint64_t>(i));
        }
        std::cout << "  - Phase 1 Extractor Failure -> Eviction Aborted & Zero KV Pruning: OK" << std::endl;

        // Test 2: Valid extractor succeeds, commits transactionally
        auto valid_extractor = [&](int32_t) -> const float* {
            return sample_vec.data();
        };
        success = surgeon.execute_eviction(range, valid_extractor, seq_rm, seq_add);
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
        std::cout << "  - Phase 2 Valid Commit -> Batch Verified & KV Pruned Cleanly: OK" << std::endl;
    }

    turbovec_free(handle);
}

static void test_repeated_eviction_invariants() {
    std::cout << "[TEST] Running Repeated Eviction Invariant Checks..." << std::endl;
    TurbovecIndexHandle* handle = nullptr;
    TurbovecStatus st = turbovec_init(64, 4, &handle);
    assert(st == TURBOVEC_STATUS_OK && handle != nullptr);

    {
        KVSurgeon surgeon(handle, 64, true);
        TokenIndexMap& tmap = surgeon.get_token_map();
        tmap.init(60); // 60 initial prompt tokens

        std::vector<float> dummy(64, 0.1f);
        auto extractor = [&](int32_t) -> const float* { return dummy.data(); };
        auto seq_rm = [](int32_t, int32_t) {};
        auto seq_add = [](int32_t, int32_t, int32_t) {};

        // Round 1: Evict [4, 24) (20 tokens)
        bool ok1 = surgeon.execute_eviction({4, 24}, extractor, seq_rm, seq_add);
        assert(ok1);
        assert(tmap.physical_size() == 40);

        // Round 2: Evict [4, 14) (10 more tokens)
        bool ok2 = surgeon.execute_eviction({4, 14}, extractor, seq_rm, seq_add);
        assert(ok2);
        assert(tmap.physical_size() == 30);

        // Verify logical mapping consistency
        for (size_t p = 0; p < tmap.physical_size(); ++p) {
            uint64_t lid = tmap.get_logical_id(static_cast<int32_t>(p));
            assert(lid != UINT64_MAX);
            assert(tmap.get_physical_pos(lid) == static_cast<int32_t>(p));
        }
        std::cout << "  - Multi-Round Consecutive Evictions -> Token Mapping Monotonic & Invertible: OK" << std::endl;
    }

    turbovec_free(handle);
}

int main() {
    test_transactional_eviction_rollback();
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
    assert(actual_layer_0 > 0);
    assert(total_actual >= actual_layer_0);
    std::cout << "  - mincore(2) Kernel Page Table Verification: Layer 0 resident = " 
              << actual_layer_0 << " bytes, Total = " << total_actual << " bytes: OK" << std::endl;

    // 2. Prepare Realistic KV-Cache and Context Embedding Store
    constexpr uint32_t TOTAL_STEPS = 90;
    RealisticKVCache kv_cache;
    kv_cache.embedding_dim = cfg.embedding_dim;

    std::vector<std::vector<float>> context_tokens(TOTAL_STEPS, std::vector<float>(cfg.embedding_dim, 0.0f));

    // Plant synthetic "Needle" fact vector at step 10
    constexpr uint32_t NEEDLE_POS = 10;
    std::mt19937 needle_rng(9999);
    std::normal_distribution<float> needle_dist(0.0f, 1.0f);
    for (uint32_t d = 0; d < cfg.embedding_dim; ++d) {
        context_tokens[NEEDLE_POS][d] = needle_dist(needle_rng);
    }

    auto valid_extractor = [&](int32_t pos) -> const float* {
        if (pos >= 0 && pos < static_cast<int32_t>(context_tokens.size())) {
            return context_tokens[pos].data();
        }
        return nullptr;
    };

    auto seq_rm = [&](int32_t p0, int32_t p1) {
        kv_cache.seq_rm(p0, p1);
    };

    auto seq_add = [&](int32_t p0, int32_t p1, int32_t delta) {
        kv_cache.seq_add(p0, p1, delta);
    };

    // 3. Test Atomic Archiving Guarantee: Verify that null extractor aborts eviction without pruning KV cache
    auto null_extractor = [](int32_t) -> const float* { return nullptr; };
    auto kv_surgeon = engine.get_kv_surgeon();
    
    // Simulate initial sequence growth up to 40 tokens
    for (uint32_t t = 0; t < 40; ++t) {
        kv_cache.append_token();
        engine.append_token(t);
    }
    size_t pre_abort_slots = kv_cache.active_slots();
    EvictionRange test_range = kv_surgeon->plan_eviction(40, cfg.attention_sink_tokens, cfg.recent_window_tokens);
    assert(test_range.valid());

    bool eviction_success = kv_surgeon->execute_eviction(test_range, null_extractor, seq_rm, seq_add);
    assert(!eviction_success); // Must fail atomically!
    assert(kv_cache.active_slots() == pre_abort_slots); // Zero slots pruned
    assert(kv_surgeon->get_token_map().physical_size() == pre_abort_slots); // TokenIndexMap intact
    std::cout << "  - Atomic Archiving Safety (Null Extractor Aborted Without Pruning): OK" << std::endl;

    // 4. Autoregressive Simulation with Dynamic Telemetry and Layer madvise Paging
    auto telemetry = engine.get_telemetry();
    telemetry->set_mock_mode(true);

    for (uint32_t t = 40; t < TOTAL_STEPS; ++t) {
        kv_cache.append_token();
        engine.append_token(t);

        // Populate random context representation
        if (t != NEEDLE_POS) {
            std::mt19937 bg_rng(200 + t);
            std::normal_distribution<float> bg_dist(0.0f, 1.0f);
            for (uint32_t d = 0; d < cfg.embedding_dim; ++d) {
                context_tokens[t][d] = bg_dist(bg_rng);
            }
        }

        // Layer forward pass with POSIX_MADV_WILLNEED and DONTNEED execution
        for (uint32_t l = 0; l < cfg.num_layers; ++l) {
            engine.begin_layer_forward(l);
            engine.end_layer_forward(l);
        }

        // Simulate memory ramp-up:
        // t in [40, 60): Moderate load (RSS = 10 GB) -> Regime 2
        // t in [60, 90): Critical external host starvation (Available RAM depleted to 1.0 GB < 2.0 GB) -> Regime 3
        MemoryStats s{};
        s.timestamp_sec = 20.0 + (t * 0.5);
        s.total_ram_bytes = 16ULL * 1024ULL * 1024ULL * 1024ULL;

        if (t < 60) {
            s.available_ram_bytes = 5ULL * 1024ULL * 1024ULL * 1024ULL;
            s.process_rss_bytes   = 10ULL * 1024ULL * 1024ULL * 1024ULL; // ~71% -> Regime 2
            s.swap_used_bytes     = 0;
        } else {
            // External application consumes host RAM! Process RSS is only 6 GB, but available RAM is critical!
            s.available_ram_bytes = 1ULL * 1024ULL * 1024ULL * 1024ULL;  // Breaches 2 GB reserve!
            s.process_rss_bytes   = 6ULL * 1024ULL * 1024ULL * 1024ULL;
            s.swap_used_bytes     = 800ULL * 1024ULL * 1024ULL;          // 800 MB swap thrashing
        }
        telemetry->set_mock_stats(s);

        StepDecision dec = engine.step_token(t, static_cast<uint32_t>(kv_cache.active_slots()), valid_extractor, seq_rm, seq_add);

        // Verify TokenIndexMap synchronization with physical KV-cache
        assert(kv_surgeon->get_token_map().physical_size() == kv_cache.active_slots());

        if (t == 50) {
            assert(dec.regime == Regime::REGIME_2_BALANCED_OFFLOAD);
            assert(dec.aggressive_evict == true);
        } else if (t == 75) {
            assert(dec.regime == Regime::REGIME_3_ULTRA_LOW_SURVIVAL);
            assert(dec.prefetch_depth == 0);
        }
    }

    std::cout << "  - Dynamic Regime Transitions (Regime 2 Offload & Regime 3 External Starvation): OK" << std::endl;
    std::cout << "  - Physical KV-Cache Reclaimed: " << (kv_cache.total_reclaimed_bytes / 1024) << " KB (OK)" << std::endl;
    assert(kv_cache.total_reclaimed_bytes > 0);

    // 5. Test Needle-in-a-Haystack Recall from turbovec compressed memory
    auto recall_hits = engine.recall_context(context_tokens[NEEDLE_POS].data(), 3);
    assert(!recall_hits.empty());
    assert(recall_hits[0].id == NEEDLE_POS);
    assert(recall_hits[0].score > 0.80f);
    std::cout << "  - Needle-in-a-Haystack Retrieval: Chunk ID " << recall_hits[0].id 
              << " recovered with Cosine Score " << recall_hits[0].score << " (Recall@1 = 100%): OK" << std::endl;

    // 6. Clean up physical mmap layers
    for (uint32_t l = 0; l < cfg.num_layers; ++l) {
        if (layer_mmap_ptrs[l] != nullptr && layer_mmap_ptrs[l] != MAP_FAILED) {
            munmap(layer_mmap_ptrs[l], REAL_LAYER_BYTES);
        }
    }
    std::cout << "  - Cleaned up physical mmap memory pages: OK" << std::endl;

    std::cout << "[PASS] Realistic End-to-End Pipeline Verification Succeeded.\n" << std::endl;
    return 0;
}
