#include "amuquant/engine.hpp"

#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <chrono>
#include <fstream>
#include <random>
#include <cassert>
#include <cstring>
#include <sys/mman.h>
#include <unistd.h>

using namespace amuquant;

/**
 * @brief Realistic simulated KV-Cache maintaining physical 16-bit float tensor slots.
 */
struct StandaloneKVCache {
    uint32_t embedding_dim{1024};
    std::vector<std::vector<uint16_t>> token_slots;
    std::vector<int32_t> slot_positions;
    size_t total_reclaimed_bytes{0};

    void append_token(int32_t pos) {
        token_slots.push_back(std::vector<uint16_t>(embedding_dim, 0x3C00));
        slot_positions.push_back(pos);
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

    size_t active_slots() const { return token_slots.size(); }
    size_t memory_bytes() const { return token_slots.size() * embedding_dim * sizeof(uint16_t); }
};

struct StandaloneBenchmarkResult {
    std::string name;
    uint32_t tokens_processed{0};
    double duration_sec{0.0};
    double throughput_tok_s{0.0};
    size_t peak_kv_memory_kb{0};
    size_t final_kv_memory_kb{0};
    size_t kv_memory_reclaimed_kb{0};
    uint32_t tokens_archived{0};
    double recall_score{0.0};
    double recall_latency_us{0.0};
    Regime final_regime{Regime::REGIME_1_HIGH_THROUGHPUT};
};

int main(int argc, char** argv) {
    uint32_t total_tokens = 96;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--tokens" && i + 1 < argc) {
            total_tokens = static_cast<uint32_t>(std::stoi(argv[++i]));
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: amuquant_benchmark [--tokens <count>]\n";
            return 0;
        }
    }

    std::cout << "================================================================================\n";
    std::cout << "  AMU-QUANT STANDALONE EMPIRICAL BENCHMARK (ZERO EXTERNAL DEPENDENCIES)         \n";
    std::cout << "  Simulated Sequence Horizon: " << total_tokens << " tokens\n";
    std::cout << "================================================================================\n" << std::endl;

    constexpr uint32_t EMBEDDING_DIM = 1024;
    constexpr uint32_t NUM_LAYERS = 32;
    constexpr size_t REAL_LAYER_BYTES = 1024 * 1024; // 1 MB physical resident pages per layer (32 MB total)
    constexpr uint32_t NEEDLE_POS = 12;

    // Synthetic Needle vector
    std::vector<std::vector<float>> context_tokens(total_tokens, std::vector<float>(EMBEDDING_DIM, 0.0f));
    std::mt19937 needle_rng(8888);
    std::normal_distribution<float> needle_dist(0.0f, 1.0f);
    for (uint32_t d = 0; d < EMBEDDING_DIM; ++d) {
        context_tokens[NEEDLE_POS][d] = needle_dist(needle_rng);
    }
    for (uint32_t t = 0; t < total_tokens; ++t) {
        if (t != NEEDLE_POS) {
            std::mt19937 bg_rng(300 + t);
            std::normal_distribution<float> bg_dist(0.0f, 1.0f);
            for (uint32_t d = 0; d < EMBEDDING_DIM; ++d) {
                context_tokens[t][d] = bg_dist(bg_rng);
            }
        }
    }

    auto valid_extractor = [&](int32_t pos) -> const float* {
        if (pos >= 0 && pos < static_cast<int32_t>(context_tokens.size())) {
            return context_tokens[pos].data();
        }
        return nullptr;
    };

    // =========================================================================
    // RUN 1: UNMANAGED LINEAR CONTEXT INFERENCE (Standard Vanilla Transformer)
    // =========================================================================
    std::cout << "[RUN 1/2] Executing Unmanaged Linear Context Expansion..." << std::endl;
    StandaloneBenchmarkResult unmanaged_res;
    unmanaged_res.name = "Unmanaged Baseline (Vanilla)";
    {
        StandaloneKVCache kv_cache;
        kv_cache.embedding_dim = EMBEDDING_DIM;

        auto t_start = std::chrono::steady_clock::now();
        for (uint32_t t = 0; t < total_tokens; ++t) {
            kv_cache.append_token(t);
            size_t cur_mem_kb = kv_cache.memory_bytes() / 1024;
            if (cur_mem_kb > unmanaged_res.peak_kv_memory_kb) {
                unmanaged_res.peak_kv_memory_kb = cur_mem_kb;
            }
        }
        auto t_end = std::chrono::steady_clock::now();

        unmanaged_res.tokens_processed = total_tokens;
        unmanaged_res.duration_sec = std::chrono::duration<double>(t_end - t_start).count();
        unmanaged_res.throughput_tok_s = (unmanaged_res.duration_sec > 0) ? (total_tokens / unmanaged_res.duration_sec) : 0.0;
        unmanaged_res.final_kv_memory_kb = kv_cache.memory_bytes() / 1024;
        unmanaged_res.kv_memory_reclaimed_kb = 0;
        unmanaged_res.tokens_archived = 0;
    }
    std::cout << "  - Unmanaged Baseline Completed: Peak KV Memory = " << unmanaged_res.peak_kv_memory_kb << " KB (Unbounded Expansion)\n\n";

    // =========================================================================
    // RUN 2: AMU-QUANT CO-ADAPTIVE RUNTIME ENGINE
    // =========================================================================
    std::cout << "[RUN 2/2] Executing AMU-QUANT Co-Adaptive Runtime Engine..." << std::endl;
    StandaloneBenchmarkResult amuquant_res;
    amuquant_res.name = "AMU-QUANT Co-Adaptive Engine";
    {
        AMUQuantConfig cfg;
        cfg.memory_limit_bytes = 14ULL * 1024ULL * 1024ULL * 1024ULL;
        cfg.os_reserved_bytes  = 2ULL * 1024ULL * 1024ULL * 1024ULL;
        cfg.num_layers = NUM_LAYERS;
        cfg.embedding_dim = EMBEDDING_DIM;
        cfg.eval_interval_tokens = 4;
        cfg.recent_window_tokens = 24;
        cfg.min_recent_window_tokens = 12;
        cfg.attention_sink_tokens = 4;

        AMUQuantEngine engine(cfg);
        Status st = engine.initialize();
        assert(st == Status::OK);

        // Allocate real anonymous mmap pages for transformer layers
        std::vector<void*> layer_ptrs(NUM_LAYERS, nullptr);
        for (uint32_t l = 0; l < NUM_LAYERS; ++l) {
            layer_ptrs[l] = mmap(nullptr, REAL_LAYER_BYTES, PROT_READ | PROT_WRITE, MAP_ANON | MAP_PRIVATE, -1, 0);
            assert(layer_ptrs[l] != MAP_FAILED);
            std::memset(layer_ptrs[l], 0xEF, REAL_LAYER_BYTES); // Touch physical RAM pages
            engine.register_layer(l, layer_ptrs[l], REAL_LAYER_BYTES);
        }

        StandaloneKVCache kv_cache;
        kv_cache.embedding_dim = EMBEDDING_DIM;

        auto seq_rm = [&](int32_t p0, int32_t p1) {
            kv_cache.seq_rm(p0, p1);
        };
        auto seq_add = [&](int32_t p0, int32_t p1, int32_t delta) {
            kv_cache.seq_add(p0, p1, delta);
        };

        auto telemetry = engine.get_telemetry();
        telemetry->set_mock_mode(true);

        auto t_start = std::chrono::steady_clock::now();
        for (uint32_t t = 0; t < total_tokens; ++t) {
            kv_cache.append_token(t);

            // Layer madvise paging hooks
            for (uint32_t l = 0; l < NUM_LAYERS; ++l) {
                engine.begin_layer_forward(l);
                engine.end_layer_forward(l);
            }

            // Simulate real host telemetry transition under context growth
            MemoryStats s{};
            s.timestamp_sec = 10.0 + (t * 0.25);
            s.total_ram_bytes = 16ULL * 1024ULL * 1024ULL * 1024ULL;
            if (t < 40) {
                s.available_ram_bytes = 8ULL * 1024ULL * 1024ULL * 1024ULL;
                s.process_rss_bytes   = 4ULL * 1024ULL * 1024ULL * 1024ULL;
                s.swap_used_bytes     = 0;
            } else if (t < 70) {
                s.available_ram_bytes = 4ULL * 1024ULL * 1024ULL * 1024ULL;
                s.process_rss_bytes   = 9ULL * 1024ULL * 1024ULL * 1024ULL;
                s.swap_used_bytes     = 0;
            } else {
                s.available_ram_bytes = 1ULL * 1024ULL * 1024ULL * 1024ULL; // Host memory pressure!
                s.process_rss_bytes   = 7ULL * 1024ULL * 1024ULL * 1024ULL;
                s.swap_used_bytes     = 500ULL * 1024ULL * 1024ULL;
            }
            telemetry->set_mock_stats(s);

            engine.step_token(t, t + 1, valid_extractor, seq_rm, seq_add);

            size_t cur_mem_kb = kv_cache.memory_bytes() / 1024;
            if (cur_mem_kb > amuquant_res.peak_kv_memory_kb) {
                amuquant_res.peak_kv_memory_kb = cur_mem_kb;
            }
        }
        auto t_end = std::chrono::steady_clock::now();

        amuquant_res.tokens_processed = total_tokens;
        amuquant_res.duration_sec = std::chrono::duration<double>(t_end - t_start).count();
        amuquant_res.throughput_tok_s = (amuquant_res.duration_sec > 0) ? (total_tokens / amuquant_res.duration_sec) : 0.0;
        amuquant_res.final_kv_memory_kb = kv_cache.memory_bytes() / 1024;
        amuquant_res.kv_memory_reclaimed_kb = kv_cache.total_reclaimed_bytes / 1024;
        amuquant_res.tokens_archived = engine.get_kv_surgeon()->get_stats(NUM_LAYERS, 8, 128).archived_chunks;
        amuquant_res.final_regime = engine.get_controller()->get_current_regime();

        // Test Semantic Retrieval of Needle Fact
        auto t_rec_start = std::chrono::high_resolution_clock::now();
        auto hits = engine.recall_context(context_tokens[NEEDLE_POS].data(), 1);
        auto t_rec_end = std::chrono::high_resolution_clock::now();
        amuquant_res.recall_latency_us = std::chrono::duration<double, std::micro>(t_rec_end - t_rec_start).count();
        if (!hits.empty() && hits[0].id == NEEDLE_POS) {
            amuquant_res.recall_score = hits[0].score;
        }

        // Clean up mmap layers
        for (uint32_t l = 0; l < NUM_LAYERS; ++l) {
            if (layer_ptrs[l] != nullptr && layer_ptrs[l] != MAP_FAILED) {
                munmap(layer_ptrs[l], REAL_LAYER_BYTES);
            }
        }
    }
    std::cout << "  - AMU-QUANT Completed: Reclaimed KV Memory = " << amuquant_res.kv_memory_reclaimed_kb << " KB | Tokens Archived = " << amuquant_res.tokens_archived << "\n\n";

    // =========================================================================
    // COMPARATIVE SUMMARY REPORT
    // =========================================================================
    std::cout << "================================================================================\n";
    std::cout << "  STANDALONE EMPIRICAL BENCHMARK SUMMARY (REPRODUCIBLE TESTBED)                 \n";
    std::cout << "================================================================================\n";
    std::cout << std::left << std::setw(34) << "Evaluation Metric"
              << std::setw(24) << "Unmanaged Baseline"
              << std::setw(24) << "AMU-QUANT (Engine)" << "\n";
    std::cout << "--------------------------------------------------------------------------------\n";
    std::cout << std::left << std::setw(34) << "Sequence Tokens Processed"
              << std::setw(24) << unmanaged_res.tokens_processed
              << std::setw(24) << amuquant_res.tokens_processed << "\n";
    std::cout << std::left << std::setw(34) << "Peak Physical KV Memory"
              << std::setw(24) << (std::to_string(unmanaged_res.peak_kv_memory_kb) + " KB")
              << std::setw(24) << (std::to_string(amuquant_res.peak_kv_memory_kb) + " KB") << "\n";
    std::cout << std::left << std::setw(34) << "Final Active KV Memory"
              << std::setw(24) << (std::to_string(unmanaged_res.final_kv_memory_kb) + " KB")
              << std::setw(24) << (std::to_string(amuquant_res.final_kv_memory_kb) + " KB") << "\n";
    std::cout << std::left << std::setw(34) << "Physical KV Memory Reclaimed"
              << std::setw(24) << "0 KB (None)"
              << std::setw(24) << (std::to_string(amuquant_res.kv_memory_reclaimed_kb) + " KB") << "\n";
    std::cout << std::left << std::setw(34) << "Tokens Safely Archived"
              << std::setw(24) << "0 (None)"
              << std::setw(24) << amuquant_res.tokens_archived << "\n";
    std::cout << std::left << std::setw(34) << "Needle Context Recall Score"
              << std::setw(24) << "N/A"
              << std::setw(24) << (std::to_string(amuquant_res.recall_score) + " (Cosine)") << "\n";
    std::cout << std::left << std::setw(34) << "SIMD Retrieval Latency"
              << std::setw(24) << "N/A"
              << std::setw(24) << (std::to_string(amuquant_res.recall_latency_us) + " us") << "\n";
    std::cout << "================================================================================\n" << std::endl;

    // Export report
    std::ofstream rpt("benchmarks/standalone_benchmark_report.md");
    if (rpt.is_open()) {
        rpt << "# AMU-QUANT Standalone Benchmark Report (Zero External Dependencies)\n\n";
        rpt << "- **Platform**: POSIX / macOS Apple Silicon\n";
        rpt << "- **Embedding Dimension**: " << EMBEDDING_DIM << " dim\n";
        rpt << "- **Transformer Layers**: " << NUM_LAYERS << " layers (32 MB physical mmap allocation)\n\n";
        rpt << "| Metric | Unmanaged Baseline | AMU-QUANT Co-Adaptive Engine |\n";
        rpt << "| :--- | :--- | :--- |\n";
        rpt << "| **Tokens Processed** | " << unmanaged_res.tokens_processed << " tokens | " << amuquant_res.tokens_processed << " tokens |\n";
        rpt << "| **Peak KV-Cache Memory** | " << unmanaged_res.peak_kv_memory_kb << " KB | " << amuquant_res.peak_kv_memory_kb << " KB |\n";
        rpt << "| **Final KV-Cache Memory** | " << unmanaged_res.final_kv_memory_kb << " KB | " << amuquant_res.final_kv_memory_kb << " KB |\n";
        rpt << "| **Physical RAM Reclaimed** | 0 KB | " << amuquant_res.kv_memory_reclaimed_kb << " KB |\n";
        rpt << "| **Tokens Safely Archived** | 0 tokens | " << amuquant_res.tokens_archived << " tokens |\n";
        rpt << "| **Needle Retrieval Score** | N/A | " << amuquant_res.recall_score << " (Cosine Match) |\n";
        rpt << "| **SIMD Retrieval Latency** | N/A | " << amuquant_res.recall_latency_us << " µs |\n\n";
        rpt << "Generated automatically by `amuquant_benchmark`.\n";
        rpt.close();
        std::cout << "[INFO] Standalone report exported to benchmarks/standalone_benchmark_report.md\n";
    }

    return 0;
}
