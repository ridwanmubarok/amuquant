#include "amuquant/engine.hpp"

#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>
#include <random>

using namespace amuquant;

int main() {
    std::cout << "[TEST] Starting End-to-End AMU-QUANT Pipeline Simulation..." << std::endl;

    AMUQuantConfig cfg;
    cfg.memory_limit_bytes = 14ULL * 1024ULL * 1024ULL * 1024ULL; // 14 GB Safe Limit
    cfg.os_reserved_bytes = 2ULL * 1024ULL * 1024ULL * 1024ULL;
    cfg.num_layers = 32;
    cfg.embedding_dim = 1024; // Gunakan 1024-dim untuk simulasi cepat
    cfg.eval_interval_tokens = 4;
    cfg.recent_window_tokens = 32;
    cfg.min_recent_window_tokens = 16;
    cfg.attention_sink_tokens = 4;

    AMUQuantEngine engine(cfg);
    Status st = engine.initialize();
    assert(st == Status::OK);
    std::cout << "  - Engine initialized: OK" << std::endl;

    // 1. Simulasikan alokasi mmap untuk 32 layer model
    constexpr size_t LAYER_SIZE = 128 * 1024 * 1024; // 128 MB per layer (~4 GB total model)
    std::vector<std::vector<uint8_t>> simulated_layer_buffers(cfg.num_layers, std::vector<uint8_t>(4096, 0));
    for (uint32_t l = 0; l < cfg.num_layers; ++l) {
        engine.register_layer(l, simulated_layer_buffers[l].data(), LAYER_SIZE);
    }
    std::cout << "  - Registered 32 model layers: OK" << std::endl;

    // 2. Siapkan token embedding buffer untuk melacak konteks
    constexpr uint32_t TOTAL_STEPS = 120;
    std::vector<std::vector<float>> context_tokens(TOTAL_STEPS, std::vector<float>(cfg.embedding_dim, 0.0f));

    // Masukkan "Needle" fakta unik pada token ke-15 menggunakan vektor acak unik
    constexpr uint32_t NEEDLE_POS = 15;
    std::mt19937 needle_rng(9999);
    std::normal_distribution<float> needle_dist(0.0f, 1.0f);
    for (uint32_t d = 0; d < cfg.embedding_dim; ++d) {
        context_tokens[NEEDLE_POS][d] = needle_dist(needle_rng);
    }

    auto extractor = [&](int32_t pos) -> const float* {
        if (pos >= 0 && pos < static_cast<int32_t>(context_tokens.size())) {
            return context_tokens[pos].data();
        }
        return nullptr;
    };

    uint32_t pruned_count = 0;
    auto seq_rm = [&](int32_t p0, int32_t p1) {
        pruned_count += (p1 - p0);
    };

    auto seq_add = [](int32_t, int32_t, int32_t) {};

    // 3. Jalankan loop pembuatan token autoregresif
    auto telemetry = engine.get_telemetry();
    telemetry->set_mock_mode(true);

    for (uint32_t t = 0; t < TOTAL_STEPS; ++t) {
        // Simulasikan forward pass layer demi layer
        for (uint32_t l = 0; l < cfg.num_layers; ++l) {
            engine.begin_layer_forward(l);
            // Komputasi layer selesai
            engine.end_layer_forward(l);
        }

        // Simulasikan kenaikan memori: normal di awal, lalu melonjak di t = 40
        MemoryStats s{};
        s.timestamp_sec = 10.0 + (t * 0.5); // Waktu bertambah bertahap
        if (t < 40) {
            s.process_rss_bytes = 4ULL * 1024ULL * 1024ULL * 1024ULL; // 4 GB (~33% Safe Limit -> Regime 1)
        } else if (t < 80) {
            s.process_rss_bytes = static_cast<uint64_t>(12ULL * 1024ULL * 1024ULL * 1024ULL * 0.72); // 72% Safe Limit -> Regime 2
        } else {
            s.process_rss_bytes = static_cast<uint64_t>(12ULL * 1024ULL * 1024ULL * 1024ULL * 0.92); // 92% Safe Limit -> Regime 3 Kritis
        }
        telemetry->set_mock_stats(s);

        // Buat representasi vektor untuk token t jika belum ada
        if (t != NEEDLE_POS) {
            std::mt19937 bg_rng(100 + t);
            std::normal_distribution<float> bg_dist(0.0f, 1.0f);
            for (uint32_t d = 0; d < cfg.embedding_dim; ++d) {
                context_tokens[t][d] = bg_dist(bg_rng);
            }
        }

        // Eksekusi step kontroler AMU-QUANT
        StepDecision dec = engine.step_token(t, t + 1, extractor, seq_rm, seq_add);

        if (t == 50) {
            assert(dec.regime == Regime::REGIME_2_BALANCED_OFFLOAD);
            assert(dec.aggressive_evict == true);
        } else if (t == 90) {
            assert(dec.regime == Regime::REGIME_3_ULTRA_LOW_SURVIVAL);
            assert(dec.prefetch_depth == 0);
        }
    }

    std::cout << "  - Autoregressive Generation finished 120 steps: OK" << std::endl;
    std::cout << "  - Total tokens pruned and archived to turbovec: " << pruned_count << std::endl;
    assert(pruned_count > 0);

    // 4. Uji Pengambilan Kembali Fakta Jarum (Needle-in-a-Haystack Retrieval)
    auto results = engine.recall_context(context_tokens[NEEDLE_POS].data(), 3);
    assert(!results.empty());
    std::cout << "  - Context Recall Top-1 Chunk ID: " << results[0].id
              << ", Score: " << results[0].score << std::endl;
    assert(results[0].id == NEEDLE_POS);
    assert(results[0].score > 0.85f);
    std::cout << "  - Needle-in-a-Haystack Recall verified successfully!" << std::endl;

    // 5. Verifikasi Metrik Engine
    EngineMetrics m = engine.get_metrics();
    std::cout << "  - Avg Inter-Token Latency: " << m.avg_inter_token_latency_ms << " ms" << std::endl;
    std::cout << "  - Peak RSS: " << (m.peak_rss_bytes / (1024 * 1024)) << " MB" << std::endl;

    std::cout << "[PASS] End-to-End AMU-QUANT Pipeline Simulation Succeeded.\n" << std::endl;
    return 0;
}
