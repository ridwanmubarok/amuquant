#include "amuquant/engine.hpp"
#include "amuquant/gguf_parser.hpp"
#include "llama.h"

#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <chrono>
#include <thread>
#include <cassert>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <cstring>
#include <algorithm>

#if defined(__APPLE__)
#include <sys/sysctl.h>
#endif

using namespace amuquant;

struct BenchmarkRunResult {
    std::string name;
    bool completed{false};
    bool natural_eos{false};
    uint32_t tokens_generated{0};
    double duration_sec{0.0};
    double throughput_tok_s{0.0};
    uint64_t initial_rss_mb{0};
    uint64_t peak_rss_mb{0};
    uint64_t final_rss_mb{0};
    uint32_t kv_tokens_archived{0};
    size_t kv_memory_reclaimed_kb{0};
    double recall_score{0.0};
    double recall_latency_ms{0.0};
    Regime final_regime{Regime::REGIME_1_HIGH_THROUGHPUT};
};

/**
 * @brief High-performance contiguous embedding storage eliminating per-token heap allocations.
 */
struct ContiguousEmbeddingStore {
    uint32_t embedding_dim{4096};
    std::vector<float> data_pool;
    size_t count{0};

    void reserve(size_t max_tokens, uint32_t dim) {
        embedding_dim = dim;
        data_pool.resize(max_tokens * dim, 0.0f);
        count = 0;
    }

    void append(const float* embd) {
        if (embd != nullptr) {
            std::memcpy(&data_pool[count * embedding_dim], embd, embedding_dim * sizeof(float));
        } else {
            std::memset(&data_pool[count * embedding_dim], 0, embedding_dim * sizeof(float));
        }
        count++;
    }

    const float* get(size_t index) const {
        if (index < count) return &data_pool[index * embedding_dim];
        return nullptr;
    }

    size_t size() const { return count; }

    void clear() { count = 0; }
};

static int get_optimal_pcores() {
#if defined(__APPLE__)
    int pcores = 0;
    size_t len = sizeof(pcores);
    if (sysctlbyname("hw.perflevel0.logicalcpu", &pcores, &len, nullptr, 0) == 0 && pcores > 0) {
        return pcores;
    }
#endif
    return std::max(1u, std::thread::hardware_concurrency());
}

int main(int argc, char** argv) {
    std::string model_path = "models/Meta-Llama-3-8B-Instruct-Q2_K.gguf";
    uint32_t target_tokens = 64;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--model" && i + 1 < argc) {
            model_path = argv[++i];
        } else if (arg == "--tokens" && i + 1 < argc) {
            target_tokens = static_cast<uint32_t>(std::stoi(argv[++i]));
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: live_benchmark [--model <path.gguf>] [--tokens <count>]\n";
            return 0;
        }
    }

    std::cout << "================================================================================\n";
    std::cout << "  AMU-QUANT LIVE EMPIRICAL BENCHMARK: 3-WAY METHODOLOGICAL EVALUATION           \n";
    std::cout << "  Target Model: " << model_path << "\n";
    std::cout << "  Decode Horizon: " << target_tokens << " tokens\n";
    std::cout << "================================================================================\n" << std::endl;

    if (!std::filesystem::exists(model_path)) {
        std::cerr << "Model file not found: " << model_path << "\n";
        std::cerr << "Please specify a valid GGUF model via --model <path>\n";
        return 1;
    }

    GGUFParser parser;
    GGUFModelInfo model_info;
    parser.parse_file(model_path, model_info);
    int pcore_threads = get_optimal_pcores();

    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 99; // Offload to Metal GPU if available

    llama_model* model = llama_model_load_from_file(model_path.c_str(), mparams);
    if (!model) {
        std::cerr << "Failed to load model from file: " << model_path << "\n";
        llama_backend_free();
        return 1;
    }

    const llama_vocab* vocab = llama_model_get_vocab(model);
    uint32_t embedding_length = model_info.embedding_length > 0 ? model_info.embedding_length : 4096;
    uint32_t block_count = model_info.block_count > 0 ? model_info.block_count : 32;

    std::string test_prompt = 
        "Artificial intelligence and computer architecture are converging on resource-constrained devices. "
        "Explain the mathematical principles of scalar quantization, dynamic memory equilibrium, and "
        "low-latency vector search across large-scale autoregressive language model inference.";

    std::vector<llama_token> prompt_tokens(test_prompt.size() + 32);
    int n_prompt = llama_tokenize(vocab, test_prompt.c_str(), test_prompt.size(), prompt_tokens.data(), prompt_tokens.size(), true, true);
    assert(n_prompt > 0);

    SystemTelemetry telemetry_sampler;

    // Pre-warm Metal GPU shader compilation pipeline so cold-start compilation does not distort Run 1
    std::cout << "[INFO] Pre-warming GPU Metal kernel pipelines..." << std::flush;
    {
        llama_context_params w_cparams = llama_context_default_params();
        w_cparams.n_ctx = 256;
        w_cparams.n_batch = 128;
        w_cparams.n_threads = pcore_threads;
        w_cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        llama_context* w_ctx = llama_init_from_model(model, w_cparams);
        if (w_ctx) {
            llama_batch w_batch = llama_batch_init(8, 0, 1);
            w_batch.token[0] = prompt_tokens[0];
            w_batch.pos[0] = 0;
            w_batch.n_seq_id[0] = 1;
            w_batch.seq_id[0][0] = 0;
            w_batch.logits[0] = true;
            w_batch.n_tokens = 1;
            llama_decode(w_ctx, w_batch);
            llama_batch_free(w_batch);
            llama_free(w_ctx);
        }
    }
    std::cout << " Done.\n\n";

    // =========================================================================
    // RUN 1: UNINSTRUMENTED BASELINE (Vanilla llama.cpp, embeddings = false)
    // =========================================================================
    std::cout << "[RUN 1/3] Executing Baseline 1 (Standard llama.cpp, no embeddings)..." << std::endl;
    BenchmarkRunResult raw_baseline_res;
    raw_baseline_res.name = "Baseline 1 (Raw Vanilla)";

    {
        llama_context_params cparams = llama_context_default_params();
        cparams.n_ctx = 2048;
        cparams.n_batch = 512;
        cparams.n_threads = pcore_threads;
        cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        cparams.embeddings = false;

        llama_context* ctx = llama_init_from_model(model, cparams);
        llama_sampler* smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
        llama_sampler_chain_add(smpl, llama_sampler_init_top_k(40));
        llama_sampler_chain_add(smpl, llama_sampler_init_temp(0.7f));
        llama_sampler_chain_add(smpl, llama_sampler_init_dist(42));

        MemoryStats m_init = telemetry_sampler.sample();
        raw_baseline_res.initial_rss_mb = m_init.process_rss_bytes / (1024 * 1024);
        raw_baseline_res.peak_rss_mb = raw_baseline_res.initial_rss_mb;

        llama_batch batch = llama_batch_init(512, 0, 1);
        batch.n_tokens = 0;
        for (int i = 0; i < n_prompt; ++i) {
            batch.token[batch.n_tokens] = prompt_tokens[i];
            batch.pos[batch.n_tokens] = i;
            batch.n_seq_id[batch.n_tokens] = 1;
            batch.seq_id[batch.n_tokens][0] = 0;
            batch.logits[batch.n_tokens] = (i == n_prompt - 1);
            batch.n_tokens++;
        }

        llama_decode(ctx, batch);
        int32_t n_past = n_prompt;

        auto t_start = std::chrono::steady_clock::now();
        for (uint32_t step = 0; step < target_tokens; ++step) {
            llama_token new_token = llama_sampler_sample(smpl, ctx, -1);
            llama_sampler_accept(smpl, new_token);

            if (llama_vocab_is_eog(vocab, new_token)) {
                raw_baseline_res.natural_eos = true;
                break;
            }

            batch.n_tokens = 1;
            batch.token[0] = new_token;
            batch.pos[0] = n_past;
            batch.n_seq_id[0] = 1;
            batch.seq_id[0][0] = 0;
            batch.logits[0] = true;

            if (llama_decode(ctx, batch) != 0) break;

            n_past++;
            raw_baseline_res.tokens_generated++;

            if (step % 8 == 0) {
                MemoryStats s = telemetry_sampler.sample();
                uint64_t rss_mb = s.process_rss_bytes / (1024 * 1024);
                if (rss_mb > raw_baseline_res.peak_rss_mb) raw_baseline_res.peak_rss_mb = rss_mb;
            }
        }
        auto t_end = std::chrono::steady_clock::now();

        raw_baseline_res.duration_sec = std::chrono::duration<double>(t_end - t_start).count();
        raw_baseline_res.throughput_tok_s = (raw_baseline_res.tokens_generated > 0 && raw_baseline_res.duration_sec > 0) ?
                                            (raw_baseline_res.tokens_generated / raw_baseline_res.duration_sec) : 0.0;
        raw_baseline_res.final_rss_mb = telemetry_sampler.sample().process_rss_bytes / (1024 * 1024);
        raw_baseline_res.completed = (raw_baseline_res.tokens_generated == target_tokens || raw_baseline_res.natural_eos);

        llama_batch_free(batch);
        llama_sampler_free(smpl);
        llama_free(ctx);
    }
    std::cout << "  - Baseline 1 Completed: " << raw_baseline_res.tokens_generated << "/" << target_tokens << " tokens | "
              << "Peak RSS: " << raw_baseline_res.peak_rss_mb << " MB | "
              << "Throughput: " << std::fixed << std::setprecision(1) << raw_baseline_res.throughput_tok_s << " tok/s\n\n";

    // =========================================================================
    // RUN 2: PARITY BASELINE (Standard llama.cpp with embeddings enabled)
    // =========================================================================
    std::cout << "[RUN 2/3] Executing Baseline 2 (Parity with embeddings enabled)..." << std::endl;
    BenchmarkRunResult parity_baseline_res;
    parity_baseline_res.name = "Baseline 2 (With Embeddings)";

    {
        llama_context_params cparams = llama_context_default_params();
        cparams.n_ctx = 2048;
        cparams.n_batch = 512;
        cparams.n_threads = pcore_threads;
        cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        cparams.embeddings = true;

        llama_context* ctx = llama_init_from_model(model, cparams);
        llama_sampler* smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
        llama_sampler_chain_add(smpl, llama_sampler_init_top_k(40));
        llama_sampler_chain_add(smpl, llama_sampler_init_temp(0.7f));
        llama_sampler_chain_add(smpl, llama_sampler_init_dist(42));

        ContiguousEmbeddingStore parity_embeddings;
        parity_embeddings.reserve(n_prompt + target_tokens + 32, embedding_length);

        MemoryStats m_init = telemetry_sampler.sample();
        parity_baseline_res.initial_rss_mb = m_init.process_rss_bytes / (1024 * 1024);
        parity_baseline_res.peak_rss_mb = parity_baseline_res.initial_rss_mb;

        llama_batch batch = llama_batch_init(512, 0, 1);
        batch.n_tokens = 0;
        for (int i = 0; i < n_prompt; ++i) {
            batch.token[batch.n_tokens] = prompt_tokens[i];
            batch.pos[batch.n_tokens] = i;
            batch.n_seq_id[batch.n_tokens] = 1;
            batch.seq_id[batch.n_tokens][0] = 0;
            batch.logits[batch.n_tokens] = true;
            batch.n_tokens++;
        }

        llama_decode(ctx, batch);
        for (int i = 0; i < n_prompt; ++i) {
            const float* embd = llama_get_embeddings_ith(ctx, i);
            parity_embeddings.append(embd);
        }

        int32_t n_past = n_prompt;

        auto t_start = std::chrono::steady_clock::now();
        for (uint32_t step = 0; step < target_tokens; ++step) {
            llama_token new_token = llama_sampler_sample(smpl, ctx, -1);
            llama_sampler_accept(smpl, new_token);

            if (llama_vocab_is_eog(vocab, new_token)) {
                parity_baseline_res.natural_eos = true;
                break;
            }

            batch.n_tokens = 1;
            batch.token[0] = new_token;
            batch.pos[0] = n_past;
            batch.n_seq_id[0] = 1;
            batch.seq_id[0][0] = 0;
            batch.logits[0] = true;

            if (llama_decode(ctx, batch) != 0) break;

            const float* embd = llama_get_embeddings_ith(ctx, 0);
            parity_embeddings.append(embd);

            n_past++;
            parity_baseline_res.tokens_generated++;

            if (step % 8 == 0) {
                MemoryStats s = telemetry_sampler.sample();
                uint64_t rss_mb = s.process_rss_bytes / (1024 * 1024);
                if (rss_mb > parity_baseline_res.peak_rss_mb) parity_baseline_res.peak_rss_mb = rss_mb;
            }
        }
        auto t_end = std::chrono::steady_clock::now();

        parity_baseline_res.duration_sec = std::chrono::duration<double>(t_end - t_start).count();
        parity_baseline_res.throughput_tok_s = (parity_baseline_res.tokens_generated > 0 && parity_baseline_res.duration_sec > 0) ?
                                               (parity_baseline_res.tokens_generated / parity_baseline_res.duration_sec) : 0.0;
        parity_baseline_res.final_rss_mb = telemetry_sampler.sample().process_rss_bytes / (1024 * 1024);
        parity_baseline_res.completed = (parity_baseline_res.tokens_generated == target_tokens || parity_baseline_res.natural_eos);

        llama_batch_free(batch);
        llama_sampler_free(smpl);
        llama_free(ctx);
    }
    std::cout << "  - Baseline 2 Completed: " << parity_baseline_res.tokens_generated << "/" << target_tokens << " tokens | "
              << "Peak RSS: " << parity_baseline_res.peak_rss_mb << " MB | "
              << "Throughput: " << std::fixed << std::setprecision(1) << parity_baseline_res.throughput_tok_s << " tok/s\n\n";

    // =========================================================================
    // RUN 3: AMU-QUANT CO-ADAPTIVE RUNTIME ENGINE
    // =========================================================================
    std::cout << "[RUN 3/3] Executing AMU-QUANT Co-Adaptive Runtime Engine..." << std::endl;
    BenchmarkRunResult amuquant_res;
    amuquant_res.name = "AMU-QUANT Co-Adaptive Engine";

    {
        AMUQuantConfig cfg;
        cfg.memory_limit_bytes = 4800ULL * 1024ULL * 1024ULL; // Target working budget
        cfg.os_reserved_bytes  = 600ULL * 1024ULL * 1024ULL;  // OS stability headroom
        cfg.num_layers = block_count;
        cfg.embedding_dim = embedding_length;
        cfg.eval_interval_tokens = 8; // Lightweight telemetry cadence
        cfg.min_eviction_batch = 16;  // Batched eviction minimizing Metal GPU flushes
        cfg.recent_window_tokens = 32;
        cfg.min_recent_window_tokens = 16;
        cfg.attention_sink_tokens = 4;

        AMUQuantEngine engine(cfg);
        Status st = engine.initialize();
        assert(st == Status::OK);

        llama_context_params cparams = llama_context_default_params();
        cparams.n_ctx = 2048;
        cparams.n_batch = 512;
        cparams.n_threads = pcore_threads;
        cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        cparams.embeddings = true;

        llama_context* ctx = llama_init_from_model(model, cparams);
        llama_sampler* smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
        llama_sampler_chain_add(smpl, llama_sampler_init_top_k(40));
        llama_sampler_chain_add(smpl, llama_sampler_init_temp(0.7f));
        llama_sampler_chain_add(smpl, llama_sampler_init_dist(42));

        // Persistent context representations in zero-allocation pool
        ContiguousEmbeddingStore session_embeddings;
        session_embeddings.reserve(n_prompt + target_tokens + 32, embedding_length);
        llama_memory_t mem = llama_get_memory(ctx);

        auto extractor = [&](int32_t logical_id) -> const float* {
            return session_embeddings.get(static_cast<size_t>(logical_id));
        };

        auto seq_rm = [&](int32_t p0, int32_t p1) {
            llama_memory_seq_rm(mem, 0, p0, p1);
            uint32_t n_kv_heads = model_info.head_count_kv > 0 ? model_info.head_count_kv : 8;
            size_t bytes_per_token = 2ULL * block_count * n_kv_heads * 128 * sizeof(uint16_t);
            amuquant_res.kv_memory_reclaimed_kb += ((p1 - p0) * bytes_per_token) / 1024;
        };

        auto seq_add = [&](int32_t p0, int32_t p1, int32_t delta) {
            llama_memory_seq_add(mem, 0, p0, p1, delta);
        };

        MemoryStats m_init = telemetry_sampler.sample();
        amuquant_res.initial_rss_mb = m_init.process_rss_bytes / (1024 * 1024);
        amuquant_res.peak_rss_mb = amuquant_res.initial_rss_mb;

        // Initialize TokenIndexMap with prompt sequence
        engine.init_token_sequence(n_prompt);

        llama_batch batch = llama_batch_init(512, 0, 1);
        batch.n_tokens = 0;
        for (int i = 0; i < n_prompt; ++i) {
            batch.token[batch.n_tokens] = prompt_tokens[i];
            batch.pos[batch.n_tokens] = i;
            batch.n_seq_id[batch.n_tokens] = 1;
            batch.seq_id[batch.n_tokens][0] = 0;
            batch.logits[batch.n_tokens] = true;
            batch.n_tokens++;
        }

        llama_decode(ctx, batch);

        // Extract genuine hidden-state embeddings for all prompt tokens into contiguous pool
        for (int i = 0; i < n_prompt; ++i) {
            const float* embd = llama_get_embeddings_ith(ctx, i);
            session_embeddings.append(embd);
        }

        int32_t n_past = n_prompt;

        auto t_start = std::chrono::steady_clock::now();
        for (uint32_t step = 0; step < target_tokens; ++step) {
            llama_token new_token = llama_sampler_sample(smpl, ctx, -1);
            llama_sampler_accept(smpl, new_token);

            if (llama_vocab_is_eog(vocab, new_token)) {
                amuquant_res.natural_eos = true;
                break;
            }

            // Register newly decoded logical token into TokenIndexMap
            engine.append_token(n_prompt + step);

            // Active physical sequence length in KV-cache
            int32_t current_physical_len = llama_memory_seq_pos_max(mem, 0) + 1;

            // AMU-QUANT controller arbitration and transactional KV surgery
            engine.step_token(n_past, current_physical_len, extractor, seq_rm, seq_add);

            llama_pos next_pos = llama_memory_seq_pos_max(mem, 0) + 1;
            batch.n_tokens = 1;
            batch.token[0] = new_token;
            batch.pos[0] = next_pos;
            batch.n_seq_id[0] = 1;
            batch.seq_id[0][0] = 0;
            batch.logits[0] = true;

            if (llama_decode(ctx, batch) != 0) break;

            const float* embd = llama_get_embeddings_ith(ctx, 0);
            session_embeddings.append(embd);

            n_past++;
            amuquant_res.tokens_generated++;

            if (step % 8 == 0) {
                MemoryStats s = telemetry_sampler.sample();
                uint64_t rss_mb = s.process_rss_bytes / (1024 * 1024);
                if (rss_mb > amuquant_res.peak_rss_mb) amuquant_res.peak_rss_mb = rss_mb;
            }
        }
        auto t_end = std::chrono::steady_clock::now();

        amuquant_res.duration_sec = std::chrono::duration<double>(t_end - t_start).count();
        amuquant_res.throughput_tok_s = (amuquant_res.tokens_generated > 0 && amuquant_res.duration_sec > 0) ?
                                        (amuquant_res.tokens_generated / amuquant_res.duration_sec) : 0.0;
        amuquant_res.final_rss_mb = telemetry_sampler.sample().process_rss_bytes / (1024 * 1024);
        amuquant_res.final_regime = engine.get_controller()->get_current_regime();
        uint32_t n_kv_heads = model_info.head_count_kv > 0 ? model_info.head_count_kv : 8;
        amuquant_res.kv_tokens_archived = engine.get_kv_surgeon()->get_stats(block_count, n_kv_heads, 128).archived_chunks;
        amuquant_res.completed = (amuquant_res.tokens_generated == target_tokens || amuquant_res.natural_eos);

        // Test Semantic Retrieval on an archived middle prompt token (e.g. index 8)
        constexpr uint32_t TEST_ARCHIVE_POS = 8;
        if (TEST_ARCHIVE_POS < session_embeddings.size()) {
            auto t_rec_start = std::chrono::high_resolution_clock::now();
            auto recall_results = engine.recall_context(session_embeddings.get(TEST_ARCHIVE_POS), 3);
            auto t_rec_end = std::chrono::high_resolution_clock::now();
            amuquant_res.recall_latency_ms = std::chrono::duration<double, std::milli>(t_rec_end - t_rec_start).count();
            if (!recall_results.empty() && recall_results[0].id == TEST_ARCHIVE_POS) {
                amuquant_res.recall_score = recall_results[0].score;
            }
        }

        llama_batch_free(batch);
        llama_sampler_free(smpl);
        llama_free(ctx);
    }

    std::cout << "  - AMU-QUANT Completed: " << amuquant_res.tokens_generated << "/" << target_tokens << " tokens | "
              << "Peak RSS: " << amuquant_res.peak_rss_mb << " MB | "
              << "Tokens Archived: " << amuquant_res.kv_tokens_archived << " | "
              << "Throughput: " << std::fixed << std::setprecision(1) << amuquant_res.throughput_tok_s << " tok/s\n\n";

    llama_model_free(model);
    llama_backend_free();

    // =========================================================================
    // 3-WAY COMPARATIVE SUMMARY REPORT
    // =========================================================================
    std::cout << "================================================================================\n";
    std::cout << "  EMPIRICAL BENCHMARK SUMMARY (3-WAY METHODOLOGICAL PARITY TESTBED)             \n";
    std::cout << "================================================================================\n";
    std::cout << std::left << std::setw(32) << "Evaluation Metric"
              << std::setw(20) << "Baseline 1 (Raw)"
              << std::setw(20) << "Baseline 2 (Parity)"
              << std::setw(20) << "AMU-QUANT" << "\n";
    std::cout << "--------------------------------------------------------------------------------\n";
    std::cout << std::left << std::setw(32) << "Target Horizon Completed"
              << std::setw(20) << (raw_baseline_res.completed ? "Yes (100%)" : "Incomplete")
              << std::setw(20) << (parity_baseline_res.completed ? "Yes (100%)" : "Incomplete")
              << std::setw(20) << (amuquant_res.completed ? "Yes (100%)" : "Incomplete") << "\n";
    std::cout << std::left << std::setw(32) << "Tokens Generated"
              << std::setw(20) << (std::to_string(raw_baseline_res.tokens_generated) + "/" + std::to_string(target_tokens))
              << std::setw(20) << (std::to_string(parity_baseline_res.tokens_generated) + "/" + std::to_string(target_tokens))
              << std::setw(20) << (std::to_string(amuquant_res.tokens_generated) + "/" + std::to_string(target_tokens)) << "\n";
    std::cout << std::left << std::setw(32) << "Peak Physical RSS (MB)"
              << std::setw(20) << raw_baseline_res.peak_rss_mb
              << std::setw(20) << parity_baseline_res.peak_rss_mb
              << std::setw(20) << amuquant_res.peak_rss_mb << "\n";
    std::cout << std::left << std::setw(32) << "Inference Throughput (tok/s)"
              << std::setw(20) << std::fixed << std::setprecision(2) << raw_baseline_res.throughput_tok_s
              << std::setw(20) << std::fixed << std::setprecision(2) << parity_baseline_res.throughput_tok_s
              << std::setw(20) << std::fixed << std::setprecision(2) << amuquant_res.throughput_tok_s << "\n";
    std::cout << std::left << std::setw(32) << "KV Memory Reclaimed (KB)"
              << std::setw(20) << "0 KB"
              << std::setw(20) << "0 KB"
              << std::setw(20) << (std::to_string(amuquant_res.kv_memory_reclaimed_kb) + " KB") << "\n";
    std::cout << std::left << std::setw(32) << "Context Tokens Archived"
              << std::setw(20) << "0"
              << std::setw(20) << "0"
              << std::setw(20) << amuquant_res.kv_tokens_archived << "\n";
    std::cout << std::left << std::setw(32) << "Semantic Context Recall"
              << std::setw(20) << "N/A"
              << std::setw(20) << "N/A"
              << std::setw(20) << std::fixed << std::setprecision(4) << amuquant_res.recall_score << "\n";
    std::cout << std::left << std::setw(32) << "SIMD Retrieval Latency"
              << std::setw(20) << "N/A"
              << std::setw(20) << "N/A"
              << std::setw(20) << (std::to_string(amuquant_res.recall_latency_ms) + " ms") << "\n";
    std::cout << "================================================================================\n";

    // Export Markdown Report
    std::ofstream rpt("benchmarks/live_benchmark_report.md");
    if (rpt.is_open()) {
        rpt << "# AMU-QUANT Live Empirical Benchmark Report (3-Way Methodological Parity)\n\n";
        rpt << "- **Model**: `" << model_path << "`\n";
        rpt << "- **Architecture**: " << block_count << " Layers, Dim " << embedding_length << "\n";
        rpt << "- **Target Generation Horizon**: " << target_tokens << " tokens\n\n";
        rpt << "| Evaluation Metric | Baseline 1 (Raw Vanilla) | Baseline 2 (With Embeddings) | AMU-QUANT Co-Adaptive Engine |\n";
        rpt << "| :--- | :--- | :--- | :--- |\n";
        rpt << "| **Target Horizon Completed** | " << (raw_baseline_res.completed ? "Yes (100%)" : "Incomplete")
            << " | " << (parity_baseline_res.completed ? "Yes (100%)" : "Incomplete")
            << " | " << (amuquant_res.completed ? "Yes (100%)" : "Incomplete") << " |\n";
        rpt << "| **Tokens Generated** | " << raw_baseline_res.tokens_generated << "/" << target_tokens
            << " | " << parity_baseline_res.tokens_generated << "/" << target_tokens
            << " | " << amuquant_res.tokens_generated << "/" << target_tokens << " |\n";
        rpt << "| **Peak Physical RSS** | " << raw_baseline_res.peak_rss_mb << " MB | "
            << parity_baseline_res.peak_rss_mb << " MB | "
            << amuquant_res.peak_rss_mb << " MB |\n";
        rpt << "| **Inference Throughput** | " << std::fixed << std::setprecision(2) << raw_baseline_res.throughput_tok_s << " tok/s | "
            << parity_baseline_res.throughput_tok_s << " tok/s | "
            << amuquant_res.throughput_tok_s << " tok/s |\n";
        rpt << "| **KV Memory Reclaimed** | 0 KB | 0 KB | "
            << amuquant_res.kv_memory_reclaimed_kb << " KB |\n";
        rpt << "| **Tokens Safely Archived** | 0 | 0 | "
            << amuquant_res.kv_tokens_archived << " |\n";
        rpt << "| **Semantic Recall Score** | N/A | N/A | "
            << std::fixed << std::setprecision(4) << amuquant_res.recall_score << " |\n";
        rpt << "| **SIMD Retrieval Latency** | N/A | N/A | "
            << std::fixed << std::setprecision(3) << amuquant_res.recall_latency_ms << " ms |\n\n";
        rpt << "Generated automatically by `live_benchmark` on Apple Silicon / Metal.\n";
        rpt.close();
        std::cout << "\n[INFO] Live benchmark report exported to benchmarks/live_benchmark_report.md" << std::endl;
    }

    return 0;
}
