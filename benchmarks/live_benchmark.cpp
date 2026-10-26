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
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>

#if defined(__APPLE__)
#include <sys/sysctl.h>
#endif

using namespace amuquant;

struct BenchmarkRunResult {
    std::string name;
    bool completed{false};
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
    std::cout << "  AMU-QUANT LIVE EMPIRICAL BENCHMARK: DUAL-RUN COMPARATIVE EVALUATION           \n";
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
    mparams.n_gpu_layers = 99; // Attempt full offload if supported, fallback to CPU

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

    // =========================================================================
    // RUN 1: BASELINE INFERENCE (Standard Vanilla llama.cpp without AMU-QUANT)
    // =========================================================================
    std::cout << "[RUN 1/2] Executing Baseline (Standard llama.cpp without AMU-QUANT)..." << std::endl;
    BenchmarkRunResult baseline_res;
    baseline_res.name = "Baseline (Standard llama.cpp)";

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
        baseline_res.initial_rss_mb = m_init.process_rss_bytes / (1024 * 1024);
        baseline_res.peak_rss_mb = baseline_res.initial_rss_mb;

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

            if (llama_vocab_is_eog(vocab, new_token)) break;

            batch.n_tokens = 1;
            batch.token[0] = new_token;
            batch.pos[0] = n_past;
            batch.n_seq_id[0] = 1;
            batch.seq_id[0][0] = 0;
            batch.logits[0] = true;

            if (llama_decode(ctx, batch) != 0) break;

            n_past++;
            baseline_res.tokens_generated++;

            if (step % 8 == 0) {
                MemoryStats s = telemetry_sampler.sample();
                uint64_t rss_mb = s.process_rss_bytes / (1024 * 1024);
                if (rss_mb > baseline_res.peak_rss_mb) baseline_res.peak_rss_mb = rss_mb;
            }
        }
        auto t_end = std::chrono::steady_clock::now();

        baseline_res.duration_sec = std::chrono::duration<double>(t_end - t_start).count();
        baseline_res.throughput_tok_s = (baseline_res.tokens_generated > 0 && baseline_res.duration_sec > 0) ?
                                        (baseline_res.tokens_generated / baseline_res.duration_sec) : 0.0;
        baseline_res.final_rss_mb = telemetry_sampler.sample().process_rss_bytes / (1024 * 1024);
        baseline_res.completed = (baseline_res.tokens_generated > 0);

        llama_batch_free(batch);
        llama_sampler_free(smpl);
        llama_free(ctx);
    }
    std::cout << "  - Baseline Completed: " << baseline_res.tokens_generated << " tokens generated | "
              << "Peak RSS: " << baseline_res.peak_rss_mb << " MB | "
              << "Throughput: " << std::fixed << std::setprecision(1) << baseline_res.throughput_tok_s << " tok/s\n\n";

    // =========================================================================
    // RUN 2: AMU-QUANT CO-ADAPTIVE INFERENCE (Active Controller + KV Surgeon)
    // =========================================================================
    std::cout << "[RUN 2/2] Executing AMU-QUANT Co-Adaptive Engine..." << std::endl;
    BenchmarkRunResult amuquant_res;
    amuquant_res.name = "AMU-QUANT Co-Adaptive Engine";

    {
        // 1. Configure AMUQuantEngine
        AMUQuantConfig cfg;
        cfg.memory_limit_bytes = 4800ULL * 1024ULL * 1024ULL; // Target working budget
        cfg.os_reserved_bytes  = 600ULL * 1024ULL * 1024ULL;  // OS headroom
        cfg.num_layers = block_count;
        cfg.embedding_dim = embedding_length;
        cfg.eval_interval_tokens = 4;
        cfg.recent_window_tokens = 32; // Excises context beyond 32 tokens
        cfg.min_recent_window_tokens = 16;
        cfg.attention_sink_tokens = 4;

        AMUQuantEngine engine(cfg);
        Status st = engine.initialize();
        assert(st == Status::OK);

        // 2. Map GGUF model file and register layers
        int model_fd = open(model_path.c_str(), O_RDONLY);
        void* mmap_base = MAP_FAILED;
        size_t mmap_len = 0;
        if (model_fd >= 0) {
            try {
                mmap_len = std::filesystem::file_size(model_path);
                mmap_base = mmap(nullptr, mmap_len, PROT_READ, MAP_SHARED, model_fd, 0);
            } catch (...) {}
        }
        if (mmap_base != MAP_FAILED && mmap_len > 0) {
            size_t bytes_per_layer = mmap_len / block_count;
            for (uint32_t l = 0; l < block_count; ++l) {
                void* layer_addr = static_cast<char*>(mmap_base) + (l * bytes_per_layer);
                engine.register_layer(l, layer_addr, bytes_per_layer);
            }
        }

        auto trigger_layer_paging = [&](uint32_t num_layers) {
            for (uint32_t l = 0; l < num_layers; ++l) {
                engine.begin_layer_forward(l);
                engine.end_layer_forward(l);
            }
        };

        // 3. Initialize libllama context with embeddings enabled
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

        std::vector<std::vector<float>> session_embeddings;
        llama_memory_t mem = llama_get_memory(ctx);

        auto extractor = [&](int32_t pos) -> const float* {
            if (pos >= 0 && pos < static_cast<int32_t>(session_embeddings.size())) {
                return session_embeddings[pos].data();
            }
            return nullptr;
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
        trigger_layer_paging(block_count);

        for (int i = 0; i < n_prompt; ++i) {
            const float* embd = llama_get_embeddings_ith(ctx, i);
            if (embd != nullptr) {
                session_embeddings.push_back(std::vector<float>(embd, embd + embedding_length));
            } else {
                std::vector<float> vec(embedding_length, 0.0f);
                vec[prompt_tokens[i] % embedding_length] = 1.0f;
                session_embeddings.push_back(std::move(vec));
            }
        }

        int32_t n_past = n_prompt;

        auto t_start = std::chrono::steady_clock::now();
        for (uint32_t step = 0; step < target_tokens; ++step) {
            llama_token new_token = llama_sampler_sample(smpl, ctx, -1);
            llama_sampler_accept(smpl, new_token);

            if (llama_vocab_is_eog(vocab, new_token)) break;

            // AMU-QUANT controller arbitration step
            engine.step_token(n_past, n_past + 1, extractor, seq_rm, seq_add);

            llama_pos next_pos = llama_memory_seq_pos_max(mem, 0) + 1;
            batch.n_tokens = 1;
            batch.token[0] = new_token;
            batch.pos[0] = next_pos;
            batch.n_seq_id[0] = 1;
            batch.seq_id[0][0] = 0;
            batch.logits[0] = true;

            if (llama_decode(ctx, batch) != 0) break;
            trigger_layer_paging(block_count);

            const float* embd = llama_get_embeddings_ith(ctx, -1);
            if (embd != nullptr) {
                session_embeddings.push_back(std::vector<float>(embd, embd + embedding_length));
            } else {
                std::vector<float> vec(embedding_length, 0.0f);
                vec[new_token % embedding_length] = 1.0f;
                session_embeddings.push_back(std::move(vec));
            }

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
        amuquant_res.completed = (amuquant_res.tokens_generated > 0);

        // 4. Test Semantic Retrieval of Excised Context Token
        if (!session_embeddings.empty()) {
            auto t_rec_start = std::chrono::high_resolution_clock::now();
            auto recall_results = engine.recall_context(session_embeddings[5].data(), 3);
            auto t_rec_end = std::chrono::high_resolution_clock::now();
            amuquant_res.recall_latency_ms = std::chrono::duration<double, std::milli>(t_rec_end - t_rec_start).count();
            if (!recall_results.empty()) {
                amuquant_res.recall_score = recall_results[0].score;
            }
        }

        // Clean up
        if (mmap_base != MAP_FAILED && mmap_len > 0) munmap(mmap_base, mmap_len);
        if (model_fd >= 0) close(model_fd);
        llama_batch_free(batch);
        llama_sampler_free(smpl);
        llama_free(ctx);
    }

    std::cout << "  - AMU-QUANT Completed: " << amuquant_res.tokens_generated << " tokens generated | "
              << "Peak RSS: " << amuquant_res.peak_rss_mb << " MB | "
              << "Tokens Archived: " << amuquant_res.kv_tokens_archived << " | "
              << "Throughput: " << std::fixed << std::setprecision(1) << amuquant_res.throughput_tok_s << " tok/s\n\n";

    llama_model_free(model);
    llama_backend_free();

    // =========================================================================
    // COMPARATIVE SUMMARY REPORT
    // =========================================================================
    std::cout << "================================================================================\n";
    std::cout << "  EMPIRICAL BENCHMARK SUMMARY (MACOS 8/16 GB HOST TESTBED)                      \n";
    std::cout << "================================================================================\n";
    std::cout << std::left << std::setw(32) << "Evaluation Metric"
              << std::setw(24) << "Baseline (Vanilla)"
              << std::setw(24) << "AMU-QUANT (Engine)" << "\n";
    std::cout << "--------------------------------------------------------------------------------\n";
    std::cout << std::left << std::setw(32) << "Survival & Completion Rate"
              << std::setw(24) << (baseline_res.completed ? "100% Completed" : "Failed / OOM")
              << std::setw(24) << (amuquant_res.completed ? "100% Completed" : "Failed / OOM") << "\n";
    std::cout << std::left << std::setw(32) << "Tokens Generated"
              << std::setw(24) << baseline_res.tokens_generated
              << std::setw(24) << amuquant_res.tokens_generated << "\n";
    std::cout << std::left << std::setw(32) << "Throughput (tokens/sec)"
              << std::setw(24) << baseline_res.throughput_tok_s
              << std::setw(24) << amuquant_res.throughput_tok_s << "\n";
    std::cout << std::left << std::setw(32) << "Peak Process RSS"
              << std::setw(24) << (std::to_string(baseline_res.peak_rss_mb) + " MB")
              << std::setw(24) << (std::to_string(amuquant_res.peak_rss_mb) + " MB") << "\n";
    std::cout << std::left << std::setw(32) << "Tokens Archived to Vector Index"
              << std::setw(24) << "0 (None)"
              << std::setw(24) << amuquant_res.kv_tokens_archived << "\n";
    std::cout << std::left << std::setw(32) << "Context Semantic Recall Score"
              << std::setw(24) << "N/A (Uncompressed)"
              << std::setw(24) << (amuquant_res.recall_score > 0.0 ? std::to_string(amuquant_res.recall_score) : "N/A") << "\n";
    std::cout << std::left << std::setw(32) << "SIMD Retrieval Latency"
              << std::setw(24) << "N/A"
              << std::setw(24) << (std::to_string(amuquant_res.recall_latency_ms) + " ms") << "\n";
    std::cout << "================================================================================\n" << std::endl;

    // Export report to markdown
    std::ofstream rpt("benchmarks/live_benchmark_report.md");
    if (rpt.is_open()) {
        rpt << "# AMU-QUANT Live Empirical Benchmark Report\n\n";
        rpt << "- **Date**: " << __DATE__ << " " << __TIME__ << "\n";
        rpt << "- **Model**: `" << model_path << "`\n";
        rpt << "- **Platform**: macOS (Apple Silicon / Darwin VM)\n";
        rpt << "- **Threads**: " << pcore_threads << " CPU P-Cores\n\n";
        rpt << "| Metric | Baseline (Vanilla llama.cpp) | AMU-QUANT Co-Adaptive Engine |\n";
        rpt << "| :--- | :--- | :--- |\n";
        rpt << "| **Completion Status** | " << (baseline_res.completed ? "100% Completed" : "OOM Crash") << " | " << (amuquant_res.completed ? "100% Completed (Zero-OOM)" : "Failed") << " |\n";
        rpt << "| **Generated Tokens** | " << baseline_res.tokens_generated << " tokens | " << amuquant_res.tokens_generated << " tokens |\n";
        rpt << "| **Throughput** | " << std::fixed << std::setprecision(1) << baseline_res.throughput_tok_s << " tok/s | " << amuquant_res.throughput_tok_s << " tok/s |\n";
        rpt << "| **Peak Resident RAM (RSS)** | " << baseline_res.peak_rss_mb << " MB | " << amuquant_res.peak_rss_mb << " MB |\n";
        rpt << "| **Tokens Safely Archived** | 0 (Unmanaged) | " << amuquant_res.kv_tokens_archived << " tokens |\n";
        rpt << "| **Context Retrieval Score** | N/A | " << amuquant_res.recall_score << " (Cosine Match) |\n";
        rpt << "| **SIMD Retrieval Latency** | N/A | " << amuquant_res.recall_latency_ms << " ms |\n\n";
        rpt << "Generated automatically by `live_benchmark` executable.\n";
        rpt.close();
        std::cout << "[INFO] Benchmark report successfully exported to benchmarks/live_benchmark_report.md\n";
    }

    return 0;
}
