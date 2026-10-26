#include "amuquant/engine.hpp"
#include "amuquant/gguf_parser.hpp"
#include "llama.h"

#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <chrono>
#include <sstream>
#include <fstream>
#include <cstring>
#include <thread>
#include <algorithm>
#include <filesystem>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>

#if defined(__APPLE__)
#include <sys/sysctl.h>
#endif

using namespace amuquant;

namespace {

// ANSI escape sequences for terminal interface styling
const char* ANSI_RESET   = "\033[0m";
const char* ANSI_BOLD    = "\033[1m";
const char* ANSI_CYAN    = "\033[36m";
const char* ANSI_GREEN   = "\033[32m";
const char* ANSI_YELLOW  = "\033[33m";
const char* ANSI_GRAY    = "\033[90m";

int get_optimal_pcore_threads() {
#if defined(__APPLE__)
    int pcores = 0;
    size_t len = sizeof(pcores);
    if (sysctlbyname("hw.perflevel0.logicalcpu", &pcores, &len, nullptr, 0) == 0 && pcores > 0) {
        return pcores;
    }
#endif
    unsigned int total = std::thread::hardware_concurrency();
    if (total >= 8) {
        return total / 2;
    }
    return std::max(1, static_cast<int>(total));
}

int calculate_optimal_gpu_layers(const std::string& model_path, const GGUFModelInfo& model_info, const MemoryStats& mem_stats) {
    // 1. Environment variable override (AMUQUANT_GPU_LAYERS)
    const char* env_val = std::getenv("AMUQUANT_GPU_LAYERS");
    if (env_val != nullptr) {
        try {
            int custom = std::stoi(env_val);
            if (custom >= 0) return custom;
        } catch (...) {}
    }

    // 2. Check model weight file size on disk
    uint64_t file_bytes = 0;
    try {
        if (std::filesystem::exists(model_path)) {
            file_bytes = std::filesystem::file_size(model_path);
        }
    } catch (...) {}

    // Models <= 3.5 GB (e.g. 8B Q2_K 2.96 GB, 3B Q4_K_M 2.0 GB):
    // Full GPU offloading (99 layers) without Metal host buffer exhaustion
    if (file_bytes > 0 && file_bytes <= 3500ULL * 1024ULL * 1024ULL) {
        return 99;
    }

    // 3. Compact models (<= 28 blocks or small hidden dim): full GPU offloading
    if (model_info.block_count <= 28 || model_info.embedding_length < 3000) {
        return 99;
    }

    // 4. Systems with >= 16 GB unified RAM: full GPU offloading
    if (mem_stats.total_ram_bytes >= 15ULL * 1024ULL * 1024ULL * 1024ULL) {
        return 99;
    }

    // 5. Large models (> 3.5 GB) on 8 GB unified host: hybrid CPU/GPU partitioning
    uint64_t avail_mb = mem_stats.available_ram_bytes / (1024 * 1024);
    
    // Reserve ~700 MB headroom for OS host stability
    uint64_t target_vram_mb = (avail_mb > 1400) ? (avail_mb - 700) : 2200;
    if (target_vram_mb > 2500) target_vram_mb = 2500;
    if (target_vram_mb < 2000) target_vram_mb = 2000;

    int est_layer_mb = (model_info.block_count > 0) ? static_cast<int>(4692 / model_info.block_count) : 146;
    int layers = static_cast<int>(target_vram_mb / est_layer_mb);

    if (layers > 16) layers = 16;
    if (layers < 14) layers = 14;

    return layers;
}

void print_banner(const GGUFModelInfo& info, const MemoryStats& mem, int gpu_layers, int pcores, bool flash_attn) {
    std::cout << ANSI_CYAN << ANSI_BOLD;
    std::cout << "================================================================================\n";
    std::cout << "  AMU-QUANT CHAT: Accelerated Neural LLM Inference (AMU-QUANT Controller)       \n";
    std::cout << "  Engine: libllama (Hybrid Metal/CPU) + TurboQuant Vector Index (turbovec FFI) \n";
    std::cout << "================================================================================\n";
    std::cout << ANSI_RESET;
    std::cout << ANSI_GRAY;
    std::cout << "  Model: " << info.architecture << " (" << info.block_count << " layers, " 
              << info.embedding_length << " dim) | Max Context: " << info.context_length << " tokens\n";
    std::cout << "  Hardware: Metal GPU: " << gpu_layers << "/" << info.block_count << " layers offloaded | "
              << "CPU P-Cores: " << pcores << " Threads | FlashAttn: " << (flash_attn ? "ENABLED" : "OFF") << "\n";
    std::cout << "  Memory: System RAM: " << (mem.total_ram_bytes / (1024 * 1024 * 1024)) << " GB | "
              << "Available: " << (mem.available_ram_bytes / (1024 * 1024)) << " MB | "
              << "Process RSS: " << (mem.process_rss_bytes / (1024 * 1024)) << " MB\n";
    std::cout << "  Commands: /stats, /regime, /clear, /help, /exit\n";
    std::cout << "--------------------------------------------------------------------------------\n";
    std::cout << ANSI_RESET << std::endl;
}

void print_stats(const AMUQuantEngine& engine, const GGUFModelInfo& info, int gpu_layers, int pcores) {
    EngineMetrics m = engine.get_metrics();
    MemoryStats s = const_cast<AMUQuantEngine&>(engine).get_telemetry()->sample();

    std::cout << ANSI_GRAY << "--------------------------------------------------------------------------------\n";
    std::cout << ANSI_BOLD << "  AMU-QUANT TELEMETRY & SYSTEM STATS REPORT\n" << ANSI_RESET;
    std::cout << "  - Active Regime: " << ANSI_YELLOW << regime_to_string(m.final_regime) << ANSI_RESET << "\n";
    std::cout << "  - Hardware Acceleration: Metal GPU (" << gpu_layers << "/" << info.block_count << " layers) + CPU (" << pcores << " P-Cores)\n";
    std::cout << "  - Process Resident Memory (RSS): " << (s.process_rss_bytes / (1024 * 1024)) << " MB\n";
    std::cout << "  - Available Physical RAM: " << (s.available_ram_bytes / (1024 * 1024)) << " MB\n";
    std::cout << "  - Major Page Faults (ru_majflt): " << s.major_page_faults << "\n";
    std::cout << "  - Total Tokens Generated: " << m.total_tokens_generated << "\n";
    std::cout << "  - Total KV Evictions to turbovec: " << m.total_kv_evictions << "\n";
    std::cout << "  - Average Inter-Token Latency: " << std::fixed << std::setprecision(2) << m.avg_inter_token_latency_ms << " ms/token\n";
    std::cout << "  - Model Weights Mmapped: " << info.block_count << " layers (" << info.tensor_count << " tensors)\n";
    std::cout << ANSI_GRAY << "--------------------------------------------------------------------------------\n" << ANSI_RESET;
}

} // namespace

int main(int argc, char** argv) {
    std::string model_path = "models/Meta-Llama-3-8B-Instruct-Q2_K.gguf";
    if (argc > 1) {
        model_path = argv[1];
    } else {
        std::ifstream f(model_path);
        if (!f.is_open()) {
            model_path = "models/Meta-Llama-3-8B-Instruct-Q3_K_S.gguf";
            std::ifstream f2(model_path);
            if (!f2.is_open()) {
                model_path = "models/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf";
                std::ifstream f3(model_path);
                if (!f3.is_open()) {
                    model_path = "models/qwen2.5-3b-instruct-q4_k_m.gguf";
                    std::ifstream f4(model_path);
                    if (!f4.is_open()) {
                        model_path = "models/qwen2.5-0.5b-instruct-q4_k_m.gguf";
                    }
                }
            }
        }
    }

    // 1. Parsing metadata model GGUF
    GGUFParser parser;
    GGUFModelInfo model_info;
    if (!parser.parse_file(model_path, model_info)) {
        std::cerr << "File GGUF tidak ditemukan di: " << model_path << "\n";
        return 1;
    }

    // 2. Configure AMU-QUANT engine runtime budget
    AMUQuantConfig cfg;
    cfg.num_layers = model_info.block_count > 0 ? model_info.block_count : 32;
    cfg.embedding_dim = model_info.embedding_length > 0 ? model_info.embedding_length : 4096;
    
    if (model_info.embedding_length >= 4096 || model_info.block_count >= 32) {
        // 8B class model profile (~4.9 GB weights on 8 GB unified host)
        cfg.memory_limit_bytes = 4800ULL * 1024ULL * 1024ULL; // 4.8 GB safe budget
        cfg.os_reserved_bytes  = 600ULL * 1024ULL * 1024ULL;  // 600 MB OS headroom
    } else if (model_info.block_count >= 28 || model_info.embedding_length >= 2048) {
        // 3B class model profile (~2.0 GB weights on unified host)
        cfg.memory_limit_bytes = 2400ULL * 1024ULL * 1024ULL; // 2.4 GB budget
        cfg.os_reserved_bytes  = 400ULL * 1024ULL * 1024ULL;  // 400 MB OS headroom
    } else {
        // Sub-1B class model profile
        cfg.memory_limit_bytes = 800ULL * 1024ULL * 1024ULL;
        cfg.os_reserved_bytes  = 200ULL * 1024ULL * 1024ULL;
    }
    cfg.eval_interval_tokens = 4;
    cfg.recent_window_tokens = 64;
    cfg.min_recent_window_tokens = 16;
    cfg.attention_sink_tokens = 4;

    AMUQuantEngine engine(cfg);
    if (engine.initialize() != Status::OK) {
        std::cerr << "Failed to initialize AMUQuantEngine.\n";
        return 1;
    }

    // 3. Initialize libllama backend with auto-calibrated hybrid acceleration
    llama_log_set([](enum ggml_log_level level, const char* text, void*) {
        if (level == GGML_LOG_LEVEL_ERROR) {
            std::cerr << "[LLAMA ERROR] " << text;
        }
    }, nullptr);

    llama_backend_init();

    SystemTelemetry telemetry;
    MemoryStats init_mem = telemetry.sample();

    int pcore_threads = get_optimal_pcore_threads();
    int gpu_layers = calculate_optimal_gpu_layers(model_path, model_info, init_mem);

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = gpu_layers;

    std::cout << "[INFO] Memuat model " << model_path << " (Akselerasi Metal GPU: " 
              << gpu_layers << " layers, CPU P-Cores: " << pcore_threads << " threads, FlashAttn: ON)...\n" << std::flush;

    llama_model* model = llama_model_load_from_file(model_path.c_str(), mparams);
    if (!model) {
        std::cerr << "Gagal memuat model dari file: " << model_path << "\n";
        return 1;
    }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = (model_info.embedding_length >= 4096) ? 2048 : 4096;
    cparams.n_batch = 512;
    cparams.n_threads = pcore_threads;
    cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    cparams.embeddings = true;

    llama_context* ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        std::cerr << "Gagal membuat llama_context.\n";
        llama_model_free(model);
        return 1;
    }

    // Map GGUF model file to register physical transformer layer address ranges with LayerManager
    void* mmap_base = MAP_FAILED;
    size_t mmap_len = 0;
    int model_fd = open(model_path.c_str(), O_RDONLY);
    if (model_fd >= 0) {
        try {
            mmap_len = std::filesystem::file_size(model_path);
            mmap_base = mmap(nullptr, mmap_len, PROT_READ, MAP_SHARED, model_fd, 0);
        } catch (...) {}
    }

    uint32_t total_model_layers = model_info.block_count > 0 ? model_info.block_count : 32;
    if (mmap_base != MAP_FAILED && mmap_len > 0) {
        size_t bytes_per_layer = mmap_len / total_model_layers;
        for (uint32_t l = 0; l < total_model_layers; ++l) {
            void* layer_addr = static_cast<char*>(mmap_base) + (l * bytes_per_layer);
            engine.register_layer(l, layer_addr, bytes_per_layer);
        }
        std::cout << "[INFO] LayerManager: Registered " << total_model_layers 
                  << " physical transformer layers with mmap virtual memory paging.\n";
    }

    auto trigger_layer_paging = [&](uint32_t num_layers) {
        for (uint32_t l = 0; l < num_layers; ++l) {
            engine.begin_layer_forward(l);
            engine.end_layer_forward(l);
        }
    };

    const llama_vocab* vocab = llama_model_get_vocab(model);

    // 4. Inisialisasi Sampler Chain
    llama_sampler* smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(smpl, llama_sampler_init_top_k(40));
    llama_sampler_chain_add(smpl, llama_sampler_init_top_p(0.9f, 1));
    llama_sampler_chain_add(smpl, llama_sampler_init_temp(0.7f));
    llama_sampler_chain_add(smpl, llama_sampler_init_dist(42));

    print_banner(model_info, init_mem, gpu_layers, pcore_threads, true);

    // Context tracking state
    int32_t n_past = 0;
    std::vector<std::vector<float>> session_embeddings;
    llama_memory_t mem = llama_get_memory(ctx);

    auto extractor = [&](int32_t pos) -> const float* {
        if (pos >= 0 && pos < static_cast<int32_t>(session_embeddings.size())) {
            return session_embeddings[pos].data();
        }
        return nullptr;
    };

    int32_t total_rope_shift = 0;

    auto seq_rm = [&](int32_t p0, int32_t p1) {
        llama_memory_seq_rm(mem, 0, p0, p1);
    };

    auto seq_add = [&](int32_t p0, int32_t p1, int32_t delta) {
        llama_memory_seq_add(mem, 0, p0, p1, delta);
        total_rope_shift += delta;
    };

    llama_batch batch = llama_batch_init(512, 0, 1);
    bool is_llama = (model_info.architecture == "llama");

    // Kirim system message awal sesuai format model
    std::string system_msg;
    if (is_llama) {
        system_msg = "<|start_header_id|>system<|end_header_id|>\n\nYou are a helpful and intelligent AI assistant powered by Llama-3 and AMU-QUANT. You assist in both Indonesian and English.<|eot_id|>";
    } else {
        system_msg = "<|im_start|>system\nYou are a helpful and intelligent AI assistant powered by Qwen2.5 and AMU-QUANT. You can assist with general conversation, coding, and technical explanations in both Indonesian and English.<|im_end|>\n";
    }
    std::vector<llama_token> sys_tokens(system_msg.size() + 16);
    int n_sys = llama_tokenize(vocab, system_msg.c_str(), system_msg.size(), sys_tokens.data(), sys_tokens.size(), true, true);
    if (n_sys > 0) {
        batch.n_tokens = 0;
        for (int i = 0; i < n_sys; ++i) {
            batch.token[batch.n_tokens] = sys_tokens[i];
            batch.pos[batch.n_tokens] = n_past + i;
            batch.n_seq_id[batch.n_tokens] = 1;
            batch.seq_id[batch.n_tokens][0] = 0;
            batch.logits[batch.n_tokens] = false;
            batch.n_tokens++;
        }
        if (llama_decode(ctx, batch) == 0) {
            trigger_layer_paging(total_model_layers);
            for (int i = 0; i < n_sys; ++i) {
                const float* embd = llama_get_embeddings_ith(ctx, i);
                if (embd != nullptr) {
                    session_embeddings.push_back(std::vector<float>(embd, embd + model_info.embedding_length));
                } else {
                    std::vector<float> vec(model_info.embedding_length, 0.0f);
                    vec[sys_tokens[i] % model_info.embedding_length] = 1.0f;
                    session_embeddings.push_back(std::move(vec));
                }
            }
            n_past += n_sys;
        }
    }

    while (true) {
        std::cout << ANSI_BOLD << ANSI_GREEN << "user> " << ANSI_RESET;
        std::string user_input;
        if (!std::getline(std::cin, user_input)) {
            break;
        }

        if (user_input.empty()) {
            continue;
        }

        // Perintah CLI khusus
        if (user_input == "/exit" || user_input == "/quit") {
            std::cout << ANSI_GRAY << "Mengakhiri sesi chat AMU-QUANT.\n" << ANSI_RESET;
            break;
        } else if (user_input == "/stats" || user_input == "/telemetry") {
            print_stats(engine, model_info, gpu_layers, pcore_threads);
            continue;
        } else if (user_input == "/regime") {
            auto reg = engine.get_controller()->get_current_regime();
            double phi = engine.get_controller()->get_last_phi();
            std::cout << ANSI_YELLOW << "Rejim Aktif: " << regime_to_string(reg) 
                      << " (Skor Tekanan Φ_t = " << std::fixed << std::setprecision(3) << phi << ")\n" << ANSI_RESET;
            continue;
        } else if (user_input == "/clear") {
            llama_memory_clear(mem, true);
            n_past = 0;
            total_rope_shift = 0;
            session_embeddings.clear();
            engine.get_controller()->reset();
            std::cout << ANSI_GRAY << "Riwayat percakapan dan KV-cache telah dibersihkan.\n" << ANSI_RESET;
            continue;
        } else if (user_input == "/help") {
            std::cout << ANSI_GRAY << "Perintah yang tersedia:\n"
                      << "  /stats    - Menampilkan telemetri memori real-time dan metrik AMU-QUANT\n"
                      << "  /regime   - Menampilkan status rejim adaptasi dan skor prediktif Φ_t\n"
                      << "  /clear    - Mengosongkan riwayat konteks dan mereset KV-Cache\n"
                      << "  /help     - Menampilkan bantuan ini\n"
                      << "  /exit     - Keluar dari program\n" << ANSI_RESET;
            continue;
        }

        // 5. Format prompt according to model architecture (Llama-3 or Qwen ChatML)
        std::string user_prompt;
        if (is_llama) {
            user_prompt = "<|start_header_id|>user<|end_header_id|>\n\n" + user_input + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n";
        } else {
            user_prompt = "<|im_start|>user\n" + user_input + "<|im_end|>\n<|im_start|>assistant\n";
        }
        std::vector<llama_token> prompt_tokens(user_prompt.size() + 16);
        int n_tokens = llama_tokenize(vocab, user_prompt.c_str(), user_prompt.size(), prompt_tokens.data(), prompt_tokens.size(), false, true);

        if (n_tokens <= 0) {
            continue;
        }

        // Decode batch prompt input
        llama_pos cur_max_pos = llama_memory_seq_pos_max(mem, 0);
        llama_pos prompt_start_pos = (cur_max_pos >= 0) ? (cur_max_pos + 1) : 0;

        batch.n_tokens = 0;
        for (int i = 0; i < n_tokens; ++i) {
            batch.token[batch.n_tokens] = prompt_tokens[i];
            batch.pos[batch.n_tokens] = prompt_start_pos + i;
            batch.n_seq_id[batch.n_tokens] = 1;
            batch.seq_id[batch.n_tokens][0] = 0;
            batch.logits[batch.n_tokens] = (i == n_tokens - 1);
            batch.n_tokens++;
        }

        if (llama_decode(ctx, batch) != 0) {
            std::cerr << "Failed to decode prompt batch.\n";
            continue;
        }
        trigger_layer_paging(total_model_layers);
        for (int i = 0; i < n_tokens; ++i) {
            const float* embd = llama_get_embeddings_ith(ctx, i);
            if (embd != nullptr) {
                session_embeddings.push_back(std::vector<float>(embd, embd + model_info.embedding_length));
            } else {
                std::vector<float> vec(model_info.embedding_length, 0.0f);
                vec[prompt_tokens[i] % model_info.embedding_length] = 1.0f;
                session_embeddings.push_back(std::move(vec));
            }
        }
        n_past += n_tokens;

        // 6. Autoregressive streaming token generation loop
        std::cout << ANSI_BOLD << ANSI_CYAN << "assistant> " << ANSI_RESET;

        auto t_start = std::chrono::steady_clock::now();
        uint32_t generated_tokens = 0;

        for (int step = 0; step < 512; ++step) {
            // Sample next token from logits distribution
            llama_token new_token_id = llama_sampler_sample(smpl, ctx, -1);
            llama_sampler_accept(smpl, new_token_id);

            // Check end-of-generation token
            if (llama_vocab_is_eog(vocab, new_token_id)) {
                break;
            }

            // Stream token piece to standard output
            char piece_buf[128];
            int piece_len = llama_token_to_piece(vocab, new_token_id, piece_buf, sizeof(piece_buf), 0, true);
            if (piece_len > 0) {
                std::cout << std::string(piece_buf, piece_len) << std::flush;
            }

            // Trigger AMU-QUANT controller hook to arbitrate memory pressure and KV eviction
            engine.step_token(n_past, n_past + 1, extractor, seq_rm, seq_add);

            // Fetch highest valid sequence position from physical KV memory
            llama_pos next_pos = llama_memory_seq_pos_max(mem, 0) + 1;

            // Prepare batch for next single-token decode step
            batch.n_tokens = 1;
            batch.token[0] = new_token_id;
            batch.pos[0] = next_pos;
            batch.n_seq_id[0] = 1;
            batch.seq_id[0][0] = 0;
            batch.logits[0] = true;

            if (llama_decode(ctx, batch) != 0) {
                break;
            }
            trigger_layer_paging(total_model_layers);

            const float* embd = llama_get_embeddings_ith(ctx, -1);
            if (embd != nullptr) {
                session_embeddings.push_back(std::vector<float>(embd, embd + model_info.embedding_length));
            } else {
                std::vector<float> vec(model_info.embedding_length, 0.0f);
                vec[new_token_id % model_info.embedding_length] = 1.0f;
                session_embeddings.push_back(std::move(vec));
            }

            n_past++;
            generated_tokens++;
        }
        std::cout << "\n";

        auto t_end = std::chrono::steady_clock::now();
        double elapsed_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
        double tok_per_sec = (generated_tokens > 0 && elapsed_ms > 0) ? (generated_tokens / (elapsed_ms / 1000.0)) : 0.0;

        // Status HUD bar dengan rincian akselerasi hardware
        MemoryStats cur_mem = engine.get_telemetry()->sample();
        Regime cur_reg = engine.get_controller()->get_current_regime();

        std::cout << ANSI_GRAY << "[" 
                  << regime_to_string(cur_reg) << " | "
                  << "Metal: " << gpu_layers << "/" << model_info.block_count << "L | "
                  << "P-Cores: " << pcore_threads << "T | "
                  << "FlashAttn: ON | "
                  << "RSS: " << (cur_mem.process_rss_bytes / (1024 * 1024)) << " MB | "
                  << "Throughput: " << std::fixed << std::setprecision(1) << tok_per_sec << " tok/s | "
                  << "Context: " << n_past << " tokens]"
                  << ANSI_RESET << "\n\n";
    }

    // Bersihkan sumber daya
    if (mmap_base != MAP_FAILED && mmap_len > 0) {
        munmap(mmap_base, mmap_len);
    }
    if (model_fd >= 0) {
        close(model_fd);
    }

    llama_batch_free(batch);
    llama_sampler_free(smpl);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();

    return 0;
}
