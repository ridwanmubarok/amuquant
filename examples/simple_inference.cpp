#include "amuquant/engine.hpp"
#include <iostream>
#include <vector>

int main() {
    std::cout << "AMU-QUANT: Minimal C++ API Integration Example\n";

    // 1. Inisialisasi konfigurasi engine AMU-QUANT
    amuquant::AMUQuantConfig cfg;
    cfg.memory_limit_bytes = 2048ULL * 1024ULL * 1024ULL; // 2 GB limit
    cfg.os_reserved_bytes  = 200ULL * 1024ULL * 1024ULL;  // 200 MB OS headroom
    cfg.num_layers = 32;
    cfg.embedding_dim = 4096;
    cfg.eval_interval_tokens = 4;
    cfg.recent_window_tokens = 32;
    cfg.min_recent_window_tokens = 12;
    cfg.attention_sink_tokens = 4;

    // 2. Buat instance engine
    amuquant::AMUQuantEngine engine(cfg);
    if (engine.initialize() != amuquant::Status::OK) {
        std::cerr << "Inisialisasi AMUQuantEngine gagal.\n";
        return 1;
    }

    std::cout << "AMU-QUANT Engine terinisialisasi dengan sukses.\n";
    std::cout << "Siap digunakan untuk inferensi terakselerasi dengan kompresi memori turbovec.\n";

    return 0;
}
