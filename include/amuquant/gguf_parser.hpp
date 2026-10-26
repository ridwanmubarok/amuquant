#ifndef AMUQUANT_GGUF_PARSER_HPP
#define AMUQUANT_GGUF_PARSER_HPP

#include <string>
#include <vector>
#include <unordered_map>
#include <cstdint>
#include <cstddef>

namespace amuquant {

enum class GGUFType : uint32_t {
    UINT8 = 0,
    INT8 = 1,
    UINT16 = 2,
    INT16 = 3,
    UINT32 = 4,
    INT32 = 5,
    FLOAT32 = 6,
    BOOL = 7,
    STRING = 8,
    ARRAY = 9,
    UINT64 = 10,
    INT64 = 11,
    FLOAT64 = 12
};

struct GGUFTensorInfo {
    std::string name;
    uint32_t n_dims{0};
    std::vector<uint64_t> ne; // Jumlah elemen tiap dimensi
    uint32_t type{0};        // Format kuantisasi (Q4_K, Q8_0, F16, dll.)
    uint64_t offset{0};      // Offset relatif terhadap data buffer
    size_t size_bytes{0};    // Estimasi ukuran byte
    int32_t layer_idx{-1};   // Nomor layer hasil parsing nama tensor
};

struct GGUFModelInfo {
    std::string architecture;
    uint32_t block_count{0};     // Jumlah layer transformator
    uint32_t embedding_length{0}; // Dimensi tersembunyi
    uint32_t context_length{0};
    uint32_t head_count{0};
    uint32_t head_count_kv{0};
    uint64_t tensor_count{0};
    uint64_t metadata_kv_count{0};
    std::vector<GGUFTensorInfo> tensors;
    
    // Pemetaan layer ke daftar rentang tensor
    std::unordered_map<uint32_t, std::vector<size_t>> layer_tensors;
};

class GGUFParser {
public:
    GGUFParser() = default;
    ~GGUFParser() = default;

    /**
     * @brief Mem-parsing file model .gguf secara cepat dengan memetakan header ke memori.
     */
    bool parse_file(const std::string& filepath, GGUFModelInfo& out_info);
};

} // namespace amuquant

#endif // AMUQUANT_GGUF_PARSER_HPP
