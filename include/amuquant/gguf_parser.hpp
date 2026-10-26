/**
 * @file gguf_parser.hpp
 * @brief Fast zero-copy metadata and tensor layout parser for GGUF model files.
 */

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
    std::vector<uint64_t> ne; // Elements per dimension
    uint32_t type{0};        // Quantization format enum (GGML_TYPE)
    uint64_t offset{0};      // Byte offset relative to tensor data buffer
    size_t size_bytes{0};    // Estimated tensor payload in bytes
    int32_t layer_idx{-1};   // Extracted transformer block index (-1 if non-layer)
};

struct GGUFModelInfo {
    std::string architecture;
    uint32_t block_count{0};
    uint32_t embedding_length{0};
    uint32_t context_length{0};
    uint32_t head_count{0};
    uint32_t head_count_kv{0};
    uint64_t tensor_count{0};
    uint64_t metadata_kv_count{0};
    std::vector<GGUFTensorInfo> tensors;
    
    // Mapping from layer index to tensor indices
    std::unordered_map<uint32_t, std::vector<size_t>> layer_tensors;
};

class GGUFParser {
public:
    GGUFParser() = default;
    ~GGUFParser() = default;

    /**
     * @brief Parses GGUF header and metadata key-value pairs without loading tensor payloads.
     */
    bool parse_file(const std::string& filepath, GGUFModelInfo& out_info);
};

} // namespace amuquant

#endif // AMUQUANT_GGUF_PARSER_HPP
