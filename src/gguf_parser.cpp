#include "amuquant/gguf_parser.hpp"

#include <fstream>
#include <iostream>
#include <cstring>
#include <regex>

namespace amuquant {

namespace {

std::string read_gguf_string(std::ifstream& f) {
    uint64_t len = 0;
    f.read(reinterpret_cast<char*>(&len), sizeof(len));
    if (!f || len > 1024 * 1024) return "";
    std::string s(len, '\0');
    f.read(&s[0], len);
    return s;
}

void skip_gguf_value(std::ifstream& f, uint32_t type) {
    switch (static_cast<GGUFType>(type)) {
        case GGUFType::UINT8:
        case GGUFType::INT8:
        case GGUFType::BOOL:
            f.seekg(1, std::ios::cur);
            break;
        case GGUFType::UINT16:
        case GGUFType::INT16:
            f.seekg(2, std::ios::cur);
            break;
        case GGUFType::UINT32:
        case GGUFType::INT32:
        case GGUFType::FLOAT32:
            f.seekg(4, std::ios::cur);
            break;
        case GGUFType::UINT64:
        case GGUFType::INT64:
        case GGUFType::FLOAT64:
            f.seekg(8, std::ios::cur);
            break;
        case GGUFType::STRING: {
            uint64_t len = 0;
            f.read(reinterpret_cast<char*>(&len), sizeof(len));
            f.seekg(len, std::ios::cur);
            break;
        }
        case GGUFType::ARRAY: {
            uint32_t elem_type = 0;
            uint64_t count = 0;
            f.read(reinterpret_cast<char*>(&elem_type), sizeof(elem_type));
            f.read(reinterpret_cast<char*>(&count), sizeof(count));
            for (uint64_t i = 0; i < count; ++i) {
                skip_gguf_value(f, elem_type);
            }
            break;
        }
    }
}

} // namespace

bool GGUFParser::parse_file(const std::string& filepath, GGUFModelInfo& out_info) {
    std::ifstream f(filepath, std::ios::binary);
    if (!f.is_open()) {
        return false;
    }

    // 1. Periksa Magic Header "GGUF" (0x46554747)
    char magic[4] = {0};
    f.read(magic, 4);
    if (std::memcmp(magic, "GGUF", 4) != 0) {
        return false;
    }

    uint32_t version = 0;
    f.read(reinterpret_cast<char*>(&version), sizeof(version));
    if (version < 2) {
        return false;
    }

    f.read(reinterpret_cast<char*>(&out_info.tensor_count), sizeof(out_info.tensor_count));
    f.read(reinterpret_cast<char*>(&out_info.metadata_kv_count), sizeof(out_info.metadata_kv_count));

    // 2. Baca Pasangan Metadata Key-Value
    for (uint64_t i = 0; i < out_info.metadata_kv_count; ++i) {
        std::string key = read_gguf_string(f);
        uint32_t val_type = 0;
        f.read(reinterpret_cast<char*>(&val_type), sizeof(val_type));

        if (key == "general.architecture" && val_type == static_cast<uint32_t>(GGUFType::STRING)) {
            out_info.architecture = read_gguf_string(f);
        } else if (key.find(".block_count") != std::string::npos && val_type == static_cast<uint32_t>(GGUFType::UINT32)) {
            f.read(reinterpret_cast<char*>(&out_info.block_count), 4);
        } else if (key.find(".embedding_length") != std::string::npos && val_type == static_cast<uint32_t>(GGUFType::UINT32)) {
            f.read(reinterpret_cast<char*>(&out_info.embedding_length), 4);
        } else if (key.find(".context_length") != std::string::npos && val_type == static_cast<uint32_t>(GGUFType::UINT32)) {
            f.read(reinterpret_cast<char*>(&out_info.context_length), 4);
        } else if (key.find(".attention.head_count") != std::string::npos && key.find("kv") == std::string::npos && val_type == static_cast<uint32_t>(GGUFType::UINT32)) {
            f.read(reinterpret_cast<char*>(&out_info.head_count), 4);
        } else if (key.find(".attention.head_count_kv") != std::string::npos && val_type == static_cast<uint32_t>(GGUFType::UINT32)) {
            f.read(reinterpret_cast<char*>(&out_info.head_count_kv), 4);
        } else if (key.find(".attention.key_length") != std::string::npos && val_type == static_cast<uint32_t>(GGUFType::UINT32)) {
            f.read(reinterpret_cast<char*>(&out_info.head_dim), 4);
        } else {
            skip_gguf_value(f, val_type);
        }
    }

    if (out_info.head_dim == 0 && out_info.head_count > 0 && out_info.embedding_length > 0) {
        out_info.head_dim = out_info.embedding_length / out_info.head_count;
    }

    // 3. Baca Metadata Tensor Info
    out_info.tensors.reserve(out_info.tensor_count);
    std::regex block_regex(R"(blk\.(\d+)\.)");

    for (uint64_t i = 0; i < out_info.tensor_count; ++i) {
        GGUFTensorInfo t;
        t.name = read_gguf_string(f);
        f.read(reinterpret_cast<char*>(&t.n_dims), sizeof(t.n_dims));
        t.ne.resize(t.n_dims);
        for (uint32_t d = 0; d < t.n_dims; ++d) {
            f.read(reinterpret_cast<char*>(&t.ne[d]), sizeof(uint64_t));
        }
        f.read(reinterpret_cast<char*>(&t.type), sizeof(t.type));
        f.read(reinterpret_cast<char*>(&t.offset), sizeof(t.offset));

        // Extract transformer block layer index from tensor identifier (e.g. "blk.0.attn_q.weight")
        std::smatch match;
        if (std::regex_search(t.name, match, block_regex) && match.size() > 1) {
            t.layer_idx = std::stoi(match[1].str());
            out_info.layer_tensors[static_cast<uint32_t>(t.layer_idx)].push_back(i);
        }

        out_info.tensors.push_back(std::move(t));
    }

    return true;
}

} // namespace amuquant
