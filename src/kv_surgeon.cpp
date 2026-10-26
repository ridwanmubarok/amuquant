#include "amuquant/kv_surgeon.hpp"

#include <algorithm>

namespace amuquant {

KVSurgeon::KVSurgeon(TurbovecIndexHandle* turbovec_idx, uint32_t embedding_dim)
    : turbovec_idx_(turbovec_idx), embedding_dim_(embedding_dim) {}

EvictionRange KVSurgeon::plan_eviction(uint32_t current_token_len, uint32_t sink_size, uint32_t recent_window) const {
    if (current_token_len <= sink_size + recent_window) {
        return {-1, -1};
    }

    int32_t p_start = (last_evicted_pos_ >= 0) ? (last_evicted_pos_ + 1) : static_cast<int32_t>(sink_size);
    int32_t p_end = static_cast<int32_t>(current_token_len - recent_window);

    if (p_end > p_start) {
        return {p_start, p_end};
    }
    return {-1, -1};
}

bool KVSurgeon::execute_eviction(
    const EvictionRange& range,
    const KVVectorExtractor& extractor,
    const KVSeqRmFunc& seq_rm,
    const KVSeqAddFunc& seq_add
) {
    if (!range.valid() || turbovec_idx_ == nullptr) {
        return false;
    }

    // 1. Ekstraksi vektor token tengah dan simpan ke turbovec via zero-copy FFI
    for (int32_t pos = range.p_start; pos < range.p_end; ++pos) {
        const float* vec_ptr = extractor(pos);
        if (vec_ptr != nullptr) {
            turbovec_insert_zero_copy(turbovec_idx_, static_cast<uint64_t>(pos), vec_ptr, embedding_dim_);
            total_archived_tokens_++;
        }
    }

    // 2. Pemotongan sequence pada KV-Cache fisik
    if (seq_rm) {
        seq_rm(range.p_start, range.p_end);
    }

    // 3. Pergeseran offset RoPE untuk menutup celah posisi
    int32_t shift_delta = -(range.p_end - range.p_start);
    if (seq_add) {
        seq_add(range.p_end, -1, shift_delta);
    }

    last_evicted_pos_ = range.p_end - 1;
    cumulative_rope_shift_ += shift_delta;
    return true;
}

KVSurgeryStats KVSurgeon::get_stats(uint32_t num_layers, uint32_t kv_heads, uint32_t head_dim) const {
    KVSurgeryStats stats;
    stats.archived_chunks = total_archived_tokens_;
    stats.current_rope_shift = cumulative_rope_shift_;

    // Perhitungan ukuran KV-cache fisik (format FP16 = 2 byte per elemen)
    // Memory per token = 2 * num_layers * kv_heads * head_dim * sizeof(fp16)
    size_t bytes_per_token = 2ULL * num_layers * kv_heads * head_dim * sizeof(uint16_t);
    stats.kv_memory_bytes = active_tokens_ * bytes_per_token;

    if (turbovec_idx_ != nullptr) {
        size_t tb_bytes = 0;
        turbovec_get_memory_usage(turbovec_idx_, &tb_bytes);
        stats.turbovec_memory_bytes = tb_bytes;
    }

    return stats;
}

void KVSurgeon::reset() {
    total_archived_tokens_ = 0;
    active_tokens_ = 0;
    cumulative_rope_shift_ = 0;
    last_evicted_pos_ = -1;
}

} // namespace amuquant
