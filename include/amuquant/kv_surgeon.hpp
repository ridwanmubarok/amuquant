#ifndef AMUQUANT_KV_SURGEON_HPP
#define AMUQUANT_KV_SURGEON_HPP

#include "amuquant/types.hpp"
#include "amuquant/turbovec_ffi.h"

#include <cstdint>
#include <vector>
#include <functional>

namespace amuquant {

struct EvictionRange {
    int32_t p_start{-1};
    int32_t p_end{-1};
    bool valid() const { return p_start >= 0 && p_end > p_start; }
};

struct KVSurgeryStats {
    uint32_t active_kv_tokens{0};
    uint32_t archived_chunks{0};
    int32_t current_rope_shift{0};
    uint64_t kv_memory_bytes{0};
    uint64_t turbovec_memory_bytes{0};
};

using KVVectorExtractor = std::function<const float*(int32_t token_pos)>;
using KVSeqRmFunc = std::function<void(int32_t p0, int32_t p1)>;
using KVSeqAddFunc = std::function<void(int32_t p0, int32_t p1, int32_t delta)>;

class KVSurgeon {
public:
    KVSurgeon(TurbovecIndexHandle* turbovec_idx, uint32_t embedding_dim);
    ~KVSurgeon() = default;

    EvictionRange plan_eviction(uint32_t current_token_len, uint32_t sink_size, uint32_t recent_window) const;

    bool execute_eviction(
        const EvictionRange& range,
        const KVVectorExtractor& extractor,
        const KVSeqRmFunc& seq_rm,
        const KVSeqAddFunc& seq_add
    );

    KVSurgeryStats get_stats(uint32_t num_layers, uint32_t kv_heads, uint32_t head_dim) const;

    void reset();

private:
    TurbovecIndexHandle* turbovec_idx_{nullptr};
    uint32_t embedding_dim_{4096};
    uint32_t total_archived_tokens_{0};
    uint32_t active_tokens_{0};
    int32_t cumulative_rope_shift_{0};
    mutable int32_t last_evicted_pos_{-1};
};

} // namespace amuquant

#endif // AMUQUANT_KV_SURGEON_HPP
