#include "amuquant/kv_surgeon.hpp"

#include <algorithm>
#include <vector>

namespace amuquant {

KVSurgeon::KVSurgeon() = default;

EvictionRange KVSurgeon::plan_eviction(uint32_t current_token_len, uint32_t sink_size, uint32_t recent_window, uint32_t min_batch) const {
    if (current_token_len <= sink_size + recent_window) {
        return {-1, -1};
    }

    int32_t p_start = static_cast<int32_t>(sink_size);
    int32_t p_end = static_cast<int32_t>(current_token_len - recent_window);

    if (p_end > p_start && static_cast<uint32_t>(p_end - p_start) >= min_batch) {
        return {p_start, p_end};
    }
    return {-1, -1};
}

bool KVSurgeon::execute_eviction(
    const EvictionRange& range,
    const KVSeqRmFunc& seq_rm,
    const KVSeqAddFunc& seq_add
) {
    if (!range.valid()) {
        return false;
    }

    // Auto-populate token map if caller hasn't explicitly initialized it
    if (token_map_.physical_size() < static_cast<size_t>(range.p_end)) {
        size_t start = token_map_.physical_size();
        for (size_t i = start; i < static_cast<size_t>(range.p_end); ++i) {
            token_map_.append_token(static_cast<uint64_t>(i));
        }
    }

    size_t count = static_cast<size_t>(range.p_end - range.p_start);
    std::vector<uint64_t> logical_ids;
    logical_ids.reserve(count);

    for (int32_t pos = range.p_start; pos < range.p_end; ++pos) {
        uint64_t logical_id = token_map_.get_logical_id(pos);
        if (logical_id != UINT64_MAX) {
            logical_ids.push_back(logical_id);
        }
    }

    // Phase 1: Prune physical KV sequence slot range [p_start, p_end)
    bool rm_ok = false;
    try {
        if (seq_rm) {
            rm_ok = seq_rm(range.p_start, range.p_end);
        }
    } catch (...) {
        rm_ok = false;
    }

    if (!rm_ok) {
        return false;
    }

    // Phase 2: Apply RoPE positional shift to close the sequence gap:
    // All tokens at physical positions >= p_end are shifted left by (p_end - p_start)
    int32_t shift_delta = -(range.p_end - range.p_start);
    bool add_ok = true;
    try {
        if (seq_add) {
            add_ok = seq_add(range.p_end, -1, shift_delta);
        }
    } catch (...) {
        add_ok = false;
    }

    if (!add_ok) {
        return false;
    }

    // Phase 3: Update physical-to-logical translation table
    token_map_.prune_range(range.p_start, range.p_end);

    if (eviction_cb_) {
        eviction_cb_(logical_ids);
    }

    cumulative_rope_shift_ += shift_delta;
    total_pruned_tokens_ += static_cast<uint32_t>(count);
    return true;
}

KVSurgeryStats KVSurgeon::get_stats(uint32_t num_layers, uint32_t kv_heads, uint32_t head_dim, uint32_t bytes_per_element) const {
    KVSurgeryStats stats;
    stats.active_kv_tokens = static_cast<uint32_t>(token_map_.physical_size());
    stats.total_pruned_tokens = total_pruned_tokens_;
    stats.archived_chunks = total_pruned_tokens_;
    stats.current_rope_shift = cumulative_rope_shift_;

    // Physical KV-cache memory: 2 * num_layers * kv_heads * head_dim * bytes_per_element
    size_t bytes_per_token = 2ULL * num_layers * kv_heads * head_dim * bytes_per_element;
    stats.kv_memory_bytes = stats.active_kv_tokens * bytes_per_token;

    return stats;
}

void KVSurgeon::reset() {
    active_tokens_ = 0;
    total_pruned_tokens_ = 0;
    cumulative_rope_shift_ = 0;
    token_map_.reset();
}

} // namespace amuquant
