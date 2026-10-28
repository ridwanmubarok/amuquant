/**
 * @file kv_surgeon.hpp
 * @brief Dynamic KV-cache sliding window eviction surgery and RoPE positional realignment.
 */

#ifndef AMUQUANT_KV_SURGEON_HPP
#define AMUQUANT_KV_SURGEON_HPP

#include "amuquant/types.hpp"

#include <cstdint>
#include <vector>
#include <functional>
#include <memory>

namespace amuquant {

struct EvictionRange {
    int32_t p_start{-1};
    int32_t p_end{-1};
    bool valid() const { return p_start >= 0 && p_end > p_start; }
};

struct KVSurgeryStats {
    uint32_t active_kv_tokens{0};
    uint32_t total_pruned_tokens{0};
    uint32_t archived_chunks{0};
    int32_t current_rope_shift{0};
    uint64_t kv_memory_bytes{0};
};

/**
 * @brief Thread-safe mapping between physical KV-cache slots and immutable logical token IDs.
 * Decouples shifting physical positions from logical context.
 */
class TokenIndexMap {
public:
    TokenIndexMap() = default;

    void init(uint32_t prompt_length) {
        physical_to_logical_.clear();
        physical_to_logical_.reserve(prompt_length);
        for (uint32_t i = 0; i < prompt_length; ++i) {
            physical_to_logical_.push_back(i);
        }
    }

    void append_token(uint64_t logical_id) {
        physical_to_logical_.push_back(logical_id);
    }

    void prune_range(int32_t p0, int32_t p1) {
        if (p0 >= 0 && p1 > p0 && static_cast<size_t>(p1) <= physical_to_logical_.size()) {
            physical_to_logical_.erase(
                physical_to_logical_.begin() + p0,
                physical_to_logical_.begin() + p1
            );
        }
    }

    uint64_t get_logical_id(int32_t physical_pos) const {
        if (physical_pos >= 0 && static_cast<size_t>(physical_pos) < physical_to_logical_.size()) {
            return physical_to_logical_[physical_pos];
        }
        return UINT64_MAX;
    }

    int32_t get_physical_pos(uint64_t logical_id) const {
        for (size_t i = 0; i < physical_to_logical_.size(); ++i) {
            if (physical_to_logical_[i] == logical_id) {
                return static_cast<int32_t>(i);
            }
        }
        return -1;
    }

    size_t physical_size() const {
        return physical_to_logical_.size();
    }

    bool contains_logical(uint64_t logical_id) const {
        return get_physical_pos(logical_id) >= 0;
    }

    const std::vector<uint64_t>& get_raw_map() const {
        return physical_to_logical_;
    }

    void reset() {
        physical_to_logical_.clear();
    }

private:
    std::vector<uint64_t> physical_to_logical_;
};

using KVSeqRmFunc = std::function<bool(int32_t p0, int32_t p1)>;
using KVSeqAddFunc = std::function<bool(int32_t p0, int32_t p1, int32_t delta)>;
using KVEvictionCallback = std::function<void(const std::vector<uint64_t>& evicted_logical_ids)>;

class KVSurgeon {
public:
    KVSurgeon();
    ~KVSurgeon() = default;

    /**
     * @brief Computes middle-token eviction bounds preserving initial sink tokens and recent window.
     */
    EvictionRange plan_eviction(uint32_t current_token_len, uint32_t sink_size, uint32_t recent_window, uint32_t min_batch = 1) const;

    /**
     * @brief Prunes middle KV cache sequence and applies RoPE positional realignment.
     * Transactional execution: validates parameters and executes sequence removal and realignment.
     * Updates physical-to-logical token mappings atomically.
     */
    bool execute_eviction(
        const EvictionRange& range,
        const KVSeqRmFunc& seq_rm,
        const KVSeqAddFunc& seq_add
    );

    /**
     * @brief Returns active KV memory metrics.
     */
    KVSurgeryStats get_stats(uint32_t num_layers, uint32_t kv_heads, uint32_t head_dim, uint32_t bytes_per_element = 2) const;

    TokenIndexMap& get_token_map() { return token_map_; }
    const TokenIndexMap& get_token_map() const { return token_map_; }

    void set_eviction_callback(KVEvictionCallback cb) { eviction_cb_ = std::move(cb); }

    void reset();

private:
    uint32_t active_tokens_{0};
    uint32_t total_pruned_tokens_{0};
    int32_t cumulative_rope_shift_{0};
    TokenIndexMap token_map_;
    KVEvictionCallback eviction_cb_;
};

} // namespace amuquant

#endif // AMUQUANT_KV_SURGEON_HPP
