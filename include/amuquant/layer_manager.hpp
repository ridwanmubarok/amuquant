/**
 * @file layer_manager.hpp
 * @brief Layer-wise virtual memory paging and kernel madvise management.
 */

#ifndef AMUQUANT_LAYER_MANAGER_HPP
#define AMUQUANT_LAYER_MANAGER_HPP

#include "amuquant/types.hpp"
#include <vector>
#include <cstdint>
#include <cstddef>
#include <unordered_set>
#include <mutex>

namespace amuquant {

struct LayerDescriptor {
    uint32_t layer_idx{0};
    void* addr_start{nullptr};
    size_t byte_len{0};
    bool is_resident{false};
};

class LayerManager {
public:
    LayerManager();
    ~LayerManager() = default;

    /**
     * @brief Registers the virtual address range of a model layer's mmap weights.
     */
    void register_layer(uint32_t layer_idx, void* addr, size_t size_bytes);

    /**
     * @brief Issues prefetch advisories (POSIX_MADV_WILLNEED) for upcoming layers.
     */
    void on_layer_forward_begin(uint32_t layer_idx, uint32_t prefetch_depth);

    /**
     * @brief Post-forward hook to evict cold layer pages via POSIX_MADV_DONTNEED.
     */
    void on_layer_forward_end(uint32_t layer_idx, bool aggressive_evict);

    /**
     * @brief Returns the estimated resident weight footprint across all registered layers.
     * Tracks AMU-QUANT's logical prefetch/eviction intent state (is_resident).
     */
    uint64_t get_resident_weight_bytes() const;

    /**
     * @brief Queries actual resident bytes for a layer directly from kernel page tables via mincore(2).
     *
     * Disambiguates internal advisory tracking status (is_resident) from physical RAM residency.
     * Note: posix_madvise is an OS advisory; mincore(2) verifies whether page frames are physical core-resident.
     *
     * @param layer_idx Index of the layer to query.
     * @return Physical resident memory in bytes, or 0 if unmapped / failed.
     */
    uint64_t get_actual_resident_bytes(uint32_t layer_idx) const;

    /**
     * @brief Queries total actual resident bytes across all registered layers via mincore(2).
     *
     * Aggregates physical kernel page residency across all layer virtual memory spans.
     *
     * @return Total physical resident memory in bytes.
     */
    uint64_t get_total_actual_resident_bytes() const;

    size_t num_layers() const;
    void reset();

private:
    std::vector<LayerDescriptor> layers_;
    std::unordered_set<uint32_t> resident_layers_;
    mutable std::mutex mutex_;

    void evict_layer_pages(LayerDescriptor& layer);
    void prefetch_layer_pages(LayerDescriptor& layer);
    uint64_t query_actual_resident_bytes_locked(const LayerDescriptor& layer) const;
};

} // namespace amuquant

#endif // AMUQUANT_LAYER_MANAGER_HPP
