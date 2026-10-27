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
     * @brief Returns the scheduled resident weight footprint across all registered layers.
     * Reflects AMU-QUANT's logical prefetch and eviction advisory intent state.
     */
    uint64_t get_scheduled_resident_bytes() const;

    /**
     * @brief Backwards-compatible alias for get_scheduled_resident_bytes().
     */
    uint64_t get_resident_weight_bytes() const { return get_scheduled_resident_bytes(); }

    /**
     * @brief Returns the total registered weight footprint across all layers.
     */
    uint64_t get_total_registered_bytes() const;

    /**
     * @brief Returns the verified physically evicted weight bytes across all layers.
     * Computed as: total_registered_bytes - total_actual_resident_bytes (via mincore).
     */
    uint64_t get_actual_evicted_bytes() const;

    /**
     * @brief Queries actual resident bytes for a layer directly from kernel page tables via mincore(2).
     *
     * Disambiguates internal advisory tracking status (is_resident) from physical RAM residency.
     * Note: posix_madvise is an OS advisory; mincore(2) verifies whether page frames are physical core-resident
     * in the CPU virtual address space for file-backed mmap regions. It does not measure separate GPU device
     * or unified memory buffers managed outside CPU page tables.
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

    /**
     * @brief Synchronizes scheduled advisory state with actual kernel page table residency.
     * Inspects mincore(2) for all registered layers and updates scheduled_layers_ and is_resident flags.
     * @return Number of layers verified physically resident in RAM.
     */
    size_t sync_scheduled_with_actual_residency();

    /**
     * @brief Returns the set of layer indices currently verified resident via mincore(2).
     */
    std::unordered_set<uint32_t> get_actual_resident_layers() const;

    /**
     * @brief Returns the set of layer indices currently scheduled as resident.
     */
    std::unordered_set<uint32_t> get_scheduled_resident_layers() const;

    size_t num_layers() const;
    void reset();

private:
    std::vector<LayerDescriptor> layers_;
    std::unordered_set<uint32_t> scheduled_layers_;
    mutable std::mutex mutex_;

    void evict_layer_pages(LayerDescriptor& layer);
    void prefetch_layer_pages(LayerDescriptor& layer);
    uint64_t query_actual_resident_bytes_locked(const LayerDescriptor& layer) const;
};

} // namespace amuquant

#endif // AMUQUANT_LAYER_MANAGER_HPP
