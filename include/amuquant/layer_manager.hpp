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
     */
    uint64_t get_resident_weight_bytes() const;

    size_t num_layers() const;
    void reset();

private:
    std::vector<LayerDescriptor> layers_;
    std::unordered_set<uint32_t> resident_layers_;
    mutable std::mutex mutex_;

    void evict_layer_pages(LayerDescriptor& layer);
    void prefetch_layer_pages(LayerDescriptor& layer);
};

} // namespace amuquant

#endif // AMUQUANT_LAYER_MANAGER_HPP
