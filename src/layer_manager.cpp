#include "amuquant/layer_manager.hpp"

#include <sys/mman.h>
#include <algorithm>

namespace amuquant {

LayerManager::LayerManager() = default;

void LayerManager::register_layer(uint32_t layer_idx, void* addr, size_t size_bytes) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (layer_idx >= layers_.size()) {
        layers_.resize(layer_idx + 1);
    }
    layers_[layer_idx] = {layer_idx, addr, size_bytes, false};
}

void LayerManager::on_layer_forward_begin(uint32_t layer_idx, uint32_t prefetch_depth) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (layer_idx >= layers_.size()) return;

    layers_[layer_idx].is_resident = true;
    resident_layers_.insert(layer_idx);

    // Prefetch layer mendatang sesuai kedalaman prefetch_depth
    for (uint32_t d = 1; d <= prefetch_depth; ++d) {
        uint32_t target_idx = layer_idx + d;
        if (target_idx < layers_.size()) {
            prefetch_layer_pages(layers_[target_idx]);
        }
    }
}

void LayerManager::on_layer_forward_end(uint32_t layer_idx, bool aggressive_evict) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (layer_idx >= layers_.size()) return;

    if (aggressive_evict) {
        evict_layer_pages(layers_[layer_idx]);
    }
}

void LayerManager::evict_layer_pages(LayerDescriptor& layer) {
    if (layer.addr_start != nullptr && layer.byte_len > 0) {
#if defined(POSIX_MADV_DONTNEED)
        posix_madvise(layer.addr_start, layer.byte_len, POSIX_MADV_DONTNEED);
#elif defined(MADV_DONTNEED)
        madvise(layer.addr_start, layer.byte_len, MADV_DONTNEED);
#endif
    }
    layer.is_resident = false;
    resident_layers_.erase(layer.layer_idx);
}

void LayerManager::prefetch_layer_pages(LayerDescriptor& layer) {
    if (layer.is_resident) {
        return;
    }
    if (layer.addr_start != nullptr && layer.byte_len > 0) {
#if defined(POSIX_MADV_WILLNEED)
        posix_madvise(layer.addr_start, layer.byte_len, POSIX_MADV_WILLNEED);
#elif defined(MADV_WILLNEED)
        madvise(layer.addr_start, layer.byte_len, MADV_WILLNEED);
#endif
    }
    layer.is_resident = true;
    resident_layers_.insert(layer.layer_idx);
}

uint64_t LayerManager::get_resident_weight_bytes() const {
    std::lock_guard<std::mutex> lock(mutex_);
    uint64_t total = 0;
    for (uint32_t idx : resident_layers_) {
        if (idx < layers_.size()) {
            total += layers_[idx].byte_len;
        }
    }
    return total;
}

size_t LayerManager::num_layers() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return layers_.size();
}

void LayerManager::reset() {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& l : layers_) {
        if (l.is_resident) {
            evict_layer_pages(l);
        }
    }
    layers_.clear();
    resident_layers_.clear();
}

} // namespace amuquant
