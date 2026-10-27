#include "amuquant/layer_manager.hpp"

#include <sys/types.h>
#include <sys/mman.h>
#include <unistd.h>
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
    scheduled_layers_.insert(layer_idx);

    // Prefetch upcoming layers according to scheduled prefetch depth
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
        int rc = 0;
#if defined(POSIX_MADV_DONTNEED)
        rc = posix_madvise(layer.addr_start, layer.byte_len, POSIX_MADV_DONTNEED);
#elif defined(MADV_DONTNEED)
        rc = madvise(layer.addr_start, layer.byte_len, MADV_DONTNEED);
#endif
        if (rc != 0) {
            // Advisory system call failed; preserve residency intent to avoid state desync
            return;
        }
    }
    layer.is_resident = false;
    scheduled_layers_.erase(layer.layer_idx);
}

void LayerManager::prefetch_layer_pages(LayerDescriptor& layer) {
    if (layer.is_resident) {
        return;
    }
    if (layer.addr_start != nullptr && layer.byte_len > 0) {
        int rc = 0;
#if defined(POSIX_MADV_WILLNEED)
        rc = posix_madvise(layer.addr_start, layer.byte_len, POSIX_MADV_WILLNEED);
#elif defined(MADV_WILLNEED)
        rc = madvise(layer.addr_start, layer.byte_len, MADV_WILLNEED);
#endif
        if (rc != 0) {
            // Advisory prefetch call failed; do not update intent tracking
            return;
        }
    }
    layer.is_resident = true;
    scheduled_layers_.insert(layer.layer_idx);
}

uint64_t LayerManager::get_scheduled_resident_bytes() const {
    std::lock_guard<std::mutex> lock(mutex_);
    uint64_t total = 0;
    for (uint32_t idx : scheduled_layers_) {
        if (idx < layers_.size()) {
            total += layers_[idx].byte_len;
        }
    }
    return total;
}

uint64_t LayerManager::get_total_registered_bytes() const {
    std::lock_guard<std::mutex> lock(mutex_);
    uint64_t total = 0;
    for (const auto& layer : layers_) {
        total += layer.byte_len;
    }
    return total;
}

uint64_t LayerManager::get_actual_evicted_bytes() const {
    std::lock_guard<std::mutex> lock(mutex_);
    uint64_t total_mapped = 0;
    uint64_t actual_resident = 0;
    for (const auto& layer : layers_) {
        total_mapped += layer.byte_len;
        actual_resident += query_actual_resident_bytes_locked(layer);
    }
    return (total_mapped > actual_resident) ? (total_mapped - actual_resident) : 0;
}

uint64_t LayerManager::query_actual_resident_bytes_locked(const LayerDescriptor& layer) const {
    if (layer.addr_start == nullptr || layer.byte_len == 0) {
        return 0;
    }

    const long page_size = sysconf(_SC_PAGESIZE);
    if (page_size <= 0) {
        return 0;
    }

    const uintptr_t start = reinterpret_cast<uintptr_t>(layer.addr_start);
    const uintptr_t page_mask = static_cast<uintptr_t>(page_size - 1);
    const uintptr_t aligned_start = start & ~page_mask;
    const size_t offset = static_cast<size_t>(start - aligned_start);
    const size_t aligned_len = ((layer.byte_len + offset + page_mask) / page_size) * page_size;
    const size_t num_pages = aligned_len / page_size;

    if (num_pages == 0) {
        return 0;
    }

    std::vector<unsigned char> vec(num_pages, 0);

#if defined(__APPLE__) || defined(__MACH__)
    int res = mincore(reinterpret_cast<caddr_t>(aligned_start), aligned_len, reinterpret_cast<char*>(vec.data()));
#elif defined(_WIN32)
    int res = -1;
#else
    int res = mincore(reinterpret_cast<void*>(aligned_start), aligned_len, vec.data());
#endif

    if (res != 0) {
        return 0;
    }

    uint64_t resident_pages = 0;
    for (size_t i = 0; i < num_pages; ++i) {
        if (vec[i] & 1) {
            resident_pages++;
        }
    }

    return resident_pages * static_cast<uint64_t>(page_size);
}

uint64_t LayerManager::get_actual_resident_bytes(uint32_t layer_idx) const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (layer_idx >= layers_.size()) {
        return 0;
    }
    return query_actual_resident_bytes_locked(layers_[layer_idx]);
}

uint64_t LayerManager::get_total_actual_resident_bytes() const {
    std::lock_guard<std::mutex> lock(mutex_);
    uint64_t total = 0;
    for (const auto& layer : layers_) {
        total += query_actual_resident_bytes_locked(layer);
    }
    return total;
}

std::unordered_set<uint32_t> LayerManager::get_actual_resident_layers() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::unordered_set<uint32_t> actual_set;
    for (const auto& layer : layers_) {
        if (query_actual_resident_bytes_locked(layer) > 0) {
            actual_set.insert(layer.layer_idx);
        }
    }
    return actual_set;
}

std::unordered_set<uint32_t> LayerManager::get_scheduled_resident_layers() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return scheduled_layers_;
}

size_t LayerManager::sync_scheduled_with_actual_residency() {
    std::lock_guard<std::mutex> lock(mutex_);
    scheduled_layers_.clear();
    for (auto& layer : layers_) {
        uint64_t resident_bytes = query_actual_resident_bytes_locked(layer);
        layer.is_resident = (resident_bytes > 0);
        if (layer.is_resident) {
            scheduled_layers_.insert(layer.layer_idx);
        }
    }
    return scheduled_layers_.size();
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
    scheduled_layers_.clear();
}

} // namespace amuquant
