#include "amuquant/kv_surgeon.hpp"

#include <algorithm>
#include <vector>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>

namespace amuquant {

class AsyncArchiver {
public:
    AsyncArchiver(TurbovecIndexHandle* turbovec_idx, uint32_t embedding_dim)
        : turbovec_idx_(turbovec_idx), embedding_dim_(embedding_dim) {
        worker_ = std::thread(&AsyncArchiver::worker_loop, this);
    }

    ~AsyncArchiver() {
        stop();
    }

    void enqueue(uint64_t logical_id, const float* data) {
        if (!data || stop_.load(std::memory_order_relaxed)) return;
        Task task;
        task.logical_id = logical_id;
        task.data.assign(data, data + embedding_dim_);

        pending_count_.fetch_add(1, std::memory_order_acq_rel);
        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            queue_.push(std::move(task));
        }
        cv_work_.notify_one();
    }

    void flush() {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        cv_done_.wait(lock, [this] {
            return pending_count_.load(std::memory_order_acquire) == 0;
        });
    }

    void stop() {
        bool expected = false;
        if (stop_.compare_exchange_strong(expected, true)) {
            cv_work_.notify_all();
            if (worker_.joinable()) {
                worker_.join();
            }
        }
    }

private:
    void worker_loop() {
        while (true) {
            Task task;
            {
                std::unique_lock<std::mutex> lock(queue_mutex_);
                cv_work_.wait(lock, [this] {
                    return stop_.load(std::memory_order_relaxed) || !queue_.empty();
                });

                if (stop_.load(std::memory_order_relaxed) && queue_.empty()) {
                    break;
                }

                task = std::move(queue_.front());
                queue_.pop();
            }

            turbovec_insert_zero_copy(
                turbovec_idx_,
                task.logical_id,
                task.data.data(),
                embedding_dim_
            );

            size_t rem = pending_count_.fetch_sub(1, std::memory_order_acq_rel) - 1;
            if (rem == 0) {
                std::lock_guard<std::mutex> lock(queue_mutex_);
                cv_done_.notify_all();
            }
        }
    }

    TurbovecIndexHandle* turbovec_idx_{nullptr};
    uint32_t embedding_dim_{0};
    struct Task {
        uint64_t logical_id{0};
        std::vector<float> data;
    };
    std::queue<Task> queue_;
    std::mutex queue_mutex_;
    std::condition_variable cv_work_;
    std::condition_variable cv_done_;
    std::atomic<bool> stop_{false};
    std::atomic<size_t> pending_count_{0};
    std::thread worker_;
};

KVSurgeon::KVSurgeon(TurbovecIndexHandle* turbovec_idx, uint32_t embedding_dim, bool enable_async)
    : turbovec_idx_(turbovec_idx), embedding_dim_(embedding_dim) {
    if (enable_async && turbovec_idx_ != nullptr) {
        async_archiver_ = std::make_unique<AsyncArchiver>(turbovec_idx_, embedding_dim_);
    }
}

KVSurgeon::~KVSurgeon() {
    if (async_archiver_) {
        async_archiver_->stop();
    }
}

void KVSurgeon::flush_archiver() {
    if (async_archiver_) {
        async_archiver_->flush();
    }
}

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
    const KVVectorExtractor& extractor,
    const KVSeqRmFunc& seq_rm,
    const KVSeqAddFunc& seq_add
) {
    if (!range.valid() || turbovec_idx_ == nullptr) {
        return false;
    }

    // Auto-populate token map if caller hasn't explicitly initialized it
    if (token_map_.physical_size() < static_cast<size_t>(range.p_end)) {
        size_t start = token_map_.physical_size();
        for (size_t i = start; i < static_cast<size_t>(range.p_end); ++i) {
            token_map_.append_token(static_cast<uint64_t>(i));
        }
    }

    // Phase 1: Staging and Verification
    // Extract representations using immutable logical IDs
    size_t count = static_cast<size_t>(range.p_end - range.p_start);
    std::vector<uint64_t> logical_ids;
    std::vector<const float*> extracted_vecs;
    logical_ids.reserve(count);
    extracted_vecs.reserve(count);

    for (int32_t pos = range.p_start; pos < range.p_end; ++pos) {
        uint64_t logical_id = token_map_.get_logical_id(pos);
        if (logical_id == UINT64_MAX) {
            return false;
        }
        const float* vec_ptr = extractor ? extractor(static_cast<int32_t>(logical_id)) : nullptr;
        if (vec_ptr == nullptr) {
            // Abort Phase 1: Missing vector representation, zero KV modifications
            return false;
        }
        logical_ids.push_back(logical_id);
        extracted_vecs.push_back(vec_ptr);
    }

    // Insert verified representations into TurboQuant vector index under immutable logical IDs
    if (async_archiver_) {
        for (size_t i = 0; i < count; ++i) {
            async_archiver_->enqueue(logical_ids[i], extracted_vecs[i]);
        }
    } else {
        for (size_t i = 0; i < count; ++i) {
            TurbovecStatus insert_st = turbovec_insert_zero_copy(
                turbovec_idx_, logical_ids[i], extracted_vecs[i], embedding_dim_
            );
            if (insert_st != TURBOVEC_STATUS_OK) {
                // Insertion into vector memory failed: abort eviction immediately to protect unarchived context
                return false;
            }
        }
    }

    // Phase 2: Transactional Commit
    // 1. Prune physical KV sequence slot range only after guaranteed vector archiving
    if (seq_rm) {
        seq_rm(range.p_start, range.p_end);
    }

    // 2. Apply RoPE positional shift to close the sequence gap
    int32_t shift_delta = -(range.p_end - range.p_start);
    if (seq_add) {
        seq_add(range.p_end, -1, shift_delta);
    }

    // 3. Update physical-to-logical translation table
    token_map_.prune_range(range.p_start, range.p_end);

    total_archived_tokens_ += static_cast<uint32_t>(count);
    cumulative_rope_shift_ += shift_delta;
    return true;
}

KVSurgeryStats KVSurgeon::get_stats(uint32_t num_layers, uint32_t kv_heads, uint32_t head_dim) const {
    if (async_archiver_) {
        const_cast<AsyncArchiver*>(async_archiver_.get())->flush();
    }

    KVSurgeryStats stats;
    stats.archived_chunks = total_archived_tokens_;
    stats.current_rope_shift = cumulative_rope_shift_;

    // Physical FP16 KV-cache memory: 2 * num_layers * kv_heads * head_dim * sizeof(uint16_t)
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
    if (async_archiver_) {
        async_archiver_->flush();
    }
    total_archived_tokens_ = 0;
    active_tokens_ = 0;
    cumulative_rope_shift_ = 0;
    token_map_.reset();
}

} // namespace amuquant
