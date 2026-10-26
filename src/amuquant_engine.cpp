#include "amuquant/engine.hpp"

#include <chrono>

namespace amuquant {

AMUQuantEngine::AMUQuantEngine(const AMUQuantConfig& config)
    : config_(config) {}

AMUQuantEngine::~AMUQuantEngine() {
    controller_.reset();
    kv_surgeon_.reset();
    layer_mgr_.reset();
    telemetry_.reset();
    if (turbovec_handle_ != nullptr) {
        turbovec_free(turbovec_handle_);
        turbovec_handle_ = nullptr;
    }
}

Status AMUQuantEngine::initialize() {
    TurbovecStatus status = turbovec_init(config_.embedding_dim, config_.initial_bit_width, &turbovec_handle_);
    if (status != TURBOVEC_STATUS_OK || turbovec_handle_ == nullptr) {
        return Status::ERR_OUT_OF_MEMORY;
    }

    telemetry_ = std::make_shared<SystemTelemetry>();
    layer_mgr_ = std::make_shared<LayerManager>();
    kv_surgeon_ = std::make_shared<KVSurgeon>(turbovec_handle_, config_.embedding_dim);
    controller_ = std::make_shared<AMUQuantController>(
        config_, telemetry_, layer_mgr_, kv_surgeon_, turbovec_handle_
    );

    return Status::OK;
}

void AMUQuantEngine::register_layer(uint32_t layer_idx, void* addr, size_t size_bytes) {
    if (layer_mgr_) {
        layer_mgr_->register_layer(layer_idx, addr, size_bytes);
    }
}

void AMUQuantEngine::begin_layer_forward(uint32_t layer_idx) {
    if (layer_mgr_ && controller_) {
        layer_mgr_->on_layer_forward_begin(layer_idx, controller_->get_prefetch_depth());
    }
}

void AMUQuantEngine::end_layer_forward(uint32_t layer_idx) {
    if (layer_mgr_ && controller_) {
        layer_mgr_->on_layer_forward_end(layer_idx, controller_->is_aggressive_evict());
    }
}

StepDecision AMUQuantEngine::step_token(
    uint32_t token_step,
    uint32_t current_seq_len,
    const KVVectorExtractor& extractor,
    const KVSeqRmFunc& seq_rm,
    const KVSeqAddFunc& seq_add
) {
    double t_start = SystemTelemetry::get_time_seconds();

    StepDecision decision = controller_->on_token_step(
        token_step, current_seq_len, extractor, seq_rm, seq_add
    );

    double t_end = SystemTelemetry::get_time_seconds();
    double latency_ms = (t_end - t_start) * 1000.0;

    metrics_.total_tokens_generated++;
    metrics_.total_inter_token_latency_ms += latency_ms;
    metrics_.avg_inter_token_latency_ms = metrics_.total_inter_token_latency_ms / metrics_.total_tokens_generated;
    metrics_.final_regime = decision.regime;

    if (decision.kv_pruned) {
        metrics_.total_kv_evictions++;
    }

    if (token_step % config_.eval_interval_tokens == 0) {
        MemoryStats current_stats = telemetry_->sample();
        if (current_stats.process_rss_bytes > metrics_.peak_rss_bytes) {
            metrics_.peak_rss_bytes = current_stats.process_rss_bytes;
        }
        metrics_.major_page_faults = current_stats.major_page_faults;
    }

    return decision;
}

std::vector<SearchResult> AMUQuantEngine::recall_context(const float* query_data, uint32_t top_k) {
    if (turbovec_handle_ == nullptr || query_data == nullptr || top_k == 0) {
        return {};
    }

    if (kv_surgeon_) {
        kv_surgeon_->flush_archiver();
    }

    metrics_.total_turbovec_queries++;

    std::vector<uint64_t> out_ids(top_k);
    std::vector<float> out_scores(top_k);
    uint32_t count = 0;

    TurbovecStatus st = turbovec_search_simd(
        turbovec_handle_, query_data, top_k, out_ids.data(), out_scores.data(), &count
    );

    if (st != TURBOVEC_STATUS_OK) {
        return {};
    }

    std::vector<SearchResult> res(count);
    for (uint32_t i = 0; i < count; ++i) {
        res[i] = {out_ids[i], out_scores[i]};
    }
    return res;
}

EngineMetrics AMUQuantEngine::get_metrics() const {
    return metrics_;
}

} // namespace amuquant
