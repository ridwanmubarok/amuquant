#include "amuquant/engine.hpp"

#include <chrono>
#include <algorithm>

namespace amuquant {

AMUQuantEngine::AMUQuantEngine(const AMUQuantConfig& config)
    : config_(config) {
    history_store_ = std::make_shared<CanonicalHistoryStore>();
}

AMUQuantEngine::~AMUQuantEngine() {
    controller_.reset();
    kv_surgeon_.reset();
    layer_mgr_.reset();
    telemetry_.reset();
    history_store_.reset();
}

Status AMUQuantEngine::initialize() {
    telemetry_ = std::make_shared<SystemTelemetry>();
    layer_mgr_ = std::make_shared<LayerManager>();
    kv_surgeon_ = std::make_shared<KVSurgeon>();
    controller_ = std::make_shared<AMUQuantController>(
        config_, telemetry_, layer_mgr_, kv_surgeon_
    );

    // Register eviction listener to automatically mark evicted sequence boundaries in canonical history
    kv_surgeon_->set_eviction_callback([this](const std::vector<uint64_t>& evicted_logical_ids) {
        if (history_store_) {
            for (uint64_t id : evicted_logical_ids) {
                history_store_->mark_evicted_range(static_cast<int32_t>(id), static_cast<int32_t>(id + 1));
            }
        }
    });

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
    const KVSeqRmFunc& seq_rm,
    const KVSeqAddFunc& seq_add
) {
    double t_start = SystemTelemetry::get_time_seconds();

    StepDecision decision = controller_->on_token_step(
        token_step, current_seq_len, seq_rm, seq_add
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

uint64_t AMUQuantEngine::record_turn(const std::string& role, const std::string& content, int32_t token_start, int32_t token_end) {
    if (!history_store_) {
        history_store_ = std::make_shared<CanonicalHistoryStore>();
    }
    return history_store_->append_turn(role, content, token_start, token_end);
}

std::vector<SearchResult> AMUQuantEngine::recall_context(const std::string& query_text, uint32_t top_k) {
    if (!history_store_ || query_text.empty()) {
        return {};
    }
    auto matches = history_store_->search_lexical(query_text, /*evicted_only=*/false);
    std::vector<SearchResult> results;
    for (size_t i = 0; i < matches.size() && (top_k == 0 || i < top_k); ++i) {
        results.push_back({matches[i].first, matches[i].second});
    }
    return results;
}

std::vector<SearchResult> AMUQuantEngine::recall_context(const float* /*embd*/, uint32_t /*top_k*/) {
    return {};
}

std::string AMUQuantEngine::reconstruct_memory_prompt(const std::string& query_text, const float* /*dense_query*/, uint32_t top_k) {
    if (!history_store_ || query_text.empty()) {
        return "";
    }

    auto matches = history_store_->search_lexical(query_text, /*evicted_only=*/true);
    if (matches.empty()) {
        matches = history_store_->search_lexical(query_text, /*evicted_only=*/false);
    }

    if (matches.empty()) {
        return "";
    }

    std::vector<RetrievedMemory> memories;
    for (const auto& match : matches) {
        if (match.second < 0.20f) continue;
        const auto* turn = history_store_->get_turn(match.first);
        if (turn) {
            RetrievedMemory m;
            m.turn_id = turn->turn_id;
            m.role = turn->role;
            m.content = turn->content;
            m.lexical_score = match.second;
            m.combined_score = match.second;
            memories.push_back(m);
            if (top_k > 0 && memories.size() >= top_k) break;
        }
    }

    if (memories.empty()) {
        return "";
    }

    return history_store_->format_system_memory_context(memories);
}

EngineMetrics AMUQuantEngine::get_metrics() const {
    return metrics_;
}

} // namespace amuquant
