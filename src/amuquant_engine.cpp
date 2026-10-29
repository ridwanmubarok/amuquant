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
    uint64_t tid = history_store_->append_turn(role, content, token_start, token_end);
    reliable_metrics_.total_facts_registered = static_cast<uint32_t>(history_store_->get_facts(false).size());
    reliable_metrics_.active_facts_count = static_cast<uint32_t>(history_store_->get_facts(true).size());
    reliable_metrics_.superseded_facts_count = reliable_metrics_.total_facts_registered - reliable_metrics_.active_facts_count;
    reliable_metrics_.conflict_resolutions_triggered = history_store_->get_conflict_count();
    return tid;
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

std::string AMUQuantEngine::reconstruct_memory_prompt(const std::string& query_text, const float* /*dense_query*/, uint32_t /*top_k*/) {
    if (!history_store_ || query_text.empty()) {
        return "";
    }

    // Use grounded prompt formatting with provenance and conflict awareness
    float rel = 0.0f;
    bool abstain = false;
    return history_store_->format_grounded_memory_prompt(query_text, config_.evidence_threshold, &rel, &abstain);
}

std::string AMUQuantEngine::construct_grounded_prompt(
    const std::string& user_query,
    float* out_relevance,
    bool* out_abstain_recommended
) {
    if (!history_store_) {
        history_store_ = std::make_shared<CanonicalHistoryStore>();
    }
    float rel = 0.0f;
    bool abstain = false;
    std::string prompt_block = history_store_->format_grounded_memory_prompt(
        user_query,
        config_.evidence_threshold,
        &rel,
        &abstain
    );
    if (out_relevance) *out_relevance = rel;
    if (out_abstain_recommended) *out_abstain_recommended = abstain;

    if (abstain) {
        reliable_metrics_.abstentions_recommended++;
    }

    return prompt_block;
}

bool AMUQuantEngine::verify_response_grounding(const std::string& model_output, const std::string& context_block) const {
    std::string lower_out = model_output;
    std::transform(lower_out.begin(), lower_out.end(), lower_out.begin(), [](unsigned char c) {
        return std::tolower(c);
    });

    if (context_block.find("[RELIABLE MEMORY GUARD: INSUFFICIENT EVIDENCE]") != std::string::npos) {
        static const std::vector<std::string> ABSTAIN_MARKERS = {
            "tidak tahu", "tidak ditemukan", "tidak tercatat", "belum tercatat",
            "tidak ada informasi", "tidak terdapat", "tidak disebutkan", "unknown",
            "not mentioned", "not recorded", "cannot answer", "maaf"
        };
        for (const auto& marker : ABSTAIN_MARKERS) {
            if (lower_out.find(marker) != std::string::npos) {
                reliable_metrics_.grounding_verifications_passed++;
                return true;
            }
        }
        reliable_metrics_.grounding_verifications_failed++;
        return false;
    }

    if (history_store_) {
        auto facts = history_store_->get_facts(false);
        for (const auto& f : facts) {
            if (f.status == FactStatus::SUPERSEDED) {
                std::string lower_stale_val = f.entity_value;
                std::transform(lower_stale_val.begin(), lower_stale_val.end(), lower_stale_val.begin(), [](unsigned char c) {
                    return std::tolower(c);
                });
                if (!lower_stale_val.empty() && lower_out.find(lower_stale_val) != std::string::npos) {
                    if (lower_out.find("supersede") == std::string::npos && lower_out.find("sebelumnya") == std::string::npos) {
                        reliable_metrics_.grounding_verifications_failed++;
                        return false;
                    }
                }
            }
        }
    }

    reliable_metrics_.grounding_verifications_passed++;
    return true;
}

ReliableMemoryMetrics AMUQuantEngine::get_reliable_memory_metrics() const {
    if (history_store_) {
        reliable_metrics_.total_facts_registered = static_cast<uint32_t>(history_store_->get_facts(false).size());
        reliable_metrics_.active_facts_count = static_cast<uint32_t>(history_store_->get_facts(true).size());
        reliable_metrics_.superseded_facts_count = reliable_metrics_.total_facts_registered - reliable_metrics_.active_facts_count;
        reliable_metrics_.conflict_resolutions_triggered = history_store_->get_conflict_count();
    }
    return reliable_metrics_;
}

EngineMetrics AMUQuantEngine::get_metrics() const {
    return metrics_;
}

} // namespace amuquant
