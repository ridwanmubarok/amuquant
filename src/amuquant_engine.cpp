#include "amuquant/engine.hpp"

#include <chrono>
#include <unordered_map>
#include <unordered_set>
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

uint64_t AMUQuantEngine::record_turn(const std::string& role, const std::string& content, int32_t token_start, int32_t token_end) {
    if (!history_store_) {
        history_store_ = std::make_shared<CanonicalHistoryStore>();
    }
    return history_store_->append_turn(role, content, token_start, token_end);
}

std::vector<RetrievedMemory> AMUQuantEngine::query_hybrid_memory(
    const std::string& query_text,
    const float* query_embedding,
    uint32_t top_k,
    float dense_weight,
    float lexical_weight
) {
    if (!history_store_ || history_store_->size() == 0 || top_k == 0) {
        return {};
    }

    // 1. Lexical search across evicted turns first
    auto lexical_matches = history_store_->search_lexical(query_text, true);
    if (lexical_matches.empty()) {
        // Fallback: search all turns if no evicted turns had direct lexical hits
        lexical_matches = history_store_->search_lexical(query_text, false);
    }

    std::unordered_map<uint64_t, float> lexical_scores;
    for (const auto& lm : lexical_matches) {
        lexical_scores[lm.first] = lm.second;
    }

    // 2. Dense search via TurboQuant vector index if query_embedding is provided
    std::unordered_map<uint64_t, float> dense_scores;
    if (query_embedding != nullptr && turbovec_handle_ != nullptr) {
        auto vec_hits = recall_context(query_embedding, top_k * 4);
        for (const auto& hit : vec_hits) {
            const HistoryTurn* turn = history_store_->find_turn_by_token(static_cast<int32_t>(hit.id));
            if (turn) {
                float& cur_score = dense_scores[turn->turn_id];
                if (hit.score > cur_score) {
                    cur_score = hit.score;
                }
            }
        }
    }

    // 3. Score Fusion: aggregate candidate turns
    std::unordered_set<uint64_t> candidates;
    for (const auto& kv : lexical_scores) candidates.insert(kv.first);
    for (const auto& kv : dense_scores) candidates.insert(kv.first);

    if (candidates.empty()) {
        return {};
    }

    std::vector<RetrievedMemory> results;
    for (uint64_t tid : candidates) {
        const HistoryTurn* turn = history_store_->get_turn(tid);
        if (!turn) continue;

        float d_score = dense_scores.count(tid) ? dense_scores[tid] : 0.0f;
        float l_score = lexical_scores.count(tid) ? lexical_scores[tid] : 0.0f;

        float combined = 0.0f;
        if (d_score > 0.0f && l_score > 0.0f) {
            combined = dense_weight * d_score + lexical_weight * l_score;
        } else if (l_score > 0.0f) {
            combined = 0.5f + 0.5f * l_score;
        } else {
            combined = d_score;
        }

        RetrievedMemory rm;
        rm.turn_id = tid;
        rm.role = turn->role;
        rm.content = turn->content;
        rm.dense_score = d_score;
        rm.lexical_score = l_score;
        rm.combined_score = combined;
        results.push_back(std::move(rm));
    }

    std::sort(results.begin(), results.end(), [](const auto& a, const auto& b) {
        return a.combined_score > b.combined_score;
    });

    if (results.size() > top_k) {
        results.resize(top_k);
    }
    return results;
}

std::string AMUQuantEngine::reconstruct_memory_prompt(
    const std::string& query_text,
    const float* query_embedding,
    uint32_t top_k
) {
    if (!history_store_) return "";
    auto memories = query_hybrid_memory(query_text, query_embedding, top_k);
    return history_store_->format_system_memory_context(memories);
}

EngineMetrics AMUQuantEngine::get_metrics() const {
    return metrics_;
}

} // namespace amuquant
