/**
 * @file engine.hpp
 * @brief Unified AMU-QUANT runtime engine coordinating layers, telemetry, and KV surgery.
 */

#ifndef AMUQUANT_ENGINE_HPP
#define AMUQUANT_ENGINE_HPP

#include "amuquant/types.hpp"
#include "amuquant/telemetry.hpp"
#include "amuquant/layer_manager.hpp"
#include "amuquant/kv_surgeon.hpp"
#include "amuquant/controller.hpp"
#include "amuquant/canonical_history.hpp"
#include "amuquant/turboquant.hpp"
#include "amuquant/turbovec_ffi.h"

#include <memory>
#include <string>

namespace amuquant {

struct EngineMetrics {
    uint32_t total_tokens_generated{0};
    uint32_t total_kv_evictions{0};
    uint32_t total_turbovec_queries{0};
    double total_inter_token_latency_ms{0.0};
    double avg_inter_token_latency_ms{0.0};
    uint64_t peak_rss_bytes{0};
    uint64_t major_page_faults{0};
    Regime final_regime{Regime::REGIME_1_HIGH_THROUGHPUT};
};

class AMUQuantEngine {
public:
    explicit AMUQuantEngine(const AMUQuantConfig& config);
    ~AMUQuantEngine();

    /**
     * @brief Initializes subcomponents, allocates the TurboQuant vector index, and tests telemetry.
     */
    Status initialize();

    /**
     * @brief Registers an mmap-backed weight slice for a transformer layer.
     */
    void register_layer(uint32_t layer_idx, void* addr, size_t size_bytes);

    /**
     * @brief Hooks invoked around layer forward passes for prefetching and page eviction.
     */
    void begin_layer_forward(uint32_t layer_idx);
    void end_layer_forward(uint32_t layer_idx);

    /**
     * @brief Periodic token generation step hook driving KV eviction and regime arbitration.
     */
    StepDecision step_token(
        uint32_t token_step,
        uint32_t current_seq_len,
        const KVVectorExtractor& extractor,
        const KVSeqRmFunc& seq_rm,
        const KVSeqAddFunc& seq_add
    );

    /**
     * @brief Queries the compressed TurboQuant vector index for evicted context chunks.
     */
    std::vector<SearchResult> recall_context(const float* query_data, uint32_t top_k);

    /**
     * @brief Records a verbatim conversational turn into the canonical history store.
     */
    uint64_t record_turn(const std::string& role, const std::string& content, int32_t token_start, int32_t token_end);

    /**
     * @brief Performs query-driven hybrid retrieval (dense TurboQuant + lexical keyword) over evicted turns.
     */
    std::vector<RetrievedMemory> query_hybrid_memory(
        const std::string& query_text,
        const float* query_embedding,
        uint32_t top_k = 3,
        float dense_weight = 0.6f,
        float lexical_weight = 0.4f
    );

    /**
     * @brief Generates an explicit structured system memory context block from retrieved evicted turns.
     */
    std::string reconstruct_memory_prompt(
        const std::string& query_text,
        const float* query_embedding,
        uint32_t top_k = 3
    );

    EngineMetrics get_metrics() const;

    TurbovecIndexHandle* get_turbovec_handle() { return turbovec_handle_; }
    std::shared_ptr<SystemTelemetry> get_telemetry() { return telemetry_; }
    std::shared_ptr<LayerManager> get_layer_manager() { return layer_mgr_; }
    std::shared_ptr<KVSurgeon> get_kv_surgeon() { return kv_surgeon_; }
    std::shared_ptr<AMUQuantController> get_controller() { return controller_; }
    std::shared_ptr<CanonicalHistoryStore> get_history_store() { return history_store_; }
    const std::shared_ptr<CanonicalHistoryStore> get_history_store() const { return history_store_; }

    TokenIndexMap& get_token_map() { return kv_surgeon_->get_token_map(); }
    const TokenIndexMap& get_token_map() const { return kv_surgeon_->get_token_map(); }
    void init_token_sequence(uint32_t prompt_tokens) { if (kv_surgeon_) kv_surgeon_->get_token_map().init(prompt_tokens); }
    void append_token(uint64_t logical_id) { if (kv_surgeon_) kv_surgeon_->get_token_map().append_token(logical_id); }

private:
    AMUQuantConfig config_;
    TurbovecIndexHandle* turbovec_handle_{nullptr};
    std::shared_ptr<SystemTelemetry> telemetry_;
    std::shared_ptr<LayerManager> layer_mgr_;
    std::shared_ptr<KVSurgeon> kv_surgeon_;
    std::shared_ptr<AMUQuantController> controller_;
    std::shared_ptr<CanonicalHistoryStore> history_store_;
    EngineMetrics metrics_;
};

} // namespace amuquant

#endif // AMUQUANT_ENGINE_HPP
