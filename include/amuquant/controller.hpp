/**
 * @file controller.hpp
 * @brief Predictive memory pressure controller and co-adaptive finite state machine.
 */

#ifndef AMUQUANT_CONTROLLER_HPP
#define AMUQUANT_CONTROLLER_HPP

#include "amuquant/types.hpp"
#include "amuquant/telemetry.hpp"
#include "amuquant/layer_manager.hpp"
#include "amuquant/kv_surgeon.hpp"
#include "amuquant/turbovec_ffi.h"

#include <memory>
#include <cstdint>

namespace amuquant {

class AMUQuantController {
public:
    AMUQuantController(
        const AMUQuantConfig& config,
        std::shared_ptr<SystemTelemetry> telemetry,
        std::shared_ptr<LayerManager> layer_mgr,
        std::shared_ptr<KVSurgeon> kv_surgeon,
        TurbovecIndexHandle* turbovec_idx
    );
    ~AMUQuantController() = default;

    /**
     * @brief Periodic arbitration step evaluating memory pressure, KV eviction, and prefetch depth.
     */
    StepDecision on_token_step(
        uint32_t token_step,
        uint32_t current_seq_len,
        const KVVectorExtractor& extractor,
        const KVSeqRmFunc& seq_rm,
        const KVSeqAddFunc& seq_add
    );

    /**
     * @brief Calculates predictive pressure metric Phi_t combining static saturation and differential slope.
     */
    double compute_phi_score(
        const MemoryStats& current_stats,
        double elapsed_sec,
        double delta_rss_bytes
    ) const;

    /**
     * @brief Evaluates composite cost-benefit utility J(a) for a candidate action regime.
     */
    double evaluate_action_cost(
        Regime candidate_regime,
        const MemoryStats& current_stats,
        double projected_process_rss,
        double projected_available_ram,
        uint32_t current_seq_len
    ) const;

    /**
     * @brief Computes optimal action regime minimizing the composite utility objective.
     */
    Regime arbitrate_predictive_cost_benefit(
        const MemoryStats& current_stats,
        double elapsed_sec,
        double delta_rss_bytes,
        uint32_t current_seq_len,
        double& out_optimal_cost
    ) const;

    Regime get_current_regime() const { return current_regime_; }
    uint32_t get_prefetch_depth() const { return prefetch_depth_; }
    bool is_aggressive_evict() const { return aggressive_evict_; }
    double get_last_phi() const { return last_phi_; }
    uint8_t get_current_bit_width() const { return current_bit_width_; }

    /**
     * @brief Records actual measured physical memory reclamation post-eviction.
     * Updates adaptive calibration factor to minimize estimation bias.
     */
    void record_actual_reclaimed(uint64_t actual_reclaimed_bytes);

    double get_adaptive_calibration_factor() const { return adaptive_calib_factor_; }
    uint64_t get_last_estimated_savings() const { return last_estimated_savings_bytes_; }
    uint64_t get_last_actual_reclaimed() const { return last_actual_reclaimed_bytes_; }
    int64_t get_last_estimation_delta() const { return last_estimation_delta_bytes_; }

    void reset();

private:
    void apply_regime_transition(
        Regime target,
        uint32_t current_seq_len,
        const KVVectorExtractor& extractor,
        const KVSeqRmFunc& seq_rm,
        const KVSeqAddFunc& seq_add,
        StepDecision& decision
    );

    AMUQuantConfig config_;
    std::shared_ptr<SystemTelemetry> telemetry_;
    std::shared_ptr<LayerManager> layer_mgr_;
    std::shared_ptr<KVSurgeon> kv_surgeon_;
    TurbovecIndexHandle* turbovec_idx_{nullptr};

    Regime current_regime_{Regime::REGIME_1_HIGH_THROUGHPUT};
    uint32_t prefetch_depth_{3};
    bool aggressive_evict_{false};
    uint8_t current_bit_width_{4};
    double last_phi_{0.0};

    double adaptive_calib_factor_{1.0};
    uint64_t last_estimated_savings_bytes_{0};
    uint64_t last_actual_reclaimed_bytes_{0};
    int64_t last_estimation_delta_bytes_{0};

    double last_eval_time_{0.0};
    uint64_t last_eval_rss_{0};
    bool first_eval_{true};
};

} // namespace amuquant

#endif // AMUQUANT_CONTROLLER_HPP
