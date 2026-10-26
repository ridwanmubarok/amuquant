#include "amuquant/controller.hpp"

#include <cmath>
#include <algorithm>

namespace amuquant {

AMUQuantController::AMUQuantController(
    const AMUQuantConfig& config,
    std::shared_ptr<SystemTelemetry> telemetry,
    std::shared_ptr<LayerManager> layer_mgr,
    std::shared_ptr<KVSurgeon> kv_surgeon,
    TurbovecIndexHandle* turbovec_idx
) : config_(config),
    telemetry_(std::move(telemetry)),
    layer_mgr_(std::move(layer_mgr)),
    kv_surgeon_(std::move(kv_surgeon)),
    turbovec_idx_(turbovec_idx),
    current_regime_(Regime::REGIME_1_HIGH_THROUGHPUT),
    prefetch_depth_(3),
    aggressive_evict_(false),
    current_bit_width_(config.initial_bit_width) {}

double AMUQuantController::compute_phi_score(
    const MemoryStats& current_stats,
    double elapsed_sec,
    double delta_rss_bytes
) const {
    uint64_t safe_limit = (config_.memory_limit_bytes > config_.os_reserved_bytes) ? 
                          (config_.memory_limit_bytes - config_.os_reserved_bytes) : config_.memory_limit_bytes;
    if (safe_limit == 0) return 0.0;

    // 1. Instantaneous memory utilization ratio relative to safe budget
    double utilization = static_cast<double>(current_stats.process_rss_bytes) / static_cast<double>(safe_limit);

    // 2. Projected memory expansion rate over horizon window tau
    double dt = std::max(elapsed_sec, 0.05); // Numerical guard against near-zero delta time
    double dM_dt = delta_rss_bytes / dt;
    double projected_growth = (dM_dt * config_.tau_horizon_sec) / static_cast<double>(safe_limit);
    
    // Growth velocity only penalizes when positive
    projected_growth = std::max(projected_growth, 0.0);
    // Clamp growth penalty to [0.0, 1.0] to dampen transient allocation spikes
    projected_growth = std::min(projected_growth, 1.0);

    // 3. Predictive pressure metric: instantaneous baseline + growth rate penalty
    double phi = utilization + (config_.beta * projected_growth);
    return phi;
}

StepDecision AMUQuantController::on_token_step(
    uint32_t token_step,
    uint32_t current_seq_len,
    const KVVectorExtractor& extractor,
    const KVSeqRmFunc& seq_rm,
    const KVSeqAddFunc& seq_add
) {
    StepDecision decision;
    decision.regime = current_regime_;
    decision.prefetch_depth = prefetch_depth_;
    decision.aggressive_evict = aggressive_evict_;
    decision.phi_score = last_phi_;

    // Sample telemetry at scheduled token stride
    if (token_step % config_.eval_interval_tokens != 0 && !first_eval_) {
        return decision;
    }

    MemoryStats stats = telemetry_->sample();
    double now = stats.timestamp_sec;
    
    double elapsed_sec = first_eval_ ? 1.0 : (now - last_eval_time_);
    double delta_rss = first_eval_ ? 0.0 : static_cast<double>(stats.process_rss_bytes) - static_cast<double>(last_eval_rss_);
    
    last_eval_time_ = now;
    last_eval_rss_ = stats.process_rss_bytes;
    first_eval_ = false;

    // Evaluate predictive pressure score Phi_t
    double phi = compute_phi_score(stats, elapsed_sec, delta_rss);
    last_phi_ = phi;
    decision.phi_score = phi;

    // Hysteresis-guarded finite state machine evaluation
    Regime target_regime = current_regime_;

    switch (current_regime_) {
        case Regime::REGIME_1_HIGH_THROUGHPUT:
            if (phi > config_.theta_high) {
                target_regime = Regime::REGIME_3_ULTRA_LOW_SURVIVAL;
            } else if (phi > config_.theta_low) {
                target_regime = Regime::REGIME_2_BALANCED_OFFLOAD;
            }
            break;

        case Regime::REGIME_2_BALANCED_OFFLOAD:
            if (phi > config_.theta_high) {
                target_regime = Regime::REGIME_3_ULTRA_LOW_SURVIVAL;
            } else if (phi <= (config_.theta_low - config_.delta_hyst)) {
                target_regime = Regime::REGIME_1_HIGH_THROUGHPUT;
            }
            break;

        case Regime::REGIME_3_ULTRA_LOW_SURVIVAL:
            if (phi <= (config_.theta_low - config_.delta_hyst)) {
                target_regime = Regime::REGIME_1_HIGH_THROUGHPUT;
            } else if (phi <= (config_.theta_high - config_.delta_hyst)) {
                target_regime = Regime::REGIME_2_BALANCED_OFFLOAD;
            }
            break;
    }

    apply_regime_transition(target_regime, current_seq_len, extractor, seq_rm, seq_add, decision);
    return decision;
}

void AMUQuantController::apply_regime_transition(
    Regime target,
    uint32_t current_seq_len,
    const KVVectorExtractor& extractor,
    const KVSeqRmFunc& seq_rm,
    const KVSeqAddFunc& seq_add,
    StepDecision& decision
) {
    current_regime_ = target;
    decision.regime = target;

    switch (target) {
        case Regime::REGIME_1_HIGH_THROUGHPUT: {
            prefetch_depth_ = 3;
            aggressive_evict_ = false;
            decision.prefetch_depth = 3;
            decision.aggressive_evict = false;
            break;
        }

        case Regime::REGIME_2_BALANCED_OFFLOAD: {
            prefetch_depth_ = 1;
            aggressive_evict_ = true;
            decision.prefetch_depth = 1;
            decision.aggressive_evict = true;

            // Evict middle tokens to compressed TurboQuant index
            if (kv_surgeon_) {
                auto range = kv_surgeon_->plan_eviction(
                    current_seq_len,
                    config_.attention_sink_tokens,
                    config_.recent_window_tokens
                );
                if (range.valid()) {
                    bool ok = kv_surgeon_->execute_eviction(range, extractor, seq_rm, seq_add);
                    decision.kv_pruned = ok;
                }
            }
            break;
        }

        case Regime::REGIME_3_ULTRA_LOW_SURVIVAL: {
            prefetch_depth_ = 0; // Strict single-layer execution
            aggressive_evict_ = true;
            decision.prefetch_depth = 0;
            decision.aggressive_evict = true;

            // In-place requantization to 2-bit representation
            if (current_bit_width_ == 4 && turbovec_idx_ != nullptr) {
                turbovec_requantize_to_2bit(turbovec_idx_);
                current_bit_width_ = 2;
                decision.requantized_to_2bit = true;
            }

            // Evict KV tokens under compressed recent window
            if (kv_surgeon_) {
                auto range = kv_surgeon_->plan_eviction(
                    current_seq_len,
                    config_.attention_sink_tokens,
                    config_.min_recent_window_tokens
                );
                if (range.valid()) {
                    bool ok = kv_surgeon_->execute_eviction(range, extractor, seq_rm, seq_add);
                    decision.kv_pruned = ok;
                }
            }
            break;
        }
    }
}

void AMUQuantController::reset() {
    current_regime_ = Regime::REGIME_1_HIGH_THROUGHPUT;
    last_phi_ = 0.0;
    prefetch_depth_ = 3;
    aggressive_evict_ = false;
    current_bit_width_ = config_.initial_bit_width;
    last_eval_time_ = 0.0;
    last_eval_rss_ = 0;
    first_eval_ = true;
}

} // namespace amuquant
