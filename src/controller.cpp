#include "amuquant/controller.hpp"

#include <cmath>
#include <algorithm>

namespace amuquant {

AMUQuantController::AMUQuantController(
    const AMUQuantConfig& config,
    std::shared_ptr<SystemTelemetry> telemetry,
    std::shared_ptr<LayerManager> layer_mgr,
    std::shared_ptr<KVSurgeon> kv_surgeon
) : config_(config),
    telemetry_(std::move(telemetry)),
    layer_mgr_(std::move(layer_mgr)),
    kv_surgeon_(std::move(kv_surgeon)),
    current_regime_(Regime::REGIME_1_HIGH_THROUGHPUT),
    prefetch_depth_(3),
    aggressive_evict_(false) {}

double AMUQuantController::compute_phi_score(
    const MemoryStats& current_stats,
    double elapsed_sec,
    double delta_rss_bytes
) const {
    uint64_t safe_limit = (config_.memory_limit_bytes > config_.os_reserved_bytes) ? 
                          (config_.memory_limit_bytes - config_.os_reserved_bytes) : config_.memory_limit_bytes;
    if (safe_limit == 0) return 0.0;

    // 1. Process memory utilization relative to process working limit
    double util_process = static_cast<double>(current_stats.process_rss_bytes) / static_cast<double>(safe_limit);

    // 2. System-wide memory utilization relative to available host capacity
    double util_system = 0.0;
    if (current_stats.total_ram_bytes > 0 && current_stats.available_ram_bytes > 0) {
        if (current_stats.available_ram_bytes <= config_.os_reserved_bytes) {
            // Available memory has breached the OS safety buffer: critical saturation
            double deficit = static_cast<double>(config_.os_reserved_bytes - current_stats.available_ram_bytes);
            util_system = 1.0 + (deficit / static_cast<double>(config_.os_reserved_bytes));
        } else if (current_stats.total_ram_bytes > config_.os_reserved_bytes) {
            // Host pressure scales dynamically as available memory approaches the reserve buffer
            double safe_headroom = static_cast<double>(config_.os_reserved_bytes) * 2.0;
            if (current_stats.available_ram_bytes < safe_headroom) {
                double span = safe_headroom - static_cast<double>(config_.os_reserved_bytes);
                double dist = safe_headroom - static_cast<double>(current_stats.available_ram_bytes);
                util_system = 0.5 + 0.5 * (dist / span);
            } else {
                util_system = 0.2; // Abundant host headroom
            }
        }
    }

    // Base memory pressure is dominated by whichever budget is closer to exhaustion
    double base_pressure = std::max(util_process, util_system);

    // 3. Projected memory expansion velocity over lookahead horizon tau
    double dt = std::max(elapsed_sec, 0.05); // Numerical guard against near-zero delta time
    double dM_dt = delta_rss_bytes / dt;
    double projected_growth = (dM_dt * config_.tau_horizon_sec) / static_cast<double>(safe_limit);
    projected_growth = std::max(projected_growth, 0.0);
    projected_growth = std::min(projected_growth, 1.0);

    // 4. Kernel swap thrashing penalty
    constexpr double SWAP_SCALE_BYTES = 2048.0 * 1024.0 * 1024.0; // 2.0 GB normalizer
    double swap_penalty = 0.0;
    if (current_stats.swap_used_bytes > 0) {
        double raw_swap = std::min(1.0, static_cast<double>(current_stats.swap_used_bytes) / SWAP_SCALE_BYTES);
        if (current_stats.available_ram_bytes > 0 && current_stats.available_ram_bytes < (config_.os_reserved_bytes * 2)) {
            double urgency = 1.0 - (static_cast<double>(current_stats.available_ram_bytes) / static_cast<double>(config_.os_reserved_bytes * 2));
            swap_penalty = raw_swap * std::max(0.1, urgency);
        } else {
            swap_penalty = raw_swap * 0.1; // Minimal baseline for dormant swap
        }
    }

    // 5. Hybrid predictive metric: baseline pressure + differential velocity + swap penalty
    double phi = base_pressure + 
                 (config_.beta * projected_growth) + 
                 (config_.gamma * swap_penalty);
    return phi;
}

double AMUQuantController::evaluate_action_cost(
    Regime candidate_regime,
    const MemoryStats& current_stats,
    double projected_process_rss,
    double projected_available_ram,
    uint32_t current_seq_len
) const {
    (void)current_stats;
    uint64_t safe_limit = (config_.memory_limit_bytes > config_.os_reserved_bytes) ? 
                          (config_.memory_limit_bytes - config_.os_reserved_bytes) : config_.memory_limit_bytes;
    if (safe_limit == 0) return 0.0;

    // Architecture-aware KV sizing: 2 * num_layers * kv_heads * head_dim * kv_bytes_per_element
    uint32_t n_heads = (config_.kv_heads > 0) ? config_.kv_heads : 8;
    uint32_t d_head = (config_.head_dim > 0) ? config_.head_dim : 128;
    uint32_t elem_bytes = (config_.kv_bytes_per_element > 0) ? config_.kv_bytes_per_element : 2;
    size_t bytes_per_token = 2ULL * config_.num_layers * n_heads * d_head * elem_bytes;
    if (bytes_per_token == 0) bytes_per_token = 36864;

    double memory_mitigation_bytes = 0.0;
    double latency_cost = 0.0;
    double quality_loss = 0.0;

    switch (candidate_regime) {
        case Regime::REGIME_1_HIGH_THROUGHPUT:
            memory_mitigation_bytes = 0.0;
            latency_cost = 0.0;
            quality_loss = 0.0;
            break;

        case Regime::REGIME_2_BALANCED_OFFLOAD: {
            uint32_t sink = config_.attention_sink_tokens;
            uint32_t win = config_.recent_window_tokens;
            if (current_seq_len > sink + win) {
                uint32_t middle_count = current_seq_len - sink - win;
                memory_mitigation_bytes = static_cast<double>(middle_count * bytes_per_token) * adaptive_calib_factor_;
            }
            latency_cost = 0.02; // Minimal sequence prune overhead
            quality_loss = 0.05;
            break;
        }

        case Regime::REGIME_3_ULTRA_LOW_SURVIVAL: {
            uint32_t sink = config_.attention_sink_tokens;
            uint32_t win = config_.min_recent_window_tokens;
            if (current_seq_len > sink + win) {
                uint32_t middle_count = current_seq_len - sink - win;
                memory_mitigation_bytes = static_cast<double>(middle_count * bytes_per_token) * adaptive_calib_factor_;
            }
            latency_cost = 0.08; // Single-layer paging advisory overhead
            quality_loss = 0.10;
            break;
        }
    }

    double mitigated_process_rss = std::max(0.0, projected_process_rss - memory_mitigation_bytes);
    double mitigated_available_ram = projected_available_ram + memory_mitigation_bytes;

    double process_deficit = (mitigated_process_rss > safe_limit) ? (mitigated_process_rss - safe_limit) : 0.0;
    double host_deficit = (mitigated_available_ram < config_.os_reserved_bytes) ? 
                          static_cast<double>(config_.os_reserved_bytes - mitigated_available_ram) : 0.0;
    
    double max_deficit = std::max(process_deficit, host_deficit);
    double normalized_deficit = max_deficit / static_cast<double>(safe_limit);
    
    // Proportional OOM risk penalty: strictly decreases as more memory is mitigated
    double oom_risk = config_.risk_oom_penalty * normalized_deficit;

    // Severe penalty when available RAM breaches 75% of OS reserve floor
    if (mitigated_available_ram < (static_cast<double>(config_.os_reserved_bytes) * 0.75)) {
        if (candidate_regime != Regime::REGIME_3_ULTRA_LOW_SURVIVAL) {
            oom_risk += 50.0;
        }
    }

    // Proactive working-set budget constraint:
    // If target_working_set_tokens is configured (> 0), penalize regimes retaining
    // excessive physical KV slots beyond the target working set.
    double working_set_bloat_penalty = 0.0;
    if (config_.target_working_set_tokens > 0) {
        uint32_t retained_slots = current_seq_len;
        if (candidate_regime == Regime::REGIME_2_BALANCED_OFFLOAD) {
            uint32_t sink = config_.attention_sink_tokens;
            uint32_t win = config_.recent_window_tokens;
            retained_slots = std::min(current_seq_len, sink + win);
        } else if (candidate_regime == Regime::REGIME_3_ULTRA_LOW_SURVIVAL) {
            uint32_t sink = config_.attention_sink_tokens;
            uint32_t win = config_.min_recent_window_tokens;
            retained_slots = std::min(current_seq_len, sink + win);
        }

        if (retained_slots > config_.target_working_set_tokens) {
            double excess = static_cast<double>(retained_slots - config_.target_working_set_tokens);
            working_set_bloat_penalty = config_.weight_working_set_penalty * (excess / static_cast<double>(config_.target_working_set_tokens));
        }
    }

    double J = oom_risk + 
               working_set_bloat_penalty +
               (config_.weight_latency_cost * latency_cost) + 
               (config_.weight_quality_loss * quality_loss);
    return J;
}

Regime AMUQuantController::arbitrate_predictive_cost_benefit(
    const MemoryStats& current_stats,
    double elapsed_sec,
    double delta_rss_bytes,
    uint32_t current_seq_len,
    double& out_optimal_cost
) const {
    double dt = std::max(elapsed_sec, 0.05);
    double dM_dt = delta_rss_bytes / dt;
    double projected_rss_growth = std::max(0.0, dM_dt * config_.tau_horizon_sec);

    double projected_process_rss = static_cast<double>(current_stats.process_rss_bytes) + projected_rss_growth;
    double projected_available_ram = (current_stats.available_ram_bytes > projected_rss_growth) ? 
                                     (static_cast<double>(current_stats.available_ram_bytes) - projected_rss_growth) : 0.0;

    Regime candidates[3] = {
        Regime::REGIME_1_HIGH_THROUGHPUT,
        Regime::REGIME_2_BALANCED_OFFLOAD,
        Regime::REGIME_3_ULTRA_LOW_SURVIVAL
    };

    Regime optimal_regime = current_regime_;
    double min_cost = 1e9;

    for (int i = 0; i < 3; ++i) {
        Regime r = candidates[i];
        double cost = evaluate_action_cost(r, current_stats, projected_process_rss, projected_available_ram, current_seq_len);
        
        // Hysteresis dampening: bias slightly towards maintaining the current operational regime
        if (r == current_regime_) {
            cost -= 0.01;
        }

        if (cost < min_cost) {
            min_cost = cost;
            optimal_regime = r;
        }
    }

    out_optimal_cost = min_cost;
    return optimal_regime;
}

StepDecision AMUQuantController::on_token_step(
    uint32_t token_step,
    uint32_t current_seq_len,
    const KVSeqRmFunc& seq_rm,
    const KVSeqAddFunc& seq_add
) {
    StepDecision decision;
    decision.policy_used = config_.policy;
    decision.calibration_factor = adaptive_calib_factor_;

    if (token_step % config_.eval_interval_tokens != 0) {
        decision.regime = current_regime_;
        decision.prefetch_depth = prefetch_depth_;
        decision.aggressive_evict = aggressive_evict_;
        decision.phi_score = last_phi_;
        return decision;
    }

    MemoryStats current_stats = telemetry_ ? telemetry_->sample() : MemoryStats{};
    double current_time = current_stats.timestamp_sec;

    double elapsed_sec = 0.1;
    double delta_rss = 0.0;

    if (!first_eval_) {
        elapsed_sec = current_time - last_eval_time_;
        delta_rss = static_cast<double>(current_stats.process_rss_bytes) - static_cast<double>(last_eval_rss_);
    } else {
        first_eval_ = false;
    }

    last_eval_time_ = current_time;
    last_eval_rss_ = current_stats.process_rss_bytes;

    double phi = compute_phi_score(current_stats, elapsed_sec, delta_rss);
    last_phi_ = phi;
    decision.phi_score = phi;

    Regime target_regime = current_regime_;

    if (config_.policy == ArbitrationPolicy::PREDICTIVE_COST_BENEFIT) {
        double optimal_utility = 0.0;
        target_regime = arbitrate_predictive_cost_benefit(
            current_stats, elapsed_sec, delta_rss, current_seq_len, optimal_utility
        );
        decision.selected_utility_cost = optimal_utility;
    } else if (config_.policy == ArbitrationPolicy::STATIC_KV_ONLY) {
        uint32_t trigger_limit = config_.attention_sink_tokens + config_.recent_window_tokens;
        if (config_.target_working_set_tokens > 0 && config_.target_working_set_tokens < trigger_limit) {
            trigger_limit = config_.target_working_set_tokens;
        }
        if (current_seq_len > trigger_limit) {
            target_regime = Regime::REGIME_2_BALANCED_OFFLOAD;
        } else {
            target_regime = Regime::REGIME_1_HIGH_THROUGHPUT;
        }
    } else {
        // THRESHOLD_HYSTERESIS
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
    }

    apply_regime_transition(target_regime, current_seq_len, seq_rm, seq_add, decision);
    return decision;
}

void AMUQuantController::apply_regime_transition(
    Regime target,
    uint32_t current_seq_len,
    const KVSeqRmFunc& seq_rm,
    const KVSeqAddFunc& seq_add,
    StepDecision& decision
) {
    current_regime_ = target;
    decision.regime = target;

    uint32_t n_heads = (config_.kv_heads > 0) ? config_.kv_heads : 8;
    uint32_t d_head = (config_.head_dim > 0) ? config_.head_dim : 128;
    uint32_t elem_bytes = (config_.kv_bytes_per_element > 0) ? config_.kv_bytes_per_element : 2;
    size_t bytes_per_token = 2ULL * config_.num_layers * n_heads * d_head * elem_bytes;
    if (bytes_per_token == 0) bytes_per_token = 36864;

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

            if (kv_surgeon_) {
                uint32_t eff_window = config_.recent_window_tokens;
                if (config_.target_working_set_tokens > config_.attention_sink_tokens) {
                    uint32_t budget_window = config_.target_working_set_tokens - config_.attention_sink_tokens;
                    if (budget_window < eff_window) {
                        eff_window = budget_window;
                    }
                }
                auto range = kv_surgeon_->plan_eviction(
                    current_seq_len,
                    config_.attention_sink_tokens,
                    eff_window,
                    config_.min_eviction_batch
                );
                if (range.valid()) {
                    uint32_t count = static_cast<uint32_t>(range.p_end - range.p_start);
                    last_estimated_savings_bytes_ = static_cast<uint64_t>(
                        static_cast<double>(count * bytes_per_token) * adaptive_calib_factor_
                    );
                    bool ok = kv_surgeon_->execute_eviction(range, seq_rm, seq_add);
                    decision.kv_pruned = ok;
                    if (ok) {
                        record_actual_reclaimed(count * bytes_per_token);
                    }
                }
            }
            break;
        }

        case Regime::REGIME_3_ULTRA_LOW_SURVIVAL: {
            prefetch_depth_ = 0; // Strict single-layer execution
            aggressive_evict_ = true;
            decision.prefetch_depth = 0;
            decision.aggressive_evict = true;

            // Evict KV tokens under compressed recent window
            if (kv_surgeon_) {
                auto range = kv_surgeon_->plan_eviction(
                    current_seq_len,
                    config_.attention_sink_tokens,
                    config_.min_recent_window_tokens,
                    config_.min_eviction_batch
                );
                if (range.valid()) {
                    uint32_t count = static_cast<uint32_t>(range.p_end - range.p_start);
                    last_estimated_savings_bytes_ = static_cast<uint64_t>(
                        static_cast<double>(count * bytes_per_token) * adaptive_calib_factor_
                    );
                    bool ok = kv_surgeon_->execute_eviction(range, seq_rm, seq_add);
                    decision.kv_pruned = ok;
                    if (ok) {
                        record_actual_reclaimed(count * bytes_per_token);
                    }
                }
            }
            break;
        }
    }

    decision.estimated_savings_bytes = last_estimated_savings_bytes_;
    decision.actual_reclaimed_bytes = last_actual_reclaimed_bytes_;
    decision.estimation_delta_bytes = last_estimation_delta_bytes_;
    decision.calibration_factor = adaptive_calib_factor_;
}

void AMUQuantController::record_actual_reclaimed(uint64_t actual_reclaimed_bytes) {
    last_actual_reclaimed_bytes_ = actual_reclaimed_bytes;
    last_estimation_delta_bytes_ = static_cast<int64_t>(last_estimated_savings_bytes_) - static_cast<int64_t>(actual_reclaimed_bytes);

    if (last_estimated_savings_bytes_ > 0) {
        double ratio = static_cast<double>(actual_reclaimed_bytes) / static_cast<double>(last_estimated_savings_bytes_);
        adaptive_calib_factor_ = (0.85 * adaptive_calib_factor_) + (0.15 * ratio);
        adaptive_calib_factor_ = std::max(0.5, std::min(2.0, adaptive_calib_factor_));
    }
}

void AMUQuantController::reset() {
    current_regime_ = Regime::REGIME_1_HIGH_THROUGHPUT;
    last_phi_ = 0.0;
    prefetch_depth_ = 3;
    aggressive_evict_ = false;
    last_eval_time_ = 0.0;
    last_eval_rss_ = 0;
    first_eval_ = true;
    adaptive_calib_factor_ = 1.0;
    last_estimated_savings_bytes_ = 0;
    last_actual_reclaimed_bytes_ = 0;
    last_estimation_delta_bytes_ = 0;
}

} // namespace amuquant
