#!/usr/bin/env python3
"""
AMU-QUANT Telemetry & Experiment Dataset Analysis Script
Undergraduate Thesis (Tugas Akhir) Data Science Toolkit

Parses time-series memory and telemetry logs exported from AMU-QUANT benchmarks
and generates descriptive statistics, comparative matrix tables, and publication-ready
visualizations for the thesis report.
"""

import sys
import os
import csv
import argparse
from collections import defaultdict

def parse_args():
    parser = argparse.ArgumentParser(description="Analyze AMU-QUANT benchmark telemetry CSV datasets.")
    parser.add_argument("--input", "-i", default="benchmarks/data/standalone_benchmark_results.csv",
                        help="Path to the input CSV dataset file")
    parser.add_argument("--plot", "-p", default="benchmarks/data/telemetry_analysis.png",
                        help="Path to save generated analysis plot (if matplotlib is available)")
    return parser.parse_args()

def load_dataset(csv_path):
    if not os.path.exists(csv_path):
        print(f"[ERROR] File not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    records = []
    with open(csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                record = {
                    "timestamp_ms": int(row.get("timestamp_ms", 0)),
                    "step": int(row.get("step", 0)),
                    "policy": row.get("policy", "Unknown").strip('"'),
                    "process_rss_mb": float(row.get("process_rss_mb", 0.0)),
                    "physical_footprint_mb": float(row.get("physical_footprint_mb", 0.0)),
                    "available_ram_mb": float(row.get("available_ram_mb", 0.0)),
                    "swap_used_mb": float(row.get("swap_used_mb", 0.0)),
                    "context_length": int(row.get("context_length", 0)),
                    "active_kv_slots": int(row.get("active_kv_slots", 0)),
                    "pruned_kv_slots": int(row.get("pruned_kv_slots", 0)),
                    "kv_logical_kb": float(row.get("kv_logical_kb", 0.0)),
                    "model_resident_mb": float(row.get("model_resident_mb", 0.0)),
                    "regime": row.get("regime", "Unknown").strip('"'),
                    "phi_score": float(row.get("phi_score", 0.0)),
                    "inter_token_latency_ms": float(row.get("inter_token_latency_ms", 0.0)),
                    "throughput_tps": float(row.get("throughput_tps", 0.0)),
                    "grounding_status": row.get("grounding_status", "N/A").strip('"')
                }
                records.append(record)
            except ValueError as e:
                continue
    return records

def print_summary_statistics(records):
    by_policy = defaultdict(list)
    for r in records:
        by_policy[r["policy"]].append(r)

    print("=" * 110)
    print("  AMU-QUANT EXPERIMENTAL TELEMETRY ANALYSIS (DATA SCIENCE THESIS PIPELINE)")
    print("=" * 110)
    print(f"Total Dataset Observations: {len(records)} time-series data points\n")

    header = f"{'Policy':<36} | {'Samples':<8} | {'Peak KV (KB)':<14} | {'Active Slots':<14} | {'Peak Phi':<10} | {'Mean Lat (ms)':<14} | {'Throughput (tps)':<16}"
    print(header)
    print("-" * 110)

    for policy, rows in by_policy.items():
        peak_kv = max(r["kv_logical_kb"] for r in rows) if rows else 0
        final_active = rows[-1]["active_kv_slots"] if rows else 0
        peak_phi = max(r["phi_score"] for r in rows) if rows else 0.0
        mean_lat = sum(r["inter_token_latency_ms"] for r in rows) / len(rows) if rows else 0.0
        mean_tps = sum(r["throughput_tps"] for r in rows) / len(rows) if rows else 0.0

        print(f"{policy:<36} | {len(rows):<8} | {peak_kv:<14.1f} | {final_active:<14} | {peak_phi:<10.3f} | {mean_lat:<14.2f} | {mean_tps:<16.2f}")

    print("=" * 110)
    print("\nRegime Distribution Analysis:")
    for policy, rows in by_policy.items():
        regime_counts = defaultdict(int)
        for r in rows:
            regime_counts[r["regime"]] += 1
        total = len(rows)
        reg_breakdown = ", ".join([f"{k}: {v} ({v*100.0/total:.1f}%)" for k, v in regime_counts.items()])
        print(f"  * {policy}: {reg_breakdown}")
    print()

def generate_visualization(records, plot_path):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[INFO] matplotlib is not installed. Skipping graphical plot generation.")
        return

    by_policy = defaultdict(list)
    for r in records:
        by_policy[r["policy"]].append(r)

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=False)
    fig.suptitle("AMU-QUANT Dynamic Memory Management & Telemetry Time-Series", fontsize=14, fontweight="bold")

    colors = {
        "llama.cpp Baseline (Unmanaged)": "#e74c3c",
        "Static Threshold Controller": "#f39c12",
        "Predictive Cost-Benefit Controller": "#2ecc71"
    }

    # Plot 1: KV Cache Logical Size (KB)
    ax1 = axes[0]
    for policy, rows in by_policy.items():
        steps = [r["step"] for r in rows]
        kv_kb = [r["kv_logical_kb"] for r in rows]
        c = colors.get(policy, "#3498db")
        ax1.plot(steps, kv_kb, label=policy, color=c, linewidth=1.8)
    ax1.set_ylabel("KV Cache (KB)")
    ax1.set_title("1. Active KV Cache Working Set Expansion vs O(1) Bounded Surgery")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper left")

    # Plot 2: Predictive Pressure Metric Phi_t
    ax2 = axes[1]
    for policy, rows in by_policy.items():
        steps = [r["step"] for r in rows]
        phi = [r["phi_score"] for r in rows]
        c = colors.get(policy, "#3498db")
        ax2.plot(steps, phi, label=policy, color=c, linewidth=1.8)
    ax2.axhline(y=0.65, color="orange", linestyle=":", label="Theta Low (0.65)")
    ax2.axhline(y=0.85, color="red", linestyle=":", label="Theta High (0.85)")
    ax2.set_ylabel("Phi Score (Pressure)")
    ax2.set_title("2. Predictive Memory Pressure Metric Phi_t Evolution")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="upper left")

    # Plot 3: Host Available RAM (MB)
    ax3 = axes[2]
    for policy, rows in by_policy.items():
        steps = [r["step"] for r in rows]
        avail = [r["available_ram_mb"] for r in rows]
        c = colors.get(policy, "#3498db")
        ax3.plot(steps, avail, label=policy, color=c, linewidth=1.8)
    ax3.set_xlabel("Token Sequence Step")
    ax3.set_ylabel("Available RAM (MB)")
    ax3.set_title("3. Host Available RAM Depletion & Safety Headroom Protection")
    ax3.grid(True, linestyle="--", alpha=0.5)
    ax3.legend(loc="lower left")

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(plot_path)), exist_ok=True)
    plt.savefig(plot_path, dpi=200)
    print(f"[INFO] High-resolution visualization exported to: {plot_path}")

def main():
    args = parse_args()
    records = load_dataset(args.input)
    if not records:
        print("[WARNING] No records found in dataset.")
        return
    print_summary_statistics(records)
    generate_visualization(records, args.plot)

if __name__ == "__main__":
    main()
