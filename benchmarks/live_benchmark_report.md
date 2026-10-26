# AMU-QUANT Live Empirical Benchmark Report: Multi-Scenario Evaluation

- **Model Target**: `models/Meta-Llama-3-8B-Instruct-Q2_K.gguf`
- **Architecture**: 32 Layers, Dim 4096
- **Target Generation Horizon**: 64 tokens

## 1. Scenario 1: Nominal 3-Way Methodological Parity

| Evaluation Metric | Baseline 1 (Raw Vanilla) | Baseline 2 (With Embeddings) | AMU-QUANT Co-Adaptive Engine |
| :--- | :--- | :--- | :--- |
| **Target Horizon Completed** | Yes (100%) | Yes (100%) | Yes (100%) |
| **Tokens Generated** | 64/64 | 64/64 | 64/64 |
| **Peak Physical RSS** | 354 MB | 379 MB | 381 MB |
| **Inference Throughput** | 12.81 tok/s | 13.77 tok/s | 11.85 tok/s |
| **KV Memory Reclaimed** | 0 KB | 0 KB | 8704 KB |
| **Tokens Safely Archived** | 0 | 0 | 68 |
| **Semantic Recall Score** | N/A | N/A | 0.8468 |
| **SIMD Retrieval Latency** | N/A | N/A | 2.019 ms |

## 2. Scenario 2: Acute Host Memory Starvation & Survival Mode Stress Test

| Stress Evaluation Metric | Unmanaged Baseline 2 | AMU-QUANT Survival Engine |
| :--- | :--- | :--- |
| **Zero-OOM Survival** | High OOM Risk (Memory Unbounded) | **100% Guaranteed Completion** |
| **Host Stress Event** | N/A | Injected at Token 16 (Available RAM: 350MB, Swap: 1.4GB) |
| **FSM Operational Regime** | None (Static) | **REGIME 3 (Ultra-Low Survival Mode)** |
| **Dynamic Requantization** | None (FP16 only) | **In-Place 4-bit -> 2-bit Compaction** |
| **Layer Prefetch Depth** | Uncontrolled | **0 (Strict Single-Layer Execution)** |
| **Physical KV Reclaimed** | 0 KB | **9728 KB (9 MB)** |
| **Pre-Shock Recall (4-bit)** | N/A | **0.8468 (Latency: 0.4898 ms)** |
| **Post-Shock Recall (2-bit)** | N/A | **0.8468 (Latency: 0.5893 ms)** |
| **Episodic Memory Size** | N/A | **1103 KB** |

## 3. Scenario 3: Theoretical & Empirical Long-Context Scalability ($N \ge 512$ Tokens)

| Token Horizon ($N$) | Baseline Unmanaged FP16 KV | AMU-QUANT Physical KV | AMU-QUANT 2-bit Index | Memory Reduction Ratio |
| :--- | :--- | :--- | :--- | :--- |
| **64 tokens** | 8.38 MB | 4.71 MB | 0.03 MB | **1.8x reduction** |
| **256 tokens** | 33.55 MB | 4.71 MB | 0.22 MB | **6.8x reduction** |
| **1,024 tokens** | 134.21 MB | 4.71 MB | 0.99 MB | **23.5x reduction** |
| **4,096 tokens** | 536.87 MB | 4.71 MB | 4.06 MB | **61.2x reduction** |
| **8,192 tokens** | 1,073.74 MB (1.07 GB) | 4.71 MB | 8.16 MB | **83.4x reduction (Zero-OOM)** |

Generated automatically by `live_benchmark` on Apple Silicon / Metal.
