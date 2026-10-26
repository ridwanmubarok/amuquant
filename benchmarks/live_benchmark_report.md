# AMU-QUANT Live Empirical Benchmark Report: Multi-Scenario Evaluation

- **Model Target**: `models/Meta-Llama-3-8B-Instruct-Q2_K.gguf`
- **Architecture**: 32 Layers, Dim 4096
- **Target Generation Horizon**: 64 tokens

## 1. Scenario 1: Nominal 3-Way Methodological Parity

| Evaluation Metric | Baseline 1 (Raw Vanilla) | Baseline 2 (With Embeddings) | AMU-QUANT Co-Adaptive Engine |
| :--- | :--- | :--- | :--- |
| **Target Horizon Completed** | Yes (100%) | Yes (100%) | Yes (100%) |
| **Tokens Generated** | 64/64 | 64/64 | 64/64 |
| **Peak Physical RSS** | 354 MB | 386 MB | 389 MB |
| **Empirical Process RSS Delta** | 7 MB | 35 MB | 36 MB |
| **Inference Throughput** | 13.47 tok/s | 15.73 tok/s | 14.47 tok/s |
| **Logical KV Capacity Reclaimed** | 0 KB | 0 KB | **8704 KB** |
| **Context Tokens Safely Archived** | 0 | 0 | **68** |
| **Tier 1: Index Sanity Match** | N/A | N/A | **0.9759** |
| **Tier 2: Factual Needle Recall** | N/A | N/A | **100% Match (Recall@3)** |
| **SIMD Retrieval Latency** | N/A | N/A | **1.308 ms** |

> [!NOTE]
> **Accounting Notice on Physical vs Logical Memory**:
> `Logical KV Capacity Reclaimed` reflects uncompressed sequence slots pruned from the attention buffer. > `llama.cpp` pre-allocates a contiguous ring buffer pool for sequence positions, which means KV eviction prevents > buffer expansion and memory exhaustion (bounded physical RSS) rather than issuing immediate heap deallocation calls to the OS.

## 2. Scenario 2A: Deterministic FSM State Transition Test (Mock Telemetry)

| Deterministic Evaluation Metric | AMU-QUANT State Machine Engine |
| :--- | :--- |
| **Zero-OOM Horizon Completed** | **Yes (100% Guaranteed)** |
| **Injected Telemetry Profile** | Available RAM: 350 MB (< 600 MB Reserve), Swap Thrashing: 1.4 GB |
| **FSM Operational Transition** | **REGIME 3 (Ultra-Low Survival Mode)** |
| **Dynamic Requantization** | **In-Place 4-bit -> 2-bit Lloyd-Max Downscaling** |
| **Logical KV Capacity Reclaimed** | **11776 KB** |
| **Pre-Shock Recall (4-bit)** | **0.9759 (Latency: 0.3215 ms)** |
| **Post-Shock Recall (2-bit)** | **0.8029 (Latency: 1.7669 ms)** |
| **Tier 2 Factual Needle Recovery** | **100% Preserved** |
| **Episodic Vector Memory Footprint** | **1119 KB (50% In-Place Compaction)** |

## 3. Scenario 2B: Controlled Real Host Memory Stress Test (Physical Memory Pressure)

| Real Stress Metric | AMU-QUANT Live Host Performance |
| :--- | :--- |
| **Zero-OOM Horizon Completed** | **Yes (100% Completion)** |
| **Host Physical Memory Allocated** | **2048 MB Dirty Pages** |
| **Host Telemetry Sensing Mode** | Live macOS Kernel (`mach_host_self`, `host_statistics64`) |
| **Autonomous FSM Transition** | **REGIME 3 (Ultra-Low Survival Mode)** |
| **Dynamic Requantization** | **Autonomous In-Place 4-bit -> 2-bit Compaction** |
| **Logical KV Capacity Reclaimed** | **9216 KB** |
| **Tier 2 Factual Needle Recovery** | **100% Preserved** |
| **Process RSS Before / After** | 355 MB / 2439 MB (Delta: 2084 MB) |

## 4. Scenario 3: Theoretical & Empirical Long-Context Scalability ($N \ge 512$ Tokens)

| Token Horizon ($N$) | Baseline Unmanaged FP16 KV | AMU-QUANT Physical KV | AMU-QUANT 2-bit Index | Memory Reduction Ratio |
| :--- | :--- | :--- | :--- | :--- |
| **64 tokens** | 8.38 MB | 4.71 MB | 0.03 MB | **1.8x reduction** |
| **256 tokens** | 33.55 MB | 4.71 MB | 0.22 MB | **6.8x reduction** |
| **1,024 tokens** | 134.21 MB | 4.71 MB | 0.99 MB | **23.5x reduction** |
| **4,096 tokens** | 536.87 MB | 4.71 MB | 4.06 MB | **61.2x reduction** |
| **8,192 tokens** | 1,073.74 MB (1.07 GB) | 4.71 MB | 8.16 MB | **83.4x reduction (Zero-OOM)** |

Generated automatically by `live_benchmark` on Apple Silicon / Metal.
