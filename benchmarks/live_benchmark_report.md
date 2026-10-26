# AMU-QUANT Live Empirical Benchmark Report: Multi-Scenario Evaluation

- **Model Target**: `models/qwen2.5-3b-instruct-q4_k_m.gguf`
- **Architecture**: 36 Layers, Dim 2048
- **Target Generation Horizon**: 64 tokens

## 1. Scenario 1: Nominal 3-Way Methodological Parity

| Evaluation Metric | Baseline 1 (Raw Vanilla) | Baseline 2 (With Embeddings) | AMU-QUANT Co-Adaptive Engine |
| :--- | :--- | :--- | :--- |
| **Target Horizon Completed** | Yes (100%) | Yes (100%) | Yes (100%) |
| **Tokens Generated** | 64/64 | 64/64 | 64/64 |
| **Peak Physical RSS** | 171 MB | 208 MB | 201 MB |
| **Empirical Process RSS Delta** | 4 MB | 32 MB | 38 MB |
| **Inference Throughput** | 22.67 tok/s | 23.10 tok/s | 25.46 tok/s |
| **Logical KV Capacity Reclaimed** | 0 KB | 0 KB | **2448 KB** |
| **Context Tokens Safely Archived** | 0 | 0 | **68** |
| **Tier 1: Index Sanity Match** | N/A | N/A | **0.8605** |
| **Tier 2: Factual Needle Recall** | N/A | N/A | **100% Match (Recall@3)** |
| **SIMD Retrieval Latency** | N/A | N/A | **0.586 ms** |

### Empirical Measurement Protocol, Repeatability & Variance Analysis

> [!NOTE]
> **Run-to-Run Variance on Apple Silicon Metal GPU**:
> Single-run generation speeds naturally fluctuate across runs (typically 22.7 tok/s to 25.6 tok/s) due to:
> 1. macOS Dynamic Power & Thermal Throttling: Clock frequencies adjust dynamically depending on SoC temperature.
> 2. Unified Memory Bus Contention: Host background daemons sharing unified RAM bandwidth.
> 3. Metal GPU Pipeline & Shader Cache: Warm cache states execute faster than initial cold invocations.

> [!NOTE]
> **Accounting Notice on Physical vs Logical Memory**:
> `Logical KV Capacity Reclaimed` reflects uncompressed sequence slots pruned from the attention buffer. `llama.cpp` pre-allocates a contiguous ring buffer pool for sequence positions, which means KV eviction prevents buffer expansion and memory exhaustion (bounded physical RSS) rather than issuing immediate heap deallocation calls (`munmap`) to the OS.

## 2. Scenario 2: Multi-Turn Autonomous Agentic Workflow (ReAct Tool Loop)

| Agent Workflow Metric | Unmanaged Baseline Agent | AMU-QUANT Co-Adaptive Agent |
| :--- | :--- | :--- |
| **Total ReAct Turns Completed** | 4/4 Turns | **4/4 Turns (100%)** |
| **Total Ingested Tool Tokens** | 293 tokens | **293 tokens** |
| **Total Generated Agent Tokens** | 64 tokens | **64 tokens** |
| **Final Physical KV Cache Slots** | 357 slots (Unbounded Expansion) | **49 slots (Strictly Bounded!)** |
| **Logical KV Capacity Reclaimed** | 0 KB | **11088 KB** |
| **Context Tokens Archived to turbovec** | 0 | **308 tokens** |
| **Turn 1 Master Key Recall** | In Attention Buffer | **100% Match (Recall@3)** |
| **Episodic SIMD Retrieval Latency** | N/A | **1.752 ms** |

## 3. Scenario 3A: Deterministic FSM State Transition Test (Mock Telemetry)

| Deterministic Evaluation Metric | AMU-QUANT State Machine Engine |
| :--- | :--- |
| **Zero-OOM Horizon Completed** | **Yes (100% Guaranteed)** |
| **Injected Telemetry Profile** | Available RAM: 350 MB (< 600 MB Reserve), Swap Thrashing: 1.4 GB |
| **FSM Operational Transition** | **REGIME 3 (Ultra-Low Survival Mode)** |
| **Dynamic Requantization** | **In-Place 4-bit -> 2-bit Lloyd-Max Downscaling** |
| **Logical KV Capacity Reclaimed** | **2736 KB** |
| **Pre-Shock Recall (4-bit)** | **0.8605 (Latency: 0.2346 ms)** |
| **Post-Shock Recall (2-bit)** | **0.6341 (Latency: 0.4024 ms)** |
| **Tier 2 Factual Needle Recovery** | **100% Preserved** |
| **Episodic Vector Memory Footprint** | **553 KB (50% In-Place Compaction)** |

## 4. Scenario 3B: Controlled Real Host Memory Stress Test (Physical Memory Pressure)

| Real Stress Metric | AMU-QUANT Live Host Performance |
| :--- | :--- |
| **Zero-OOM Horizon Completed** | **Yes (100% Completion)** |
| **Host Physical Memory Allocated** | **2048 MB Dirty Pages** |
| **Host Telemetry Sensing Mode** | Live macOS Kernel (`mach_host_self`, `host_statistics64`) |
| **Autonomous FSM Transition** | **REGIME 3 (Ultra-Low Survival Mode)** |
| **Dynamic Requantization** | **Autonomous In-Place 4-bit -> 2-bit Compaction** |
| **Logical KV Capacity Reclaimed** | **2592 KB** |
| **Tier 2 Factual Needle Recovery** | **100% Preserved** |
| **Process RSS Before / After** | 181 MB / 2268 MB (Delta: 2087 MB) |

## 5. Scenario 4: Runtime Arbitration Policy Ablation Matrix

| Policy Configuration | Target Throughput | Peak Physical RSS | KV Reclaimed | Final Regime |
| :--- | :--- | :--- | :--- | :--- |
| **Predictive Cost-Benefit (PCBM)** | 25.79 tok/s | 2269 MB | **1440 KB** | REGIME_2_BALANCED_OFFLOAD |
| **Static Threshold Hysteresis** | 24.49 tok/s | 2269 MB | **2016 KB** | REGIME_3_ULTRA_LOW_SURVIVAL |
| **Static KV Eviction Only** | 23.25 tok/s | 2269 MB | **1440 KB** | REGIME_2_BALANCED_OFFLOAD |

## 6. Scenario 6: Extended Horizon (256 Tokens) & Multi-Needle Retrieval

| Extended Horizon Metric | Measurement | Impact Analysis |
| :--- | :--- | :--- |
| **Target Horizon** | 256/256 tokens | 4x Nominal Horizon |
| **Inference Throughput** | 22.41 tok/s | Sustained Decoding Velocity |
| **Peak Physical RSS** | 2311 MB | Bounded Memory Ceiling |
| **Logical KV Capacity Reclaimed** | **11376 KB** | Buffer Overflow Prevented |
| **Context Tokens Archived** | **316 tokens** | In-Memory Episodic Storage |
| **Final Physical KV Slots** | **66 slots** | Attention Bounded to O(1) |

### Multi-Needle Factual Retrieval Matrix (Cross-Temporal Evaluation)

| Target Needle | Context Position | Recall@1 | Recall@3 | Cosine Score | Retrieval Latency |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Needle 1 (Initialization Passkey)** (`AegisOmega-90412`) | Token 7 | 100% Match | 100% Match | 0.8273 | 3.252 ms |
| **Needle 2 (Security Certificate)** (`SecCert-88319`) | Token 43 | 100% Match | 100% Match | 0.9257 | 2.455 ms |
| **Needle 3 (Database Nonce)** (`NonceDelta-44102`) | Token 90 | 100% Match | 100% Match | 0.9051 | 2.578 ms |

## 7. Scenario 7: Theoretical & Empirical Long-Context Scalability ($N \ge 512$ Tokens)

| Token Horizon ($N$) | Baseline Unmanaged FP16 KV | AMU-QUANT Physical KV | AMU-QUANT 2-bit Index | Memory Reduction Ratio |
| :--- | :--- | :--- | :--- | :--- |
| **64 tokens** | 8.38 MB | 4.71 MB | 0.03 MB | **1.8x reduction** |
| **256 tokens** | 33.55 MB | 4.71 MB | 0.22 MB | **6.8x reduction** |
| **1,024 tokens** | 134.21 MB | 4.71 MB | 0.99 MB | **23.5x reduction** |
| **4,096 tokens** | 536.87 MB | 4.71 MB | 4.06 MB | **61.2x reduction** |
| **8,192 tokens** | 1,073.74 MB (1.07 GB) | 4.71 MB | 8.16 MB | **83.4x reduction (Zero-OOM)** |

Generated automatically by `live_benchmark` on Apple Silicon / Metal.
