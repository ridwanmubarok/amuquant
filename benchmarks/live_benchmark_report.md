# AMU-QUANT Live Empirical Benchmark Report: Multi-Scenario Evaluation

- **Model Target**: `models/qwen2.5-3b-instruct-q4_k_m.gguf`
- **Architecture**: 36 Layers, Dim 2048
- **Target Generation Horizon**: 64 tokens

## 1. Scenario 1: Nominal 3-Way Methodological Parity

| Evaluation Metric | Baseline 1 (Raw Vanilla) | Baseline 2 (With Embeddings) | AMU-QUANT Co-Adaptive Engine |
| :--- | :--- | :--- | :--- |
| **Target Horizon Completed** | Yes (100%) | Yes (100%) | Yes (100%) |
| **Tokens Generated** | 64/64 | 64/64 | 64/64 |
| **Peak Physical RSS** | 168 MB | 193 MB | 194 MB |
| **Empirical Process RSS Delta** | -11 MB | 39 MB | 38 MB |
| **Inference Throughput** | 21.78 tok/s | 23.65 tok/s | 24.22 tok/s |
| **Logical KV Capacity Reclaimed** | 0 KB | 0 KB | **2448 KB** |
| **Context Tokens Safely Archived** | 0 | 0 | **68** |
| **Tier 1: Index Sanity Match** | N/A | N/A | **0.8605** |
| **Tier 2: Factual Needle Recall** | N/A | N/A | **100% Match (Recall@3)** |
| **SIMD Retrieval Latency** | N/A | N/A | **0.577 ms** |

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
| **Episodic SIMD Retrieval Latency** | N/A | **3.069 ms** |

## 3. Scenario 3A: Deterministic FSM State Transition Test (Mock Telemetry)

| Deterministic Evaluation Metric | AMU-QUANT State Machine Engine |
| :--- | :--- |
| **Zero-OOM Horizon Completed** | **Yes (100% Guaranteed)** |
| **Injected Telemetry Profile** | Available RAM: 350 MB (< 600 MB Reserve), Swap Thrashing: 1.4 GB |
| **FSM Operational Transition** | **REGIME 3 (Ultra-Low Survival Mode)** |
| **Dynamic Requantization** | **In-Place 4-bit -> 2-bit Lloyd-Max Downscaling** |
| **Logical KV Capacity Reclaimed** | **2736 KB** |
| **Pre-Shock Recall (4-bit)** | **0.8605 (Latency: 0.0999 ms)** |
| **Post-Shock Recall (2-bit)** | **0.6341 (Latency: 0.3036 ms)** |
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
| **Process RSS Before / After** | 175 MB / 214 MB (Delta: 39 MB) |

## 5. Scenario 4: Runtime Arbitration Policy Ablation Matrix

| Policy Configuration | Target Throughput | Peak Physical RSS | KV Reclaimed | Retrieval (Recall@1 / Cosine) | P95 Latency | Final Regime |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Predictive Cost-Benefit (PCBM)** | 22.93 tok/s | 214 MB | **1440 KB** | 100% (0.8604) | 50.9 ms | REGIME_2_BALANCED_OFFLOAD |
| **Static Threshold Hysteresis** | 24.24 tok/s | 214 MB | **2016 KB** | 100% (0.6601) | 50.3 ms | REGIME_3_ULTRA_LOW_SURVIVAL |
| **Static KV Eviction Only** | 22.73 tok/s | 214 MB | **1440 KB** | 100% (0.8604) | 54.9 ms | REGIME_2_BALANCED_OFFLOAD |

## 6. Scenario 6: Extended Horizon (256 Tokens) & Multi-Needle Retrieval

| Extended Horizon Metric | Measurement | Impact Analysis |
| :--- | :--- | :--- |
| **Target Horizon** | 256/256 tokens | 4x Nominal Horizon |
| **Inference Throughput** | 22.90 tok/s | Sustained Decoding Velocity |
| **Peak Physical RSS** | 259 MB | Bounded Memory Ceiling |
| **Logical KV Capacity Reclaimed** | **11376 KB** | Buffer Overflow Prevented |
| **Context Tokens Archived** | **316 tokens** | In-Memory Episodic Storage |
| **Final Physical KV Slots** | **66 slots** | Attention Bounded to O(1) |

### Multi-Needle Factual Retrieval Matrix (Cross-Temporal Evaluation)

| Target Needle | Context Position | Recall@1 | Recall@3 | Cosine Score | Retrieval Latency |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Needle 1 (Initialization Passkey)** (`AegisOmega-90412`) | Token 7 | 100% Match | 100% Match | 0.8273 | 1.189 ms |
| **Needle 2 (Security Certificate)** (`SecCert-88319`) | Token 43 | 100% Match | 100% Match | 0.9257 | 1.394 ms |
| **Needle 3 (Database Nonce)** (`NonceDelta-44102`) | Token 90 | 100% Match | 100% Match | 0.9051 | 1.123 ms |

## 7. Scenario 7: Live Long-Context Horizon Benchmark (1,024 Tokens) [Empirically Verified]

| Evaluation Metric | AMU-QUANT Co-Adaptive Engine | Operational Significance |
| :--- | :--- | :--- |
| **Target Horizon** | 1024/1024 tokens | 16x Nominal Horizon |
| **Inference Throughput** | 22.86 tok/s | Sustained Decoding Velocity |
| **Peak Physical RSS** | 201 MB | Strictly Bounded Memory Ceiling |
| **Logical KV Capacity Reclaimed** | **35856 KB (35 MB)** | Buffer Overflow Prevented |
| **Context Tokens Archived** | **996 tokens** | In-Memory Episodic Storage |
| **Final Physical KV Slots** | **61 slots** | Attention Bounded to O(1) (<= 66 slots) |

### Deep Long-Context Needle Retrieval Matrix (Intervening Context: 1,000+ Tokens)

| Target Needle | Context Position | Recall@1 | Recall@3 | Cosine Score | Retrieval Latency |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Needle 1 (Deep Long-Context Key)** (`AegisOmega-90412`) | Token 11 | 100% Match | 100% Match | 0.8083 | 4.178 ms |

## 8. Scenario 8: Analytical Mathematical Projection of Scaling (Extrapolated from the Empirical 66-Slot Clamp in Scenarios 6 & 7)

> [!NOTE]
> **Methodological Demarcation Note**: Horizons 64, 256, and 1,024 tokens are empirically executed and verified in Scenarios 1, 6, and 7. Horizons 4,096 and 8,192 tokens represent analytical mathematical projections based on the measured 66-slot physical attention clamp.

| Token Horizon ($N$) | Evaluation Mode | Baseline Unmanaged FP16 KV | AMU-QUANT Physical KV | AMU-QUANT 2-bit Index | Memory Reduction Ratio |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **64 tokens** | Empirically Verified (Scenario 1) | 8.38 MB | 4.71 MB | 0.03 MB | **1.8x reduction** |
| **256 tokens** | Empirically Verified (Scenario 6) | 33.55 MB | 4.71 MB | 0.22 MB | **6.8x reduction** |
| **1,024 tokens** | Empirically Verified (Scenario 7) | 134.21 MB | 4.71 MB | 0.99 MB | **23.5x reduction** |
| **4,096 tokens** | Analytical Projection | 536.87 MB | 4.71 MB | 4.06 MB | **61.2x reduction** |
| **8,192 tokens** | Analytical Projection | 1,073.74 MB (1.07 GB) | 4.71 MB | 8.16 MB | **83.4x reduction (Bounded Memory Footprint)** |

## 9. Scenario 9: Factual QA & Hallucination Ablation Benchmark (4-Way Matrix: A, B, C, D)

| Configuration | Evaluation Mode | Working KV Slots | Factual Accuracy | Hallucination Status | Generated Output Snippet |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Config A** | Unmanaged Baseline (No Eviction) | 431 slots (O(N) Unbounded) | **50% (2/4)** | YES (Lost Evicted Context) | `Berdasarkan data konfigurasi yang telah disediakan:  - Na...` |
| **Config B** | AMU-QUANT (Eviction Disabled) | 431 slots (O(N) Unbounded) | **50% (2/4)** | YES (Lost Evicted Context) | `Berdasarkan data konfigurasi yang telah disediakan:  - Na...` |
| **Config C** | Eviction Without Context Reconstruction | 165 slots (O(N) Unbounded) | **0% (0/4)** | YES (Lost Evicted Context) | `Maaf, tetapi data konfigurasi yang Anda minta tidak diber...` |
| **Config D** | Eviction With Query-Driven Retrieval-to-Prompt | 290 slots (O(N) Unbounded) | **100% (4/4)** | NO (Context Restored) | `Nama proyek adalah Aegis, nomor versi rilis adalah 90412,...` |

### Detailed Entity Match Breakdown Across Configurations

| Entity Target | Ground Truth | Config A | Config B | Config C (Eviction Only) | Config D (AMU-QUANT Dual-Memory) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Project Name** | `Aegis` | MATCH | MATCH | MISS | **MATCH** |
| **Version Number** | `90412` | MATCH | MATCH | MISS | **MATCH** |
| **Architect Owner** | `Ridwan` | MISS | MISS | MISS | **MATCH** |
| **Deployment Port** | `8084` | MISS | MISS | MISS | **MATCH** |

Generated automatically by `live_benchmark` on Apple Silicon / Metal.
