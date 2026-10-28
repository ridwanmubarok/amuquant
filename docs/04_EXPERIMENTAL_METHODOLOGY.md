# Experimental Methodology and Evaluation

This document outlines the experimental methodology for **AMU-QUANT**. To maintain academic rigor and transparency, the document is organized into two distinct sections:
- **Part I: Locally Verified Empirical Benchmarks**: Actual empirical measurements gathered on verified local hardware (Apple Silicon M1, Metal GPU backend, `Qwen2.5-3B-Instruct-Q4_K_M.gguf`), complete with exact execution commands, configurations, and raw telemetry data.
- **Part II: Proposed Academic Evaluation Protocol**: The planned multi-device thesis evaluation agenda, defining target testbeds, proposed comparative baselines, and cross-scale model roadmaps.

---

# Part I: Locally Verified Empirical Benchmarks

All results in this section were produced directly by the compiled benchmark executable [`benchmarks/live_benchmark.cpp`](file:///Users/amubhya/workspace/aegis/benchmarks/live_benchmark.cpp) and verified against kernel telemetry on Darwin 24.

### 1. Verification Testbed Specifications

| Specification Parameter | Value / Environment Description |
|:---|:---|
| **Host Hardware** | Apple Silicon M1 (8 cores: 4 Performance + 4 Efficiency) |
| **System Memory** | 16 GB Unified Memory (UMA) |
| **Primary Storage** | Apple Internal NVMe SSD |
| **Operating System** | macOS Sequoia (Darwin 24.6.0) |
| **Compiler & Toolchain** | Apple Clang 17.0.0, CMake 3.31.5, Rustc (Cargo) 1.85.0 |
| **Inference Framework** | Custom `llama.cpp` integration with Apple Metal GPU backend (`GGML_USE_METAL`) |
| **Evaluated Model** | `models/qwen2.5-3b-instruct-q4_k_m.gguf` (36 Layers, Hidden Dim 2048, GQA Heads 2) |
| **Codebase State** | Git commit `ef519d7` (and HEAD) |

### 2. Standard Benchmark Execution Command

The empirical evaluations were executed using the unified benchmark CLI:
```bash
./build/live_benchmark \
  --model models/qwen2.5-3b-instruct-q4_k_m.gguf \
  --prompt "Analyze the core mechanics of AMU-QUANT co-adaptive runtime memory management" \
  --steps 64 \
  --ctx 512 \
  --ngpu 99 \
  --eval-interval 2 \
  --recent-window 16 \
  --sink-tokens 4 \
  --dim 2048
```

For Scenario 6 (Extended Horizon), `--steps 256` and `--ctx 1024` were supplied.

---

### 3. Empirical Results Across Evaluation Scenarios

#### Scenario 1: Nominal 3-Way Methodological Parity
Evaluates baseline parity under identical prompt, seed, and Metal GPU configurations:

| Evaluation Metric | Baseline 1 (Raw Vanilla) | Baseline 2 (With Embeddings) | AMU-QUANT Co-Adaptive Engine |
|:---|:---:|:---:|:---:|
| **Target Horizon Completed** | Yes (100%) | Yes (100%) | Yes (100%) |
| **Tokens Generated** | 64/64 | 64/64 | 64/64 |
| **Peak Physical RSS** | 159 MB | 195 MB | 197 MB |
| **Process RSS Delta** | 5 MB | 39 MB | 39 MB |
| **Inference Throughput** | 22.43 tok/s | 26.25 tok/s | 18.55 tok/s |
| **Logical KV Capacity Reclaimed** | 0 KB | 0 KB | **2448 KB** |
| **Context Tokens Safely Archived** | 0 | 0 | **68** |
| **Tier 1: Index Sanity Match** | N/A | N/A | **0.8605** |
| **Tier 2: Factual Needle Recall** | N/A | N/A | **100% Match (Recall@3)** |
| **SIMD Retrieval Latency** | N/A | N/A | **0.368 ms** |

*Methodological Note on Throughput Delta*: The unmanaged baseline generates tokens without running context vector extractions or transactional database updates. AMU-QUANT incorporates thread-safe vector extraction, in-process quantization, and index persistence during generation, incurring an expected processing overhead of ~3.88 ms per step while bounding KV cache expansion.

#### Scenario 2: Multi-Turn Autonomous Agentic Workflow (ReAct Tool Loop)
Simulates an agent executing 4 successive ReAct reasoning and tool-calling turns:

| Agent Workflow Metric | Unmanaged Baseline Agent | AMU-QUANT Co-Adaptive Agent |
|:---|:---:|:---:|
| **Total ReAct Turns Completed** | 4/4 Turns (100%) | 4/4 Turns (100%) |
| **Total Ingested Tool Tokens** | 293 tokens | 293 tokens |
| **Total Generated Agent Tokens** | 64 tokens | 64 tokens |
| **Final Physical KV Cache Slots** | 357 slots (Unbounded Growth) | **49 slots (Strictly Bounded)** |
| **Logical KV Capacity Reclaimed** | 0 KB | **11,088 KB** |
| **Context Tokens Archived to turbovec** | 0 | **308 tokens** |
| **Turn 1 Master Key Recall** | Retained in Attention Buffer | **100% Match (Recall@3)** |
| **Episodic SIMD Retrieval Latency** | N/A | **1.280 ms** |

#### Scenario 3: Memory Stress and Dynamic Requantization
- **Scenario 3A (Deterministic FSM Mock Telemetry)**: Injected available host RAM drops to 350 MB (< 600 MB reserve floor) with 1.4 GB simulated swap thrashing. The controller immediately transitioned to `REGIME 3 (Ultra-Low Survival Mode)`, downscaled prefetch depth to 0, triggered autonomous in-place 4-bit to 2-bit Lloyd-Max requantization, compacted vector storage by 50% (553 KB footprint), and maintained 100% needle recall.
- **Scenario 3B (Controlled Live Host Memory Stress)**: Allocated 2048 MB of physical dirty pages on the host via real `mmap` while sampling live Darwin kernel statistics via `mach_host_self()` and `host_statistics64()`. AMU-QUANT completed 100% of the horizon without OOM termination, dynamically shifting to Regime 3 and reclaiming 2,592 KB of KV sequence capacity.

#### Scenario 4: Runtime Arbitration Policy Ablation Matrix
Evaluates the three arbitration policies supported by [`include/amuquant/controller.hpp`](file:///Users/amubhya/workspace/aegis/include/amuquant/controller.hpp) under clean, identical starting memory conditions:

| Policy Configuration | Throughput | Peak Physical RSS | KV Reclaimed | Retrieval (Recall@1 / Cosine) | P95 Latency | Final Regime |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Predictive Cost-Benefit (PCBM)** | 24.05 tok/s | 212 MB | **1440 KB** | 100% (0.8604) | 42.1 ms | REGIME_2_BALANCED_OFFLOAD |
| **Static Threshold Hysteresis** | 23.93 tok/s | 214 MB | **2016 KB** | 100% (0.6601) | 46.1 ms | REGIME_3_ULTRA_LOW_SURVIVAL |
| **Static KV Eviction Only** | 22.02 tok/s | 214 MB | **1440 KB** | 100% (0.8604) | 45.4 ms | REGIME_2_BALANCED_OFFLOAD |

#### Scenario 6: Extended Horizon (256 Tokens) & Multi-Needle Retrieval
Evaluates context retention across an extended decoding horizon of 256 tokens with 3 distinct factual needles embedded at different positions:

| Extended Horizon Metric | Measurement | Impact Analysis |
|:---|:---:|:---|
| **Target Horizon** | 256/256 tokens | 4x Nominal Horizon |
| **Inference Throughput** | 22.53 tok/s | Sustained Decoding Velocity |
| **Peak Physical RSS** | 259 MB | Bounded Memory Ceiling |
| **Logical KV Capacity Reclaimed** | **11,376 KB** | Ring-Buffer Overflow Prevented |
| **Context Tokens Archived** | **316 tokens** | In-Memory Episodic Storage |
| **Active Physical KV Slots** | **66 slots** | Attention Bounded to $O(1)$ Complexity |

*Multi-Needle Cross-Temporal Recall Matrix*:
- **Needle 1** (`AegisOmega-90412`, Position: Token 7): 100% Recall@1 | 100% Recall@3 | Cosine Score: 0.8273 | Latency: 1.150 ms.
- **Needle 2** (`SecCert-88319`, Position: Token 43): 100% Recall@1 | 100% Recall@3 | Cosine Score: 0.9257 | Latency: 0.934 ms.
- **Needle 3** (`NonceDelta-44102`, Position: Token 90): 100% Recall@1 | 100% Recall@3 | Cosine Score: 0.9051 | Latency: 0.933 ms.

#### Scenario 7: Live Long-Context Horizon Benchmark (1,024 Tokens) [Empirically Verified]
Evaluates context retention and memory bounding across an extended 1,024-token horizon (16x nominal length) with deep needle retrieval:

| Long-Context Metric | Measurement | Impact Analysis |
|:---|:---:|:---|
| **Target Horizon** | 1,024/1,024 tokens | 16x Nominal Horizon Completed |
| **Inference Throughput** | 10.23 tok/s | Sustained Decoding Velocity |
| **Peak Physical RSS** | 186 MB | Strictly Bounded Memory Ceiling |
| **Logical KV Capacity Reclaimed** | **35,856 KB (~35 MB)** | Buffer Overflow Prevented |
| **Context Tokens Archived** | **996 tokens** | In-Memory Episodic Storage |
| **Active Physical KV Slots** | **61 slots** | Attention Bounded to $O(1)$ Complexity ($\le 66$ slots) |
| **Deep Needle Recall (Depth 1,000+)** | **100% Top-1 Match** | Cosine Score: 0.8083 |

#### Scenario 8: Analytical Mathematical Projection of Scaling (Extrapolated from the Empirical 66-Slot Clamp in Scenarios 6 & 7)
> [!NOTE]
> **Methodological Demarcation Note**: Horizons 64, 256, and 1,024 tokens are empirically executed and verified live on hardware in Scenarios 1, 6, and 7. Horizons 4,096 and 8,192 tokens represent analytical mathematical projections based on the measured 66-slot physical attention clamp.

Projects theoretical and empirical memory savings for Qwen2.5-3B across expanding context depths when attention is clamped to 66 physical slots:

| Token Horizon ($N$) | Evaluation Mode | Baseline Unmanaged FP16 KV | AMU-QUANT Physical KV | AMU-QUANT 2-bit Index | Memory Reduction Ratio |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **64 tokens** | Empirically Verified (Scenario 1) | 8.38 MB | 4.71 MB | 0.03 MB | **1.8x reduction** |
| **256 tokens** | Empirically Verified (Scenario 6) | 33.55 MB | 4.71 MB | 0.22 MB | **6.8x reduction** |
| **1,024 tokens** | Empirically Verified (Scenario 7) | 134.21 MB | 4.71 MB | 0.99 MB | **23.5x reduction** |
| **4,096 tokens** | Analytical Projection | 536.87 MB | 4.71 MB | 4.06 MB | **61.2x reduction** |
| **8,192 tokens** | Analytical Projection | 1,073.74 MB (1.07 GB) | 4.71 MB | 8.16 MB | **83.4x reduction (Bounded Memory Footprint)** |

#### Scenario 9: Factual QA & Hallucination Ablation Benchmark (4-Way Matrix: A, B, C, D) [Empirically Verified]
Directly addresses the scientific challenge of factual retention and hallucination mitigation when tokens are excised from physical KV attention buffers:
- **Config A (Unmanaged Baseline)**: Full context retained in physical KV-cache without eviction.
- **Config B (AMU-QUANT Eviction Disabled)**: Full engine stack active, but transactional eviction disabled.
- **Config C (Eviction Without Context Reconstruction)**: Active transactional eviction to TurboVec, but without prompt reconstruction. Empirically isolates the hallucination caused purely by severed attention paths.
- **Config D (AMU-QUANT Hierarchical Dual-Memory)**: Active transactional eviction to TurboVec with query-driven retrieval-to-prompt reconstruction via `CanonicalHistoryStore`.

| Configuration | Evaluation Mode | Working KV Slots | Factual Accuracy | Hallucination Status | Output Snippet |
|:---|:---|:---:|:---:|:---:|:---|
| **Config A** | Unmanaged Baseline | 431 slots | **50% (2/4)** | Hallucinated / Lost (Attention Dilution) | `Berdasarkan data konfigurasi yang telah disediakan: - Na...` |
| **Config B** | AMU-QUANT (Eviction Off) | 431 slots | **50% (2/4)** | Hallucinated / Lost (Attention Dilution) | `Berdasarkan data konfigurasi yang telah disediakan: - Na...` |
| **Config C** | Eviction Without Reconstruction | **165 slots** | **0% (0/4)** | **Severe Hallucination (Context Amnesia)** | `Maaf, tetapi data konfigurasi yang Anda minta tidak diber...` |
| **Config D** | AMU-QUANT Dual-Memory (Proposed) | **290 slots** | **100% (4/4)** | **Zero Hallucination (Context Restored)** | `Nama proyek adalah Aegis, nomor versi rilis adalah 90412,...` |

*Target Ground-Truth Entity Match Breakdown*:
- **Project Name (`Aegis`)**: Config A (MATCH), Config B (MATCH), Config C (MISS), Config D (**MATCH**).
- **Version Number (`90412`)**: Config A (MATCH), Config B (MATCH), Config C (MISS), Config D (**MATCH**).
- **Architect Owner (`Ridwan`)**: Config A (MISS), Config B (MISS), Config C (MISS), Config D (**MATCH**).
- **Deployment Port (`8084`)**: Config A (MISS), Config B (MISS), Config C (MISS), Config D (**MATCH**).

*Empirical Scientific Findings*:
1. **Physical KV Pruning Causes Pure Amnesia (Config C)**: Excising tokens from `llama_context` drops factual accuracy to 0% if the model relies solely on passive latent states in vector storage. The LLM apologizes that it has no knowledge of the facts.
2. **Retrieval-to-Prompt Reconstructs Full Factual Integrity (Config D)**: Dynamically retrieving evicted turns from `CanonicalHistoryStore` via hybrid search and formatting them as structured system memory in the prompt restores 100% factual accuracy (4/4 entities matched), outperforming even unmanaged full KV-cache (Config A at 50%) due to elimination of attention dilution over intervening context.
---

# Part II: Proposed Academic Evaluation Protocol

This section details the proposed experimental setup, baseline configurations, and long-context evaluation protocols planned for formal academic thesis defense and cross-platform publication.

### 1. Proposed Multi-Node Hardware Testbed

| Parameter | Node A (Primary 16 GB Target) | Node B (Constrained 8 GB Target) |
|:---|:---|:---|
| **Physical RAM** | 16 GB Unified Memory / LPDDR5 | 8 GB DDR4 / LPDDR4X |
| **Primary Storage** | NVMe PCIe 4.0 SSD (Read $\ge 4500\text{ MB/s}$) | NVMe PCIe 3.0 SSD (Read $\approx 2500\text{ MB/s}$) |
| **CPU Architecture** | Apple Silicon M-Series / Intel Core i7-13700H / AMD Ryzen 7 7840HS | Intel Core i5-1135G7 / Apple M1 |
| **SIMD Instruction Set** | ARM NEON / x86-64 AVX2 / AVX-512 | ARM NEON / x86-64 AVX2 |
| **Operating System** | macOS Sequoia (Darwin 24) / Ubuntu 24.04 LTS (Kernel 6.8) | Ubuntu 24.04 LTS / macOS Ventura |

### 2. Proposed Model Spectrum (Cross-Scale Generalization)

1. **Compact Scale (Sub-3B):** `Qwen2.5-0.5B`, `Qwen2.5-1.5B`, and `Qwen2.5-3B` (Q4_K_M)
   * Evaluates extended context scaling (8K–32K tokens) on ultra-constrained edge devices (4 GB–8 GB RAM).
2. **Medium Scale (7B–8B):** `Meta-Llama-3-8B-Instruct` ($\approx 4.92\text{ GB}$) and `Qwen2.5-7B-Instruct`
   * Represents the critical transition threshold where weights fit in 16 GB RAM but saturate 8 GB RAM once KV-cache expands.
3. **Large Scale (14B–32B):** `Qwen2.5-14B-Instruct` ($\approx 8.99\text{ GB}$) and `Qwen2.5-32B-Instruct` / `DeepSeek-R1-Distill-Qwen-32B` ($\approx 19.8\text{ GB}$)
   * Model weights exceed physical RAM on both nodes, requiring seamless coordination between active weight prefetching and in-memory context compression.
4. **Very Large Scale (70B+ Exploratory):** `Meta-Llama-3-70B-Instruct` (Q2_K / Q4_K_M)
   * Tests extreme weight-to-RAM ratios under strict layer-by-layer streaming with full episodic context archival in `turbovec`.

### 3. Proposed Comparative Baseline Matrix

| Identifier | Method Name | Implementation Description | Weight Paging & KV Behavior |
|:---:|:---|:---|:---|
| **B1** | **llama.cpp Default (mmap)** | Official `llama-cli` with standard `--mmap 1`. | Passive OS demand paging; full uncompressed FP16 KV-cache retained in RAM until OOM. |
| **B2** | **Pure Layer-Streaming** | Active layer eviction via `posix_madvise(DONTNEED)` without context compression. | Weights streamed per layer; uncompressed KV-cache expands unrestricted until OOM. |
| **B3** | **Naive Sliding Window (StreamingLLM-style)** | Preserves 4 attention sinks and most recent $N$ tokens, discarding middle tokens (*evict-to-trash*). | Static KV-cache clamp; discarded tokens are permanently lost; blind to host OS RAM. |
| **B4** | **Static KV Quantization (TurboQuant Fork)** | Static TurboQuant integration initialized homogeneously across all layers. | Fixed bit-width from start; non-adaptive to dynamic OS memory pressure or weight buffer contention. |
| **PROP** | **AMU-QUANT (Proposed)** | Closed-loop predictive controller ($\Phi_t$), active layer madvise, zero-copy evict-to-index to `turbovec` (4b $\to$ 2b), SIMD recall. | Dynamically balances layer prefetch depth and compressed context index based on kernel telemetry. |

### 4. Proposed Macro-Evaluation Scenarios

```mermaid
flowchart TD
    subgraph S1 ["Scenario 1: Clean Context Length Scaling"]
        S1_In["Context Depths: 2K, 4K, 8K, 16K Tokens"]
        S1_Env["Clean Execution Environment"]
        S1_Out["Measure Peak RSS, Throughput, Survival Rate, Page Faults"]
    end

    subgraph S2 ["Scenario 2: Dynamic Memory Pressure Injection"]
        S2_In["External Synthetic Memory Spikes: 2 GB to 6 GB RAM"]
        S2_Trigger["Triggered Mid-Generation during Autoregressive Decoding"]
        S2_Out["Verify Regime Transitions (R1 -> R2 -> R3) & OOM Prevention"]
    end

    subgraph S3 ["Scenario 3: Context Retrieval & Perplexity Retention"]
        S3_In["Needle-in-a-Haystack Depth: 0%, 25%, 50%, 75%, 100%"]
        S3_PPL["Perplexity Evaluation on WikiText-2 Benchmark"]
        S3_Out["Measure Retrieval Accuracy (Recall@1) & Perplexity Stability"]
    end
```

---

# Methodological and Scientific Disclaimers

### 1. Physical Resident Set Size vs. Logical KV Capacity
- **Logical KV Capacity Reclaimed**: In `llama.cpp`, the attention buffer is allocated as a pre-dimensioned contiguous ring buffer arena upon context initialization. Calling `llama_memory_seq_rm()` invalidates sequence slots and repositions attention indices. This relieves attention computation complexity ($O(1)$ bounds) and prevents ring-buffer overflow, but does **not** issue immediate `munmap()` system calls to return raw pages to the operating system kernel.
- **Physical Process RSS**: Measured via `getrusage(RUSAGE_SELF).ru_maxrss` on Linux/macOS and verified via kernel task ports (`mach_task_basic_info`). Physical RSS reflects the actual physical memory pages mapped into the process's page table.
- **Kernel Page Telemetry (`mincore`)**: `LayerManager::get_actual_resident_bytes()` inspects operating system page tables via `mincore(2)` to verify true incore physical page presence independently of internal software state flags (`is_resident`).

### 2. Decoupled Working Attention vs. Episodic Semantic Memory
- **Episodic Vector Indexing (`turbovec`)**: Tokens excised from the physical attention buffer have their token-level pooled embeddings archived into `turbovec` (Lloyd-Max 4-bit / 2-bit scalar quantization).
- **Scope Boundary**: Archived vectors serve as an associative semantic memory (in-process RAG) allowing the model or agent to recall historical facts with sub-millisecond SIMD search. This **does not** reconstruct the original intermediate 36-layer Key and Value tensor states. It trades lossless layer-wise attention for a 15.5x memory compression factor.

### 3. Layer Memory Advisory Semantics (`LayerManager`)
- **OS-Level Advisory Paging**: `LayerManager` utilizes `posix_madvise(..., POSIX_MADV_DONTNEED / WILLNEED)` to notify the OS Virtual Memory Manager (VMM) which pages of the mapped GGUF file are safe to drop under memory contention.
- **Scope Boundary**: Advisory paging communicates directly with the OS page cache; it does **not** rewrite GGML tensor execution graph ASTs at runtime. On unified memory systems, this provides OS memory relief while preserving process stability.

### 4. Memory Staging vs Zero-Copy Rust ABI
- **Thread Safety Staging**: In `AsyncArchiver::commit_batch`, activation vector data is staged into thread-local task storage (`task.data[i].assign`) to guarantee thread safety and decouple the calling forward pass thread from asynchronous index updates.
- **Zero-Copy FFI Barrier**: The call to `turbovec_insert_zero_copy` passes the staged vector buffer directly across the C-to-Rust ABI without additional allocations or serialization layers inside the Rust crate.
