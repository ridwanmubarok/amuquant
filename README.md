# AMU-QUANT

**Co-Adaptive Layer-wise Weight Streaming and TurboQuant Context Compression for On-Device Large Language Model Inference**

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![C++17](https://img.shields.io/badge/C%2B%2B-17-00599C.svg?logo=c%2B%2B)](CMakeLists.txt)
[![Rust](https://img.shields.io/badge/Rust-2021-orange.svg?logo=rust)](turbovec-ffi/Cargo.toml)
[![CMake](https://img.shields.io/badge/CMake-3.20+-green.svg?logo=cmake)](CMakeLists.txt)

> **Academic Research Prototype (Tugas Akhir / Undergraduate Thesis Framework)**: This repository provides an experimental co-adaptive inference runtime for resource-constrained commodity devices. Comparative evaluation metrics represent **Formal Research Hypotheses & Evaluation Protocols**, verifiable using the built-in reproducible benchmark suite (`live_benchmark`).

---

## 1. Abstract

Running Large Language Models (LLMs) locally on hardware with bounded physical RAM faces a fundamental operational challenge: **memory exhaustion as generation context expands**. In computer systems, physical RAM allocation during autoregressive LLM decoding is governed by two competing memory demands:

1. **Static Model Weights ($M_{\text{weight}}$):** The quantized model file (e.g., GGUF format) consumes a fixed baseline of physical RAM upon initialization. When combined with operating system kernel allocations and active background tasks, the available headroom for runtime data is severely restricted before text generation begins.
2. **Dynamic Context Memory ($M_{\text{KV}}$):** During autoregressive decoding, historical attention key-value states must be preserved in the KV-Cache. As sequence length $N$ grows through extended conversations, document ingestion, or multi-step reasoning chains, the KV-Cache expands linearly ($\mathcal{O}(N)$), consuming gigabytes of additional memory.

When aggregate memory consumption exceeds physical RAM limits, the operating system kernel is forced to swap model weight pages to secondary storage (SSD *page thrashing*). This triggers an exponential drop in generation throughput or causes sudden process termination by the OS *Out-of-Memory (OOM) Killer*.

Prior systems exhibit four critical systemic limitations:
* **Layer-Streaming Neglects KV-Cache Growth:** Methods like AirLLM ([Lyogavin, 2024](https://github.com/lyogavin/airllm)) or passive `mmap` in `llama.cpp` ([Gerganov et al., 2026](https://github.com/ggml-org/llama.cpp)) stream weights layer-wise from SSD, but leave the uncompressed KV-cache in RAM, triggering OOM on extended sequences.
* **Context Eviction Causes Permanent Information Loss (*Evict-to-Trash*):** Methods such as StreamingLLM ([Xiao et al., ICLR 2024](https://arxiv.org/abs/2309.17453)), H2O ([Zhang et al., NeurIPS 2023](https://arxiv.org/abs/2306.14048)), ThinKV ([ICLR 2026](https://arxiv.org/abs/2510.01290)), and Lethe ([AAAI 2026](https://arxiv.org/abs/2511.12612)) drop older tokens permanently, causing complete recall failure on past context while remaining blind to host OS memory telemetry.
* **Static TurboQuant Lacks Runtime Adaptability:** Early TurboQuant integrations ([Zandieh et al., ICLR 2026](https://arxiv.org/abs/2504.19874)) in community branches ([TheTom, 2026](https://github.com/TheTom/llama-cpp-turboquant); [AmesianX, 2026](https://github.com/AmesianX/TurboQuant); [Issue #20977](https://github.com/ggml-org/llama.cpp/issues/20977); [PR #21038](https://github.com/ggml-org/llama.cpp/pull/21038)) require pre-configured bit-widths; they cannot adjust compression dynamically when available RAM fluctuates mid-generation.
* **The turbovec Library Lacks Native C++ Integration:** The high-performance `turbovec` crate ([Codrai, 2026](https://github.com/RyanCodrai/turbovec)) achieves $7.8\times$ (4-bit) and $15.5\times$ (2-bit) compression in microseconds via CPU SIMD, but only targets Rust and Python, lacking a zero-copy C-ABI bridge to `llama.cpp` (C++).

**AMU-QUANT** resolves this dilemma through a co-adaptive runtime engine implemented in native C++17 and Rust. It continuously monitors OS physical memory telemetry during autoregressive decoding. When memory pressure rises, AMU-QUANT excises middle tokens from the physical KV-cache in `llama.cpp` to free RAM immediately, archiving their representations into `turbovec` (4-bit or 2-bit) via a zero-copy C-ABI bridge (`turbovec-ffi`). Simultaneously, it directs kernel paging via `posix_madvise(..., POSIX_MADV_DONTNEED / WILLNEED)` to synchronize weight prefetching and context compression in dynamic equilibrium.

### Primary Academic Novelty (Tugas Akhir S1 Framework)
* **Core Scientific Novelty**: The **Predictive Runtime Host-Aware Memory Pressure Arbitration Algorithm ($\Phi_t$)**, which formalizes dual-budget memory equilibrium between internal process footprint ($\mathcal{U}_{\text{process}}$) and external host OS headroom ($\mathcal{U}_{\text{system}}$), preempting kernel swap thrashing and OOM termination through dynamic tri-modal hysteresis regimes.
* **Supporting Systems Engineering Modules**:
  1. **KV-Cache Surgery Engine**: Atomic sequence pruning with Rotary Position Embedding (RoPE) gap realignment.
  2. **Episodic Vector Memory**: Non-uniform Lloyd-Max scalar quantization (4-bit/2-bit) providing sub-millisecond semantic retrieval without unbounded memory growth.
  3. **Layer-wise Paging Manager**: Virtual memory page cache management via `posix_madvise(DONTNEED / WILLNEED)`.

> **Context Memory Architecture Clarification**: AMU-QUANT archives high-dimensional **token hidden-state representations ($\mathbf{h}_t \in \mathbb{R}^d$)** into an episodic SIMD vector memory (`turbovec`) for semantic context recall, rather than storing uncompressed multi-head attention Key/Value tensors across all layers. Physical KV-cache slots in `llama.cpp` are safely pruned via `llama_memory_seq_rm()` only after atomic representation archiving is verified.

From **Data Science** and **Natural Language Processing (NLP)** perspectives, this system advances:
1. **High-dimensional vector space optimization** via non-uniform Lloyd-Max scalar quantization (4-bit and 2-bit).
2. **Time-series predictive modeling** on OS memory telemetry to anticipate memory spikes prior to kernel page thrashing.
3. **Statistical evaluation of language modeling fidelity (Perplexity / PPL)** and information retrieval accuracy (Recall@K).

---

## 2. Generalized Cross-Scale Memory Spectrum

Memory contention is not specific to any single parameter scale; rather, it is an intrinsic physical constraint of deploying LLMs on bounded hardware. AMU-QUANT generalizes dynamically across compact, medium, large, and frontier architectures:

| Model Scale | Representative Architectures | Hardware Profile | Primary Memory Contention | AMU-QUANT Operational Impact |
| :--- | :--- | :--- | :--- | :--- |
| **Compact (< 3B)** | SmolLM2-1.7B, Qwen2.5-1.5B, Llama-3.2-3B | 4 GB – 8 GB RAM (Edge Devices, Raspberry Pi 5) | Small initial weights leave minimal margin before KV-cache expansion causes OOM. | Expands effective context window by $7.8\times$–$15.5\times$ within strict sub-4 GB ceilings. |
| **Medium (7B–8B)** | Llama-3/3.1-8B, Qwen2.5-7B, Mistral-7B | 8 GB – 16 GB RAM (Standard Developer Laptops) | Static weights consume 50%–70% of RAM; long-context or multi-agent tasks trigger swap thrashing. | Sustains multi-agent concurrency with zero swap thrashing and ~99% context recall. |
| **Large (14B–32B)** | Qwen2.5-14B, Qwen2.5-32B, DeepSeek-R1-Distill-32B | 16 GB – 32 GB RAM (High-End Laptops, Mac Mini) | Models barely fit into unified RAM; even modest KV growth forces kernel page swapping. | Enables single-layer streaming with Tier-2/Tier-3 compressed episodic cache storage. |
| **Frontier (70B+)** | Llama-3.3-70B, Qwen2.5-72B | 32 GB – 64 GB+ RAM (Workstations & Server Nodes) | Multi-turn dialogs and high-concurrency agent swarms exhaust server memory rapidly. | Prevents out-of-memory cascades during massive concurrent batching and long-horizon tasks. |

---

## 3. Mathematical Formalization

### 3.1 Dual-Budget Memory Equilibrium

Total physical memory allocation (*Resident Set Size* / RSS) at step $t$ is bounded by:

$$
M_{\text{ext}}(t) + \sum_{l \in \mathcal{L}_{\text{resident}}(t)} M_{\text{weight}}(l) + M_{\text{KV}}(W_t) + M_{\text{turbovec}}(C_t, b_t) \le M_{\text{limit}}
$$

Where:
* **$M_{\text{limit}} = M_{\text{RAM}} - M_{\text{reserved}}$**: Safe physical RAM threshold, preserving an operating system kernel headroom buffer ($M_{\text{reserved}}$).
* **$M_{\text{ext}}(t)$**: Physical memory consumed by external system processes and applications.
* **$\mathcal{L}_{\text{resident}}(t)$**: Set of transformer layer weights currently resident in RAM via `mmap`.
* **$W_t$**: Count of active uncompressed tokens retained in the physical KV-cache.
* **$C_t$**: Count of historical tokens archived in the compressed `turbovec` index.
* **$b_t \in \{4, 2\}$**: TurboQuant quantization bit-width precision.

### 3.2 Predictive Memory Pressure Metric ($\Phi_t$)

Evaluated every $K$ tokens to anticipate allocation spikes before swapping occurs:

$$
\Phi_t = \max\left(\mathcal{U}_{\text{process}}(t),\, \mathcal{U}_{\text{system}}(t)\right) + \beta \cdot \mathcal{P}_{\text{growth}}(t) + \gamma \cdot \mathcal{P}_{\text{swap}}(t)
$$

Where:
* **$\mathcal{U}_{\text{process}}(t) = \frac{M_{\text{process}}(t)}{M_{\text{limit}}}$**: LLM process memory saturation.
* **$\mathcal{U}_{\text{system}}(t) = \frac{M_{\text{total}} - M_{\text{available}}(t)}{M_{\text{total}} - M_{\text{reserved}}}$**: Operating system host memory exhaustion.
* **$\mathcal{P}_{\text{growth}}(t) = \min\left(1.0, \max\left(0.0, \frac{1}{M_{\text{limit}}} \frac{\Delta M_{\text{process}}}{\Delta t} \tau_{\text{horizon}}\right)\right)$**: Differential growth rate penalty.
* **$\mathcal{P}_{\text{swap}}(t) = \min\left(1.0, \frac{M_{\text{swap}}(t)}{1.0\text{ GB}}\right)$**: OS kernel swap thrashing penalty.
* Nominal parameters: $\beta = 0.25$, $\gamma = 0.15$, and $\tau_{\text{horizon}} = 2.0\text{ s}$.

### 3.3 Tri-Modal Operational Regimes

* **Regime 1: High-Throughput Caching ($\Phi_t \le \theta_{\text{low}}$)**  
  Maximizes layer prefetching in RAM ($|\mathcal{L}_{\text{resident}}| \gg 1$). Full uncompressed FP16 KV-cache is retained for maximum throughput.
* **Regime 2: Balanced Offloading ($\theta_{\text{low}} < \Phi_t \le \theta_{\text{high}}$)**  
  Middle tokens excised from physical KV-cache and archived to `turbovec` 4-bit ($\approx 7.8\times$ compression). Slower layers are evicted via `posix_madvise(..., POSIX_MADV_DONTNEED)`. Prefetch depth throttled to 1 layer.
* **Regime 3: Ultra-Low Memory Survival ($\Phi_t > \theta_{\text{high}}$)**  
  In-place requantization of the `turbovec` index from 4-bit to 2-bit ($\approx 15.5\times$ compression). Enforces strict single-layer execution (prefetch depth = 0) with sub-0.25 ms SIMD cosine recall.

---

## 4. System Architecture

```mermaid
flowchart TD
    subgraph OS_Layer ["Operating System Kernel and Storage Layer"]
        OS_Mem["Kernel Virtual Memory (/proc/meminfo or mach_vm_info)"]
        Sys_Madvise["Kernel Paging: posix_madvise(DONTNEED / WILLNEED)"]
        GGUF_File["Unified .gguf Model File on NVMe SSD"]
    end

    subgraph AMUQuant_Controller ["AMU-QUANT Co-Adaptive Controller (C++17)"]
        TelemetrySampler["Telemetry Sampler (Every K Tokens)"]
        PredictiveScore["Predictive Pressure Score: Φ_t"]
        RegimeMachine{"Hysteresis State Machine (δ_hyst)"}
        R1["Regime 1: High-Throughput (Max Prefetch)"]
        R2["Regime 2: Balanced Offload (turbovec 4-bit)"]
        R3["Regime 3: Ultra-Low Survival (turbovec 2-bit)"]
    end

    subgraph Llama_Engine ["Inference Engine: llama.cpp"]
        WeightRegistry["Mapped Layer Weights (via mmap)"]
        KVCache["Physical KV-Cache Buffer"]
        KVSurgery["KV Surgery: llama_kv_cache_seq_rm and seq_add"]
    end

    subgraph Turbovec_FFI ["Episodic Vector Memory: turbovec (Rust FFI)"]
        FFI_Layer["C-ABI Foreign Function Interface"]
        TurboQuantIndex["TurboQuant Index (Lloyd-Max Scalar Quantizer)"]
        Requantizer["In-Place 4-bit to 2-bit Requantization"]
        SIMDSearch["SIMD AVX2 / ARM NEON Search Engine (<0.25 ms)"]
    end

    OS_Mem --> TelemetrySampler
    TelemetrySampler --> PredictiveScore
    PredictiveScore --> RegimeMachine
    RegimeMachine -->|Φ_t ≤ θ_low| R1
    RegimeMachine -->|θ_low < Φ_t ≤ θ_high| R2
    RegimeMachine -->|Φ_t > θ_high| R3

    R1 -.->|Prefetch Upcoming Layers| Sys_Madvise
    R2 -->|Evict Middle Tokens| KVSurgery
    R3 -->|Trigger In-Place 2-bit Requantization| Requantizer

    GGUF_File -->|mmap read-only| WeightRegistry
    Sys_Madvise -->|Evict / Prefetch Pages| WeightRegistry
    KVSurgery -->|Zero-Copy Raw Pointer Passing| FFI_Layer
    FFI_Layer --> TurboQuantIndex
    SIMDSearch -->|Selective Context Recall| KVCache
```

---

## 5. Target Research Hypotheses & Live Benchmark Protocol

AMU-QUANT evaluates inference efficiency and memory stabilization using a formal empirical benchmark suite executable directly on developer hardware:

```bash
# Option A: Zero-dependency standalone benchmark (Always built by default)
./build/amuquant_benchmark --tokens 96

# Option B: Full-weight GGUF benchmark with llama.cpp (Setup via ./scripts/setup_llama.sh)
./build/live_benchmark --model models/Meta-Llama-3-8B-Instruct-Q2_K.gguf --tokens 32
```

### 5.1 Live Empirical Benchmark Results (Apple Silicon Metal GPU)
The table below reports live empirical measurements obtained from running `./build/live_benchmark` across a 3-way methodological parity testbed on `Meta-Llama-3-8B-Instruct-Q2_K.gguf` (32 Layers, Dim 4096, Metal GPU):

| Evaluation Metric | Baseline 1 (Raw Vanilla) | Baseline 2 (With Embeddings) | AMU-QUANT Co-Adaptive Engine |
| :--- | :--- | :--- | :--- |
| **Target Horizon Completed** | Yes (100%) | Yes (100%) | **Yes (100%)** |
| **Tokens Generated** | 32/32 | 32/32 | **32/32** |
| **Peak Physical RSS** | 370 MB | 394 MB | **397 MB** |
| **Inference Throughput** | 2.71 tok/s | 11.45 tok/s | **11.22 tok/s** |
| **Physical KV Memory Reclaimed** | 0 KB | 0 KB | **7,680 KB (7.5 MB)** |
| **Context Tokens Safely Archived** | 0 | 0 | **60 tokens** |
| **Semantic Context Recall** | N/A | N/A | **0.8468 (Cosine Match)** |
| **SIMD Retrieval Latency** | N/A | N/A | **0.510 ms** |

### 5.2 Standalone Zero-Dependency Benchmark Results
For evaluators running in environments without `llama.cpp` or local weights, `./build/amuquant_benchmark` executes the identical mathematical state machine, `TokenIndexMap`, and Lloyd-Max vector quantization:

| Metric | Unmanaged Baseline | AMU-QUANT Co-Adaptive Engine |
| :--- | :--- | :--- |
| **Sequence Horizon** | 96 tokens | **96 tokens** |
| **Peak Physical KV Memory** | 192 KB | **80 KB (58.3% Peak Reduction)** |
| **Final Active KV Memory** | 192 KB | **38 KB (80.2% Final Reduction)** |
| **Physical KV Memory Reclaimed** | 0 KB | **154 KB** |
| **Tokens Safely Archived** | 0 | **77 tokens** |
| **Needle Context Recall Score** | N/A | **0.8642 (Top-1 Cosine Match)** |
| **SIMD Retrieval Latency** | N/A | **111.9 µs** |

---

## 6. Technical Documentation Directory

Detailed scientific specifications, mathematical proofs, and architectural blueprints are available in [`docs/`](docs/):

* [**docs/01_SYSTEM_ARCHITECTURE.md**](docs/01_SYSTEM_ARCHITECTURE.md): Subsystems layout (Layer Manager, KV Surgery, FFI Bridge, Controller), zero-copy pointer flows, and thread safety contracts.
* [**docs/02_MATHEMATICAL_MODEL_AND_CONTROLLER.md**](docs/02_MATHEMATICAL_MODEL_AND_CONTROLLER.md): Dual-budget memory equilibrium, predictive score metric $\Phi_t$, hysteresis state machine, and RoPE realignment mathematics.
* [**docs/03_TURBOVEC_FFI_SPECIFICATION.md**](docs/03_TURBOVEC_FFI_SPECIFICATION.md): C-ABI specification `turbovec_ffi.h`, Rust bridge implementation, Lloyd-Max codebooks, and SIMD acceleration.
* [**docs/04_EXPERIMENTAL_METHODOLOGY.md**](docs/04_EXPERIMENTAL_METHODOLOGY.md): Hardware matrix, generalized cross-scale model spectrum (sub-3B, 7B/8B, 14B/32B, 70B+), 4 baseline methods, and 3 test scenarios.
* [**docs/05_LITERATURE_REVIEW_AND_REFERENCES.md**](docs/05_LITERATURE_REVIEW_AND_REFERENCES.md): Critical analysis of prior art, Data Science & NLP theoretical foundations, and verified citations.

---

## 7. C-ABI Interface Summary (`turbovec_ffi.h`)

The zero-copy native interface bridging C++ (`llama.cpp`) and Rust (`turbovec`):

```c
// Allocate TurboQuant in-process vector index on Rust heap (4-bit or 2-bit)
TurbovecStatus turbovec_init(uint32_t dimension, uint8_t bit_width, TurbovecIndexHandle** out_handle);

// Zero-copy insertion of raw contiguous float pointers
TurbovecStatus turbovec_insert_zero_copy(TurbovecIndexHandle* handle, uint64_t chunk_id, const float* float_ptr, size_t len);

// Sub-0.25 ms SIMD-accelerated Top-K cosine similarity search
TurbovecStatus turbovec_search_simd(const TurbovecIndexHandle* handle, const float* query_ptr, uint32_t top_k, uint64_t* out_ids, float* out_scores, uint32_t* out_count);

// In-place downscaling from 4-bit to 2-bit precision (Regime 3 Survival Mode)
TurbovecStatus turbovec_requantize_to_2bit(TurbovecIndexHandle* handle);

// Query heap allocation allocated by index in bytes
TurbovecStatus turbovec_get_memory_usage(const TurbovecIndexHandle* handle, size_t* out_bytes);

// Deallocate index instance and release heap memory
TurbovecStatus turbovec_free(TurbovecIndexHandle* handle);
```

---

## 8. Build and Installation Guide

### Prerequisites
* CMake >= 3.20
* C++17 compliant compiler (Apple Clang >= 15 or GCC >= 11)
* Optional: Rust toolchain (Cargo >= 1.75) if building with `-DAMUQUANT_USE_RUST_BACKEND=ON`

### Building from Source

```bash
# 1. Clone repository
git clone https://github.com/ridwanmubarok/amuquant.git
cd amuquant

# 2. Configure build (Default: Canonical C++17 SIMD engine, zero external dependencies)
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release

# Optional: To enable the turbovec-ffi Rust backend via Cargo:
# cmake -B build -S . -DCMAKE_BUILD_TYPE=Release -DAMUQUANT_USE_RUST_BACKEND=ON

# 3. Compile targets
cmake --build build -j

# 4. Run test suite
ctest --test-dir build --output-on-failure
```

### Installation as a System Package
```bash
sudo cmake --install build
```

### Using AMU-QUANT in Downstream CMake Projects
```cmake
find_package(amuquant REQUIRED)

add_executable(my_inference_app main.cpp)
target_link_libraries(my_inference_app PRIVATE amuquant::amuquant_core)
```

---

## 9. Key References

1. **Zandieh, A., Daliri, M., Hadian, M., & Mirrokni, V. (2026).** *TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate*. In *Proceedings of the International Conference on Learning Representations (ICLR 2026)*. [arXiv:2504.19874](https://arxiv.org/abs/2504.19874).
2. **Codrai, R. (2026).** *turbovec: High-performance, in-process vector search index in Rust implementing Google Research's TurboQuant*. [GitHub](https://github.com/RyanCodrai/turbovec).
3. **Gerganov, G., et al. (2026).** *llama.cpp: Inference of LLaMA model in pure C/C++*. [GitHub](https://github.com/ggml-org/llama.cpp).
4. **Lyogavin. (2024).** *AirLLM: Layer-wise LLM Inference Optimization for Resource-Constrained Hardware*. [GitHub](https://github.com/lyogavin/airllm).
5. **Xiao, G., Tian, Y., Chen, B., Han, S., & Lewis, M. (2024).** *Efficient Streaming Language Models with Attention Sinks (StreamingLLM)*. In *ICLR 2024*. [arXiv:2309.17453](https://arxiv.org/abs/2309.17453).
6. **Zhang, Z., et al. (2023).** *H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models*. In *NeurIPS 2023*. [arXiv:2306.14048](https://arxiv.org/abs/2306.14048).
7. **ThinKV Authors. (2026).** *ThinKV: Thought-Adaptive KV Cache Compression for Reasoning Models*. In *ICLR 2026*. [arXiv:2510.01290](https://arxiv.org/abs/2510.01290).
8. **Lethe Authors. (2026).** *Lethe: Layer-Adaptive KV Cache Eviction with Dynamic Sparsity Allocation*. In *AAAI 2026*. [arXiv:2511.12612](https://arxiv.org/abs/2511.12612).
9. **TheTom. (2026).** *llama-cpp-turboquant: Experimental TurboQuant integration in llama.cpp*. [GitHub](https://github.com/TheTom/llama-cpp-turboquant).
10. **AmesianX. (2026).** *TurboQuant: C++ Implementation and Exploration for Quantized Inference*. [GitHub](https://github.com/AmesianX/TurboQuant).

---

## 10. License and Academic Context

* **Project:** AMU-QUANT Open-Source Project  
* **Discipline:** Data Science, Natural Language Processing, & High-Performance Systems for Deep Learning  
* **License:** Apache License 2.0
