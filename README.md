# AMU-QUANT: Co-Adaptive Layer-wise Weight Streaming and TurboQuant Context Compression for On-Device Large Language Model Inference

---

## 1. Abstract

Running Large Language Models (LLMs) locally on commodity devices with constrained physical RAM (e.g. 8 GB–16 GB) encounters a fundamental barrier: **memory exhaustion as generation context expands**. In computer systems, physical RAM allocation during LLM execution is divided between two competing memory demands:

1. **Static Model Weights:** The quantized model file (e.g. GGUF format) consumes a major fraction of physical RAM upon initialization. When combined with operating system kernel allocations, available memory headroom is severely constrained before text generation begins. This applies across model scales: compact models (sub-3B) on 4 GB–8 GB devices, medium models (7B–8B) on 8 GB–16 GB devices, and large models (14B, 32B, 70B+) on 16 GB–32 GB hosts.
2. **Dynamic Context Memory (Key-Value Cache / KV-Cache):** During autoregressive decoding, historical hidden representations must be preserved in the KV-Cache. As conversation turns or multi-step reasoning chains extend into thousands of tokens, the KV-Cache expands linearly, consuming substantial additional physical RAM.

When KV-Cache expansion or concurrent host applications (e.g. web browsers or IDEs) exceed physical RAM limits, the operating system kernel is forced to swap model weight pages to secondary SSD storage (*page thrashing*). This collapses generation throughput or causes process termination via the *Out-of-Memory (OOM) Killer*.

Prior systems suffer from four critical systemic gaps:
* **Layer-Streaming Neglects KV-Cache Explosion:** Methods like AirLLM ([Lyogavin, 2024](https://github.com/lyogavin/airllm)) or passive `mmap` in `llama.cpp` ([Gerganov et al., 2026](https://github.com/ggml-org/llama.cpp)) stream weights layer-wise from SSD, but leave the uncompressed KV-Cache in RAM, inevitably triggering OOM on long sequences.
* **Context Eviction Causes Permanent Memory Loss (*Evict-to-Trash*):** Methods such as StreamingLLM ([Xiao et al., ICLR 2024](https://arxiv.org/abs/2309.17453)), H2O ([Zhang et al., NeurIPS 2023](https://arxiv.org/abs/2306.14048)), ThinKV ([ICLR 2026](https://arxiv.org/abs/2510.01290)), and Lethe ([AAAI 2026](https://arxiv.org/abs/2511.12612)) delete older tokens permanently (*evict-to-trash*), resulting in permanent context loss while remaining entirely blind to OS physical RAM telemetry.
* **TurboQuant in llama.cpp Remains Statically Configured:** Early TurboQuant integrations ([Zandieh et al., ICLR 2026](https://arxiv.org/abs/2504.19874)) in community forks ([TheTom, 2026](https://github.com/TheTom/llama-cpp-turboquant); [AmesianX, 2026](https://github.com/AmesianX/TurboQuant); [Issue #20977](https://github.com/ggml-org/llama.cpp/issues/20977); [PR #21038](https://www.google.com/search?q=https://github.com/ggml-org/llama.cpp/pull/21038)) require fixed initialization prior to execution; they cannot dynamically adjust bit-width when host RAM spikes mid-generation.
* **The turbovec Library Lacks Native C++ Integration:** The high-performance `turbovec` crate ([Codrai, 2026](https://github.com/RyanCodrai/turbovec)) achieves $7.8\times$ (4-bit) and $15.5\times$ (2-bit) vector compression in microseconds via CPU SIMD, but only supports Rust and Python, lacking a C-ABI bridge to `llama.cpp` (C++).

**AMU-QUANT** introduces a co-adaptive runtime engine in native C++17 and Rust. It continuously samples OS memory telemetry during generation. When memory pressure rises, AMU-QUANT excises middle tokens from the physical KV-cache in `llama.cpp` to free RAM immediately, archiving their representations into `turbovec` (4-bit or 2-bit) via a zero-copy C-ABI bridge (`turbovec-ffi`). Simultaneously, it directs kernel paging via `posix_madvise(..., POSIX_MADV_DONTNEED / WILLNEED)` to synchronize weight prefetching and context compression in dynamic equilibrium.

From **Data Science** and **Natural Language Processing (NLP)** perspectives, this research addresses:
1. **High-dimensional vector space optimization** via Lloyd-Max scalar quantization (4-bit and 2-bit).
2. **Time-series predictive modeling** on OS memory telemetry to anticipate memory spikes before swapping occurs.
3. **Statistical evaluation of language modeling quality (Perplexity / PPL)** and information retrieval accuracy (Recall@K).

---

## 2. Mathematical Formalization

### 2.1 Dual-Budget Memory Equilibrium
Total physical memory allocation (*Resident Set Size* / RSS) at step $t$ is bounded by:

$$M_{\text{ext}}(t) + \sum_{l \in \mathcal{L}_{\text{resident}}(t)} M_{\text{weight}}(l) + M_{\text{KV}}(W_t) + M_{\text{turbovec}}(C_t, b_t) \le M_{\text{limit}}$$

Where:
* $M_{\text{limit}} = M_{\text{RAM}} - M_{\text{OS\_reserved}}$: Safe physical RAM limit.
* $\mathcal{L}_{\text{resident}}(t)$: Set of layer weights currently resident in RAM via `mmap`.
* $W_t$: Number of active tokens in physical KV-cache.
* $C_t$: Number of archived tokens in compressed `turbovec` index.
* $b_t \in \{4, 2\}$: Current TurboQuant bit-width precision.

### 2.2 Predictive Memory Pressure Metric ($\Phi_t$)
Evaluated at every $K$ tokens to anticipate allocation spikes:

$$\Phi_t = \alpha \left( \frac{M_{\text{total}}(t)}{M_{\text{limit}}} \right) + \beta \left( \frac{\widehat{\Delta M}_t}{\Delta t} \cdot \frac{\tau_{\text{horizon}}}{M_{\text{limit}}} \right)$$

Nominal parameters: $\alpha = 0.65$, $\beta = 0.35$, and $\tau_{\text{horizon}} = 2.0\text{ seconds}$.

### 2.3 Tri-Modal Regime Arbitration
* **Regime 1: High-Throughput Caching ($\Phi_t \le \theta_{\text{low}}$)**  
  Maximizes layer prefetching in RAM ($|\mathcal{L}_{\text{resident}}| \gg 1$). Full uncompressed FP16 KV-cache retained.
* **Regime 2: Balanced Offloading ($\theta_{\text{low}} < \Phi_t \le \theta_{\text{high}}$)**  
  Middle tokens excised from physical KV-cache and archived to `turbovec` 4-bit ($\approx 7.8\times$ compression). Slower layers evicted via `posix_madvise(..., POSIX_MADV_DONTNEED)`. Prefetch depth throttled to 1 layer.
* **Regime 3: Ultra-Low Memory Survival ($\Phi_t > \theta_{\text{high}}$)**  
  In-place requantization of `turbovec` index from 4-bit to 2-bit ($\approx 15.5\times$ compression). Strict single-layer streaming (prefetch depth = 0). Sub-0.25 ms SIMD cosine recall enabled.

---

## 3. System Architecture

```mermaid
flowchart TD
    subgraph OS_Layer ["Operating System Kernel & Storage Layer"]
        OS_Mem["Kernel Virtual Memory (/proc/meminfo / mach_vm_info)"]
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
        KVSurgery["KV Surgery: llama_kv_cache_seq_rm & seq_add"]
    end

    subgraph Turbovec_FFI ["Episodic Vector Memory: turbovec (Rust FFI)"]
        FFI_Layer["C-ABI Foreign Function Interface"]
        TurboQuantIndex["TurboQuant Index (Lloyd-Max Scalar Quantizer)"]
        Requantizer["In-Place 4-bit -> 2-bit Requantization"]
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

## 4. Technical Documentation Directory

Detailed scientific specifications and engineering contracts are maintained in [`docs/`](docs/):

* [**docs/01_SYSTEM_ARCHITECTURE.md**](docs/01_SYSTEM_ARCHITECTURE.md): Subsystem layout (Layer Manager, KV Surgery, FFI Bridge, Controller), zero-copy pointer flows, and thread safety.
* [**docs/02_MATHEMATICAL_MODEL_AND_CONTROLLER.md**](docs/02_MATHEMATICAL_MODEL_AND_CONTROLLER.md): Dual-budget memory equilibrium, predictive metric $\Phi_t$, hysteresis state machine, and RoPE realignment.
* [**docs/03_TURBOVEC_FFI_SPECIFICATION.md**](docs/03_TURBOVEC_FFI_SPECIFICATION.md): C-ABI header `turbovec_ffi.h`, Rust bridge implementation, Lloyd-Max codebooks, and SIMD acceleration.
* [**docs/04_EXPERIMENTAL_METHODOLOGY.md**](docs/04_EXPERIMENTAL_METHODOLOGY.md): Hardware testbeds, generalized cross-scale model spectrum (sub-3B, 7B/8B, 14B/32B, 70B+), 4 baseline methods, and 3 test scenarios.
* [**docs/05_LITERATURE_REVIEW_AND_REFERENCES.md**](docs/05_LITERATURE_REVIEW_AND_REFERENCES.md): Critical analysis of prior art, Data Science & NLP theoretical foundations, and verified citations.

---

## 5. C-ABI Interface Summary (`turbovec_ffi.h`)

The zero-copy native interface between C++ (`llama.cpp`) and Rust (`turbovec`) is summarized below:

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

## 6. Build and Installation Guide

### Prerequisites
* CMake >= 3.20
* C++17 compliant compiler (Clang >= 15 or GCC >= 11)
* Rust toolchain (Cargo >= 1.75) for compiling `turbovec-ffi`

### Building from Source
```bash
# 1. Clone repository
git clone https://github.com/<username>/amuquant.git
cd amuquant

# 2. Configure build
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release

# 3. Compile targets
cmake --build build -j

# 4. Run unit tests
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

## 7. Key References

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

## 8. Authors and Academic Context

* **Project:** AMU-QUANT Open-Source Project  
* **Discipline:** Data Science, Natural Language Processing, & High-Performance Systems for Deep Learning  
* **License:** Apache License 2.0
