# AMU-QUANT Documentation and Scientific Specifications

Dokumentasi ini memuat spesifikasi ilmiah, rancangan matematis, arsitektur sistem, dan metodologi eksperimental untuk proyek **AMU-QUANT** (*Co-Adaptive Layer-wise Weight Streaming and TurboQuant Context Compression for On-Device Large Language Model Inference*).

---

## 1. Daftar Dokumen Spesifikasi Ilmiah

| No | Dokumen | Ringkasan Isi |
|:---:|:---|:---|
| **01** | [**Proposal Usulan Tugas Akhir**](01_PROPOSAL_TUGAS_AKHIR.md) | Naskah formal usulan Tugas Akhir lengkap (Abstrak, Bab I-IV, rumusan masalah, batasan teknis, tujuan, dan metodologi). |
| **02** | [**Arsitektur Sistem dan Alur Kerja**](02_ARSITEKTUR_SISTEM.md) | Rancangan 4 subsistem (Layer Manager, KV Surgery, FFI Bridge, Controller), alur *zero-copy*, dan penanganan *thread-safety*. |
| **03** | [**Model Matematika dan Kontroler Ko-Adaptif**](03_MODEL_MATEMATIKA_DAN_KONTROLER.md) | Pertidaksamaan ekuilibrium memori ganda, formula $\Phi_t$, analisis histeresis, dan pseudocode algoritma penjadwalan. |
| **04** | [**Desain Jembatan turbovec-ffi (C-ABI)**](04_DESAIN_TURBOVEC_FFI.md) | Header C-ABI `turbovec_ffi.h`, implementasi Rust `ffi.rs`, rotasi ortogonal acak, kuantisasi Lloyd-Max, dan SIMD. |
| **05** | [**Metodologi dan Rencana Eksperimen**](05_METODOLOGI_DAN_RENCANA_EKSPERIMEN.md) | Setup perangkat keras (RAM 8 GB dan 16 GB), spektrum model uji lintas-skala (sub-3B, 7B/8B, 14B, 32B), 4 baseline, 3 skenario uji, dan metrik kuantitatif. |
| **06** | [**Tinjauan Pustaka dan Komparasi State-of-the-Art**](06_TINJAUAN_PUSTAKA_DAN_REFERENSI.md) | Kajian pustaka kritis (AirLLM, llama.cpp, StreamingLLM, H2O, ThinKV, Lethe, TurboQuant) dan analisis kebaruan komparatif. |

---

## 2. Arsitektur Alur Kerja Ko-Adaptif

```mermaid
flowchart TD
    subgraph Kernel_OS ["Lapisan Kernel OS (Virtual Memory dan Telemetri)"]
        VM_Stat["Telemetri RAM dan Swap (/proc/meminfo / mach_vm_info)"]
        KernelMadvise["Kernel Directives: posix_madvise(DONTNEED / WILLNEED)"]
    end

    subgraph AMUQuant_Engine ["AMU-QUANT Co-Adaptive Controller (C++17)"]
        Sampling["Periodic Telemetry Sampler (Setiap K Token)"]
        ScoreEngine["Kalkulator Skor Tekanan Prediktif: Φ_t"]
        RegimeLogic{"State Machine Histeresis (δ_hyst)"}
        R1["Regime 1: High-Throughput (Max Prefetch)"]
        R2["Regime 2: Balanced Offloading (turbovec 4-bit)"]
        R3["Regime 3: Ultra-Low Survival (turbovec 2-bit + Strict Streaming)"]
    end

    subgraph Llama_CPP ["Mesin Inferensi: llama.cpp"]
        MmapWeights["Model Weights GGUF (Single File via mmap)"]
        ActiveKVCache["KV-Cache Buffer Aktif"]
        Surgery["KV Surgery: llama_kv_cache_seq_rm dan seq_add"]
    end

    subgraph Turbovec_Rust ["Indeks Vektor: turbovec (Rust C-ABI via FFI)"]
        FFI["turbovec-ffi (Zero-Copy Raw Pointer Bridge)"]
        TurboQuant["TurboQuant Index (Lloyd-Max Scalar Quantizer)"]
        Requantize["In-Place 4-bit -> 2-bit Requantization"]
        SIMD["SIMD Fast Search (<0.25 ms Cosine Recall)"]
    end

    VM_Stat --> Sampling
    Sampling --> ScoreEngine
    ScoreEngine --> RegimeLogic
    RegimeLogic -->|Φ_t ≤ θ_low| R1
    RegimeLogic -->|θ_low < Φ_t ≤ θ_high| R2
    RegimeLogic -->|Φ_t > θ_high| R3

    R1 -.->|Prefetch Layer Mendatang| KernelMadvise
    R2 -->|Evict Token Tengah| Surgery
    R3 -->|Trigger 2-bit dan Strict Streaming| Surgery

    Surgery -->|Pointer Float Langsung (Zero-Copy)| FFI
    FFI --> TurboQuant
    TurboQuant --> Requantize
    SIMD -->|Selective Context Retrieval| ActiveKVCache
    KernelMadvise -->|Evict Inactive Layer Pages| MmapWeights
```

---

## 3. Daftar Referensi Ilmiah Terverifikasi

1. **Zandieh, A., Daliri, M., Hadian, M., & Mirrokni, V. (2026).** *TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate*. In *Proceedings of the International Conference on Learning Representations (ICLR 2026)*.  
   Tautan arXiv: [https://arxiv.org/abs/2504.19874](https://arxiv.org/abs/2504.19874)  
   Tautan Google Research: [https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/)

2. **Codrai, R. (2026).** *turbovec: High-performance, in-process vector search index in Rust implementing Google Research's TurboQuant (ICLR 2026)*.  
   Tautan GitHub: [https://github.com/RyanCodrai/turbovec](https://github.com/RyanCodrai/turbovec)

3. **Gerganov, G., et al. (2026).** *llama.cpp: Inference of LLaMA model in pure C/C++*.  
   Tautan GitHub: [https://github.com/ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)  
   Diskusi dan PR Terkait TurboQuant: [Issue #20977](https://github.com/ggml-org/llama.cpp/issues/20977) dan [Pull Request #21038](https://www.google.com/search?q=https://github.com/ggml-org/llama.cpp/pull/21038)

4. **Lyogavin. (2023/2024).** *AirLLM: 70B inference with single 4GB GPU (Layer-wise LLM Inference Optimization)*.  
   Tautan GitHub: [https://github.com/lyogavin/airllm](https://github.com/lyogavin/airllm)

5. **Xiao, G., Tian, Y., Chen, B., Han, S., & Lewis, M. (2024).** *Efficient Streaming Language Models with Attention Sinks (StreamingLLM)*. In *Proceedings of the International Conference on Learning Representations (ICLR 2024)*.  
   Tautan arXiv: [https://arxiv.org/abs/2309.17453](https://arxiv.org/abs/2309.17453)

6. **Zhang, Z., Sheng, Y., Zhou, T., Chen, T., Zheng, L., Cai, R., Song, Z., Tian, Y., Ré, C., Barrett, C., Wang, Z., & Chen, B. (2023).** *H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models*. In *Advances in Neural Information Processing Systems (NeurIPS 2023)*, Vol. 36.  
   Tautan arXiv: [https://arxiv.org/abs/2306.14048](https://arxiv.org/abs/2306.14048)

7. **ThinKV Authors. (2026).** *ThinKV: Thought-Adaptive KV Cache Compression for Reasoning Models*. In *Proceedings of the International Conference on Learning Representations (ICLR 2026)*.  
   Tautan arXiv: [https://arxiv.org/abs/2510.01290](https://arxiv.org/abs/2510.01290)

8. **Lethe Authors. (2026).** *Lethe: Layer-Adaptive KV Cache Eviction with Dynamic Sparsity Allocation*. In *Proceedings of the AAAI Conference on Artificial Intelligence (AAAI 2026)*.  
   Tautan arXiv: [https://arxiv.org/abs/2511.12612](https://arxiv.org/abs/2511.12612)

9. **Liu, Z., Desai, A., Liao, F., Shen, W., Xie, Z., Xu, K., Pavone, M., & Ermon, S. (2024).** *Scissorhands: Exploiting the Persistence of Importance Hypothesis for LLM KV Cache Compression*. In *Advances in Neural Information Processing Systems (NeurIPS 2024)*.  
   Tautan arXiv: [https://arxiv.org/abs/2305.17118](https://arxiv.org/abs/2305.17118)

10. **TheTom. (2026).** *llama-cpp-turboquant: Experimental TurboQuant integration in llama.cpp*.  
    Tautan GitHub: [https://github.com/TheTom/llama-cpp-turboquant](https://github.com/TheTom/llama-cpp-turboquant)

11. **AmesianX. (2026).** *TurboQuant: C++ Implementation and Exploration for Quantized Inference*.  
    Tautan GitHub: [https://github.com/AmesianX/TurboQuant](https://github.com/AmesianX/TurboQuant)
