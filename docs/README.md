# AMU-QUANT Technical Documentation

This directory contains the scientific specifications, mathematical formulations, subsystem architecture, and experimental evaluation methodology for **AMU-QUANT** (*Host-Aware Co-Adaptive Runtime Memory Policy and Sliding-Window KV-Cache Surgery for Edge Large Language Model Inference*).

---

## 1. Documentation Index

| Index | Document | Summary |
|:---:|:---|:---|
| **01** | [**System Architecture**](01_SYSTEM_ARCHITECTURE.md) | High-level native C++17 subsystem layout (Telemetry, Layer Manager, KV Surgeon, Controller, Canonical History & FactRegister), memory boundaries, and thread-safety invariants. |
| **02** | [**Mathematical Model & Controller**](02_MATHEMATICAL_MODEL_AND_CONTROLLER.md) | Dual-budget memory equilibrium inequality, predictive pressure metric $\Phi_t$, cost-benefit utility $J(a)$, temporal conflict state space, evidence threshold $\tau_{\text{evidence}}$, and scheduling algorithm. |
| **04** | [**Experimental Methodology**](04_EXPERIMENTAL_METHODOLOGY.md) | Hardware testbeds, empirical benchmarks (Triad Benchmark & Scenario 4 Tri-Dimensional Reliability Benchmark inspired by HaluMem), host memory starvation protocols, and metrics. |
| **05** | [**Literature Review & References**](05_LITERATURE_REVIEW_AND_REFERENCES.md) | Critical comparative analysis against AirLLM, llama.cpp mmap, attention sinking (StreamingLLM, H2O, ThinKV), reliable local memory architectures (HaluMem), computer systems memory modeling, and verified academic references. |

---

## 2. Co-Adaptive Execution Lifecycle

```mermaid
flowchart TD
    subgraph Kernel_OS ["Operating System Kernel & Host Hardware"]
        VM_Stat["Memory & Swap Telemetry (/proc/meminfo / mach_vm_info)"]
        KernelMadvise["Kernel Directives: posix_madvise(DONTNEED / WILLNEED) & mincore(2)"]
    end

    subgraph AMUQuant_Engine ["AMU-QUANT Co-Adaptive Controller (Native C++17)"]
        Sampling["Periodic Telemetry Sampler (K-token stride)"]
        ScoreEngine["Predictive Memory Pressure Estimator: Φ_t"]
        RegimeLogic{"Predictive Cost-Benefit Arbitration: min J(a)"}
        R1["Regime 1: High-Throughput (Max Prefetch, Unconstrained Decode)"]
        R2["Regime 2: Balanced Offload (Proactive KV Working-Set Compaction)"]
        R3["Regime 3: Ultra-Low Survival (Emergency Clamping + Single-Layer Advisory Paging)"]
    end

    subgraph Llama_CPP ["Inference Engine: llama.cpp"]
        MmapWeights["Model Weights (Unified GGUF via mmap)"]
        ActiveKVCache["Active Physical KV-Cache Working Set (Bounded O(1))"]
        Surgery["KV Surgery: llama_memory_seq_rm & seq_add (RoPE Shift)"]
    end

    subgraph History_Memory ["Canonical History & Context Reconstruction"]
        HistoryStore["Canonical History Store (Verbatim Turn Archive)"]
        ReconstructPrompt["Query-Driven Context Reconstruction Block"]
    end

    VM_Stat --> Sampling
    Sampling --> ScoreEngine
    ScoreEngine --> RegimeLogic
    RegimeLogic -->|Low Pressure & Within Budget| R1
    RegimeLogic -->|Moderate Pressure or Exceeds Budget| R2
    RegimeLogic -->|Critical RAM Pressure / Host Starvation| R3

    R1 -.->|Prefetch Upcoming Layer Pages| KernelMadvise
    R2 -->|Trigger Bounded KV Pruning & RoPE Shift| Surgery
    R3 -->|Trigger Emergency Window Squeeze & Single Layer Execution| Surgery
    Surgery -->|Mark Evicted Turn IDs| HistoryStore
    HistoryStore -->|Lexical Query Match on Evicted Context| ReconstructPrompt
    ReconstructPrompt -->|Re-inject Factual Needles into User Prompt| ActiveKVCache

    KernelMadvise --> MmapWeights
```

---

## 3. Directory Layout

```text
docs/
├── README.md                                 # Technical documentation index
├── 01_SYSTEM_ARCHITECTURE.md                 # Subsystem design & execution flow
├── 02_MATHEMATICAL_MODEL_AND_CONTROLLER.md   # Formal equilibrium & predictive metric
├── 04_EXPERIMENTAL_METHODOLOGY.md            # Benchmark protocols, hardware, & metrics
└── 05_LITERATURE_REVIEW_AND_REFERENCES.md    # Critical analysis & citations
```
