# AMU-QUANT System Architecture

This document presents the in-depth technical architecture of **AMU-QUANT**, a host-aware co-adaptive runtime engine for LLM inference on memory-constrained devices, integrating runtime memory pressure arbitration, transactional KV-cache surgery, and episodic vector memory compression.

---

## 1. High-Level Architecture Overview

AMU-QUANT consists of five integrated subsystems executing entirely in-process in native C++17 with optional Rust FFI bindings:

```mermaid
flowchart TB
    subgraph Host_OS ["Operating System Kernel & Host Headroom"]
        ProcMem["mach_vm_info / sysctl / procfs (Process RSS, Free RAM, Swap)"]
        KernelMadvise["Page Directives: posix_madvise (DONTNEED / WILLNEED)"]
    end

    subgraph AMU-QUANT_Core ["AMU-QUANT Core Runtime Engine (C++17)"]
        subgraph Mem_Controller ["Subsystem 1: Host-Aware Adaptive Controller"]
            OS_Sampler["Host Telemetry Sampler (K-token stride)"]
            PredictiveEngine["Dual-Budget Memory Pressure Estimator (Φ_t)"]
            RegimeStateMachine["Tri-Modal Hysteresis State Machine"]
        end

        subgraph KV_Manager ["Subsystem 2: Transactional KV-Cache Surgery Engine"]
            TokenMap["TokenIndexMap: Physical-to-Logical ID Translation"]
            KVSurgeon["2-Phase Commit Eviction: seq_rm & seq_add"]
            ContextBuffer["Physical KV Working Set (llama_context)"]
        end

        subgraph History_Manager ["Subsystem 5: Canonical Conversation History & Retrieval-to-Prompt"]
            HistoryStore["CanonicalHistoryStore: Turn Texts, Roles, Offsets"]
            EvictionTracker["Eviction Range Tracker (mark_evicted_range)"]
            HybridRetriever["Hybrid Semantic & Lexical Query Matcher"]
            PromptReconstructor["System Memory Context Injector"]
        end

        subgraph Layer_Manager ["Subsystem 4: Virtual Memory Page Manager (Future Extension)"]
            LayerRegistry["Layer Address Space Registry"]
            PagePaging["posix_madvise Virtual Memory Paging"]
        end
    end

    subgraph Vector_Memory ["Subsystem 3: Episodic Vector Memory (Canonical C++ / Optional Rust FFI)"]
        C_ABI_Bridge["turbovec-ffi C-ABI Interface"]
        TurboIndex["TurboVec: Block-Orthogonal Lloyd-Max Quantized Index"]
        SIMD_Search["AVX2 / ARM NEON SIMD Dot-Product Search Engine"]
    end

    %% Telemetry & Control Flows
    ProcMem -->|Process RSS, Host Available RAM, Swap| OS_Sampler
    OS_Sampler --> PredictiveEngine
    PredictiveEngine --> RegimeStateMachine
    RegimeStateMachine -->|Regime 1: Normal, Regime 2: Offload, Regime 3: Survival| KV_Manager

    %% Transactional KV Surgery Flow
    ContextBuffer -->|Physical Tokens [p_start, p_end)| KVSurgeon
    KVSurgeon -->|Phase 1: Query Logical IDs| TokenMap
    KVSurgeon -->|Phase 1: Insert Hidden States h_t| C_ABI_Bridge
    C_ABI_Bridge --> TurboIndex
    TurboIndex -->|Verify TURBOVEC_STATUS_OK| KVSurgeon
    KVSurgeon -->|Phase 2: seq_rm & seq_add| ContextBuffer
    KVSurgeon -->|Phase 2: Update Translation Mapping| TokenMap
    KVSurgeon -->|On Commit: Eviction Range Callback| EvictionTracker

    %% Retrieval & Context Reconstruction
    UserQuery["Incoming User Query / Agent Turn"] --> HybridRetriever
    HybridRetriever -->|Query Embeddings| SIMD_Search
    SIMD_Search -->|Top-K Matching Token IDs & Cosine Scores| HybridRetriever
    HybridRetriever -->|Query Keywords| HistoryStore
    EvictionTracker -.->|Filter Only Evicted Turns| HistoryStore
    HistoryStore -->|Retrieved Historical Facts| PromptReconstructor
    PromptReconstructor -->|Inject Structured System Memory| ContextBuffer
```

---

## 2. Core Subsystems Specification

### 2.1 Subsystem 1: Dual-Budget Host-Aware Adaptive Controller
The controller operates as the primary decision-maker, evaluated synchronously every $K$ generated tokens ($K \in [4, 16]$) during autoregressive decoding.

* **Key Responsibilities:**
  1. Gathers dual-budget host telemetry: Process Resident Set Size ($\mathcal{U}_{\text{process}}$), host available RAM ($\mathcal{U}_{\text{system}}$), and kernel swap thrashing ($\mathcal{P}_{\text{swap}}$).
  2. Evaluates the Host-Aware Pressure Metric:
     $$\Phi_t = \max(\mathcal{U}_{\text{process}}, \mathcal{U}_{\text{system}}) + \beta \cdot \mathcal{P}_{\text{growth}} + \gamma \cdot \mathcal{P}_{\text{swap}}$$
  3. Transitions between three operational regimes: **Regime 1 (High-Throughput)**, **Regime 2 (Balanced Offload)**, and **Regime 3 (Ultra-Low Survival)**.
  4. Enforces a hysteresis guard band ($\delta_{\text{hyst}}$) to prevent flapping between operational modes.

### 2.2 Subsystem 2: Transactional KV-Cache Surgery & TokenIndexMap
Unlike traditional methods that drop tokens irrevocably into trash (e.g., StreamingLLM, H2O), AMU-QUANT performs safe transactional eviction:

* **TokenIndexMap**:
  - Translates shifting physical KV-cache slots (`0 .. N-1`) to immutable logical token sequence IDs (`0 .. M-1`).
  - Ensures that when middle tokens are excised and trailing tokens are shifted down via `llama_memory_seq_add(..., delta)`, the representation extractor and vector memory always reference immutable logical tokens without index divergence.
* **Two-Phase Transactional Commit**:
  - **Phase 1 (Validation & Staging)**: Resolves logical IDs for all tokens in candidate slice $[p_{\text{start}}, p_{\text{end}})$. Extracts hidden-state representations $\mathbf{h}_t \in \mathbb{R}^d$ and verifies non-null pointers. Inserts vectors into `turbovec` and validates `TURBOVEC_STATUS_OK`. If ANY insertion fails or representation is missing, eviction aborts immediately with zero modifications to the physical KV cache.
  - **Phase 2 (Physical Pruning & Realignment)**: Invokes `seq_rm` and `seq_add` to excise physical slots and close the sequence gap. Prunes `TokenIndexMap` to reflect the new contiguous physical layout.
  - **Eviction Callback**: Notifies the `CanonicalHistoryStore` of the exact excised logical token span $[p_{\text{start}}, p_{\text{end}})$, flagging the corresponding historical conversation turns as needing active prompt reconstruction upon query retrieval.

### 2.3 Subsystem 3: Episodic Vector Memory (Canonical C++ / Optional Rust FFI)
Episodic memory archives high-dimensional hidden states ($\mathbf{h}_t \in \mathbb{R}^d$) into compressed vector memory:
- **Canonical Default**: Pure C++17 SIMD implementation utilizing AVX2 / ARM NEON intrinsics with zero external dependencies.
- **Optional Rust Backend**: `turbovec-ffi` static library compiled via Cargo when `-DAMUQUANT_USE_RUST_BACKEND=ON`.
- **Block-Orthogonal Rotation**: Multiplies incoming vectors with a seeded block-orthogonal Householder matrix ($\mathbf{\Pi}_b \in \mathbb{R}^{B \times B}, B=64$) to normalize coordinate distributions into independent standard Gaussians prior to scalar quantization, as established by the TurboQuant theory.
- **Lloyd-Max Quantization**: Optimal centroid lookup tables ($4\text{-bit} \to 2\text{-bit}$) minimizing mean squared quantization distortion ($\mathbb{E}[|c_i - q_j|^2]$).
- **In-Place Requantization**: Under critical memory pressure (Regime 3), downscales 4-bit nibbles to 2-bit dibits in-place with minimal quantization distortion ($|c_i - q_j|^2$).

### 2.4 Subsystem 4: Virtual Memory Page Manager (Future Work)
`LayerManager` encapsulates OS virtual memory primitives via `posix_madvise(DONTNEED / WILLNEED)`:
- Validated as an isolated page management primitive via unit test suites (`test_amuquant_pipeline`).
- Direct integration into `llama.cpp`'s internal GGML compute graph forward execution is scoped as documented future research.
- Notifies the Darwin/Linux virtual memory manager that underlying physical pages can be reclaimed from the OS page cache without writeback penalties.

### 2.5 Subsystem 5: Canonical Conversation History & Context Reconstruction (Retrieval-to-Prompt)
Solves the fundamental trade-off between KV-cache pruning and factual hallucination:
- **Problem**: Excising tokens via `llama_memory_seq_rm()` severs attention pathways. Even though TurboVec retains semantic vector embeddings, the Transformer forward pass cannot compute direct multi-head attention over removed tokens without re-injecting them into prompt context.
- **Canonical History Store (`CanonicalHistoryStore`)**:
  - Preserves immutable ground-truth turn records (role, prompt text, response text, token index intervals).
  - Maintains `mark_evicted_range()` to track which turns currently have their Key-Value attention tensors absent from the physical working cache.
- **Query-Driven Hybrid Retrieval**:
  - Combines SIMD cosine vector search over `turbovec` with fast BM25/lexical token overlap over stored turns.
  - Filters candidate turns to identify facts that are currently evicted from the working KV-cache.
- **Context Reconstruction (Retrieval-to-Prompt)**:
  - Injects retrieved facts into the immediate prompt context formatted as structured episodic memory:
    ```markdown
    [RECONSTRUCTED CONTEXT FROM EPISODIC MEMORY (Turn #1)]:
    User: Project Name: Aegis, Version: 90412, Architect: Ridwan, Port: 8084.
    [END RECONSTRUCTED CONTEXT]
    ```
  - Re-establishes direct multi-head attention over historical facts within the current forward pass, restoring 100% factual accuracy while keeping physical KV-cache bounded to an $O(1)$ working window.

---

## 3. Autoregressive Execution Lifecycle

The diagram below illustrates the exact execution lifecycle for generating a single token under AMU-QUANT orchestration:

```mermaid
sequenceDiagram
    autonumber
    participant App as Application / Chat CLI
    participant Engine as AMUQuantEngine
    participant Ctrl as Controller (Subsystem 1)
    participant LM as LayerManager (Subsystem 2)
    participant KV as KVSurgeon (Subsystem 3)
    participant TV as turbovec-ffi (Subsystem 4)
    participant Kernel as OS Kernel VM

    App->>Engine: Generate Next Token (Step t)
    Engine->>Ctrl: on_token_step(step_idx, seq_len)
    Ctrl->>Kernel: Sample Available RAM & Process RSS
    Kernel-->>Ctrl: Telemetry Snapshot (MemoryStats)
    Ctrl->>Ctrl: Compute Predictive Score Φ_t & Evaluate FSM

    alt Transition to Regime 2 / 3 (Memory Pressure)
        Ctrl->>KV: Trigger Context Eviction
        KV->>TV: turbovec_insert_zero_copy(chunk_id, float_ptr, len)
        TV-->>KV: Status OK
        KV->>Engine: llama_kv_cache_seq_rm & seq_add (RoPE Realignment)
    end

    loop For Each Transformer Layer (l = 0 to L - 1)
        Engine->>LM: begin_layer_forward(l)
        LM->>Kernel: posix_madvise(layer[l + depth], WILLNEED)
        Engine->>Engine: Compute Layer Attention & FFN
        Engine->>LM: end_layer_forward(l)
        opt If Aggressive Eviction Active (Regime 2 / 3)
            LM->>Kernel: posix_madvise(layer[l], DONTNEED)
        end
    end

    Engine->>App: Emit Decoded Token
```

---

## 4. Zero-Copy Pointer Passing & Memory Safety

To eliminate serialization overhead between C++ and Rust, AMU-QUANT adheres to strict zero-copy invariants across the foreign function boundary:

```c
// C++ callers pass raw pointers to existing contiguous float arrays
TurbovecStatus turbovec_insert_zero_copy(
    TurbovecIndexHandle* handle,
    uint64_t chunk_id,
    const float* float_ptr,
    size_t len
);
```

### Invariants:
1. **No Intermediate Buffer Copies:** Rust creates a transient, read-only slice (`std::slice::from_raw_parts(float_ptr, len)`) directly mapping C++ memory for immediate vector rotation and quantization into its own compact index structure.
2. **Panic Boundary Protection:** Every exported Rust C-ABI symbol is wrapped in `std::panic::catch_unwind(|| { ... })`. Any unhandled panic is intercepted at the FFI boundary and converted into a standard C enum status code (`TURBOVEC_ERR_PANIC_CAUGHT`), preventing undefined unwinding across foreign stack frames.
3. **Thread Safety:** The Rust handle wraps the vector index in an `std::sync::RwLock`, permitting concurrent multi-threaded SIMD queries while synchronizing rare insertion and requantization operations.
