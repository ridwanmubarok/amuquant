# Academic Novelty Formulation and Literature Comparative Matrix

## 1. Executive Summary & Thesis Research Scope

In the context of an undergraduate thesis (Tugas Akhir S1) in Computer Science / Software Engineering, establishing scientific novelty requires distinguishing between standard engineering components and genuine algorithmic contributions.

* **Non-Novel Baseline Components**:
  * *Vector Quantization (TurboQuant)*: Scalar Lloyd-Max quantization applied to embeddings has clear mathematical and empirical precedents in literature (Zandieh et al., ICLR 2026; Codrai, 2026).
  * *Attention KV-Cache Eviction*: Dropping historical sequence tokens while preserving attention sinks and recent local windows was pioneered by StreamingLLM (Xiao et al., ICLR 2024), H2O (Zhang et al., NeurIPS 2023), and ThinKV (ICLR 2026).
  * *OS Memory Sampling*: Polling `task_info`, `mach_vm_info`, or `/proc/meminfo` is standard operating systems telemetry.

* **Defensible Core Research Novelty**:
  * **The Predictive Cost-Benefit Multi-Action Arbitration Policy (PCBM-Policy)**:
    A runtime decision algorithm that monitors physical host virtual memory and memory growth velocity, projects memory saturation over a lookahead temporal horizon ($\tau_{\text{horizon}}$), and selects optimal mitigation actions from a multi-action mitigation space (transactional KV pruning, in-place 4-bit $\to$ 2-bit Lloyd-Max vector requantization, prefetch depth throttling, and advisory OS paging) by minimizing a joint cost-benefit utility loss function:
    $$ \mathcal{J}(\mathbf{a}) = \text{Risk}_{\text{OOM}}(\mathbf{a}; \widehat{M}(t+\tau)) + \lambda_{\text{lat}} \cdot \mathcal{C}_{\text{lat}}(\mathbf{a}) + \lambda_{\text{qual}} \cdot \mathcal{L}_{\text{qual}}(\mathbf{a}) $$
    under strict **transactional commit barrier** invariants.

---

## 2. Systematic State-of-the-Art Literature Matrix

The table below contrasts AMU-QUANT with the most relevant contemporary research across six key dimensions:

| Dimension | Ada-KV (arXiv:2407.11550) | IndexMem (ICLR 2025) | mzCache (MobiCom '26) | StreamingLLM / H2O | AirLLM (2024) | AMU-QUANT (This Work) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Primary Goal** | Minimize attention approximation loss | Preserve long-context recall across benchmarks | Accelerate Time-to-First-Token (TTFT) upon app resume | Prevent attention cache unbounded expansion | Run 70B models on low-VRAM GPUs via disk paging | Zero-OOM inference survival under dynamic host memory starvation |
| **Decision Signal** | Intra-model attention head entropy and score variance | Learned neural attention indexer network | Android mobile OS activity lifecycle events (`onPause`, `onResume`) | Local attention window and heavy-hitter token scores | Static single-layer sequential pipeline | Host physical RAM, process RSS growth velocity ($\dot{M}$), and swap thrashing |
| **Temporal Prediction** | None (Static budget allocated at prefill) | None (Token importance scoring per step) | Reactive (Restores state after OS eviction occurred) | None (Static heuristic window $W$) | None (Static sequential execution) | **Predictive Lookahead**: Projections over $\tau_{\text{horizon}}$ to anticipate OS swapping |
| **Action Space** | Single action: Varying token retention budget across heads | Single action: Evict token and write to latent recurrent state | Two actions: GPU-CPU buffer swap and concurrent restoration | Single action: Drop tokens permanently to trash | Single action: Sequential layer-by-layer weight loading | **Multi-Action Portfolio**: (1) Transactional KV pruning, (2) In-place 4b $\to$ 2b requantization, (3) Prefetch throttling, (4) Advisory paging |
| **Objective Function** | Theoretical attention bound $\min \|A - \widehat{A}\|$ | Latent state reconstruction loss | Minimize resume latency $\min \text{TTFT}$ | Maximize local attention mass | Minimize per-step GPU VRAM footprint | **Composite Utility Function**: $\arg\min_{\mathbf{a}} [\text{Risk}_{\text{OOM}} + \lambda_{\text{lat}}\mathcal{C}_{\text{lat}} + \lambda_{\text{qual}}\mathcal{L}_{\text{qual}}]$ |
| **Safety Invariant** | None (Dropped KV is lost) | Soft latent representation | Storage-backed re-read fallback | Irreversible context loss (*Evict-to-Trash*) | Static storage dependency | **Batch Commit Barrier**: Guaranteed rollback if vector index insertion fails |
| **Target Substrate** | Cloud GPU Clusters (NVIDIA A100/H100) | Cloud GPU Clusters (NVIDIA A100/H100) | Mobile Smartphones (Android SoCs) | Server GPUs & Edge Devices | Commodity Workstations | Commodity Unified Memory Hardware (Apple Silicon, ARM64, Linux) |

---

## 3. In-Depth Comparative Mechanism Analysis

### 3.1 AMU-QUANT vs. mzCache (MobiCom '26 / arXiv:2609.01338)
* **Problem Addressed**: `mzCache` focuses specifically on *mobile multitasking app switching*. When a smartphone user navigates away from an LLM app to open a browser or camera, the mobile OS triggers memory warnings or terminates background pages. When the user switches back, `mzCache` utilizes fine-grained shared buffers and unified memory to rapidly restore evicted KV cache and weights without stalling the initial generation.
* **Key Distinction from AMU-QUANT**:
  1. *Operating Domain*: `mzCache` is an *inter-application restoration manager* triggered by OS lifecycle events (app pause/resume). AMU-QUANT is an *intra-generation active arbitration controller* that operates token-by-token during live autoregressive decoding.
  2. *Contention Context*: `mzCache` responds *after* the OS has reclaimed memory. AMU-QUANT continuously samples host memory pressure ($\Phi_t$), anticipates impending kernel page thrashing before it happens, and adjusts KV bit-width and prefetching dynamically.
  3. *Vector Indexing*: `mzCache` does not perform episodic semantic vector quantization (no Lloyd-Max 4-bit/2-bit index); it restores raw uncompressed buffers from backing storage.

### 3.2 AMU-QUANT vs. Ada-KV (arXiv:2407.11550)
* **Problem Addressed**: `Ada-KV` observes that different attention heads have heterogeneous importance; some heads focus heavily on global context, while others focus on immediate local tokens. It computes an adaptive budget allocation across attention heads to minimize the upper bound of attention error.
* **Key Distinction from AMU-QUANT**:
  1. *Signal Boundary*: `Ada-KV` operates entirely inside the transformer attention mathematics. It is completely unaware of the operating system host environment, available RAM, swap thrashing, or background application memory demands.
  2. *Action Uniformity*: `Ada-KV` can only adjust the number of retained KV tokens per head. It cannot throttle layer prefetching, dynamically downscale vector precision, or interface with OS virtual memory paging (`madvise`).
  3. *Complementary Integration*: AMU-QUANT's runtime policy can subsume head-wise budgeting as an internal pruning action while providing the external host-awareness that `Ada-KV` lacks.

### 3.3 AMU-QUANT vs. IndexMem (ICLR 2025)
* **Problem Addressed**: `IndexMem` seeks to solve the "irreversible forgetting" problem of KV eviction in ultra-long context documents. It trains a lightweight neural network indexer to select tokens and compresses evicted KV pairs into a continuous latent memory vector.
* **Key Distinction from AMU-QUANT**:
  1. *Resource Footprint*: `IndexMem` requires training and executing an auxiliary neural network during inference, introducing parameter overhead and requiring gradient training for every model architecture. AMU-QUANT uses training-free, non-uniform Lloyd-Max scalar quantization accelerated via zero-dependency CPU SIMD (AVX2 / ARM NEON).
  2. *Decision Driver*: `IndexMem` compresses tokens based on input sequence semantics, not hardware memory pressure. If the host system experiences sudden external memory starvation, `IndexMem` has no mechanism to downscale its memory footprint dynamically.
  3. *Dynamic Requantization*: AMU-QUANT can autonomously execute in-place 4-bit $\to$ 2-bit compaction on its episodic index when memory pressure breaches critical safety thresholds.

---

## 4. Formal Mathematical Formulation of PCBM-Policy

### 4.1 System State Representation
At evaluation step $k$ (every $K$ decoded tokens), the controller observes the 5-dimensional environment state vector:
$$ \mathcal{S}_k = \left( M_{\text{proc}}(k),\, M_{\text{avail}}(k),\, M_{\text{swap}}(k),\, \dot{M}_{\text{proc}}(k),\, N_{\text{seq}}(k) \right) $$

The predictive memory saturation score $\Phi_k$ is evaluated as:
$$ \Phi_k = \max\left( \frac{M_{\text{proc}}}{M_{\text{limit}}},\, \frac{M_{\text{total}} - M_{\text{avail}}}{M_{\text{total}} - M_{\text{reserved}}} \right) + \beta \cdot \mathcal{P}_{\text{growth}} + \gamma \cdot \mathcal{P}_{\text{swap}} $$

### 4.2 Candidate Action Portfolio ($\mathcal{A}$)
The controller chooses an action vector $\mathbf{a} = (W_{\text{prune}},\, b_{\text{quant}},\, d_{\text{prefetch}},\, h_{\text{madvise}}) \in \mathcal{A}$:

1. **Action $\mathbf{a}_0$ (Nominal High-Throughput)**:
   * $W_{\text{prune}} = 0$ (No KV eviction).
   * $b_{\text{quant}} = \text{None}$ (Retain FP16 KV).
   * $d_{\text{prefetch}} = 3$ (Maximal layer prefetching).
   * $h_{\text{madvise}} = \text{WILLNEED}$ (Aggressive page caching).

2. **Action $\mathbf{a}_1$ (Mild Batched Eviction to 4-bit Index)**:
   * $W_{\text{prune}} = N_{\text{middle}}$ (Prune middle tokens past attention sink).
   * $b_{\text{quant}} = 4\text{-bit}$ (Lloyd-Max scalar quantization, $7.8\times$ compression).
   * $d_{\text{prefetch}} = 1$ (Moderate prefetch).
   * $h_{\text{madvise}} = \text{DONTNEED}$ on evicted layer segments.

3. **Action $\mathbf{a}_2$ (Survival In-Place Compaction)**:
   * $W_{\text{prune}} = N_{\text{aggressive}}$ (Clamp recent window to $W_{\text{min}} = 16$).
   * $b_{\text{quant}} = 2\text{-bit}$ (In-place requantization of vector memory, $15.5\times$ compression).
   * $d_{\text{prefetch}} = 0$ (Strict single-layer execution).
   * $h_{\text{madvise}} = \text{DONTNEED}$ immediate page release.

### 4.3 Composite Cost-Benefit Utility Objective Function
For each candidate action $\mathbf{a}_i \in \mathcal{A}$, the controller evaluates the expected system utility loss:
$$ \mathcal{J}(\mathbf{a}_i) = \mathcal{R}_{\text{OOM}}(\mathbf{a}_i) + \lambda_{\text{lat}} \cdot \mathcal{C}_{\text{lat}}(\mathbf{a}_i) + \lambda_{\text{qual}} \cdot \mathcal{L}_{\text{qual}}(\mathbf{a}_i) $$

Where:
* **$\mathcal{R}_{\text{OOM}}(\mathbf{a}_i)$ (Projected Out-of-Memory Risk)**:
  Let projected host RAM deficit at lookahead horizon $\tau$ be:
  $$ \Delta_{\text{deficit}}(\mathbf{a}_i) = \left[ M_{\text{proc}}(k) + \dot{M}_{\text{proc}} \cdot \tau + M_{\text{ext\_shock}} - \Delta M_{\text{reclaimed}}(\mathbf{a}_i) \right] - M_{\text{limit}} $$
  The risk penalty is evaluated via a steep sigmoid barrier function:
  $$ \mathcal{R}_{\text{OOM}}(\mathbf{a}_i) = \frac{\text{Penalty}_{\text{OOM}}}{1 + \exp\left(-\frac{\Delta_{\text{deficit}}(\mathbf{a}_i)}{\sigma}\right)} $$
  When projected memory exceeds $M_{\text{limit}}$, $\mathcal{R}_{\text{OOM}} \to \text{Penalty}_{\text{OOM}} = 10.0$, dominating all other considerations.

* **$\mathcal{C}_{\text{lat}}(\mathbf{a}_i)$ (Execution Latency Overhead)**:
  * For $\mathbf{a}_0$: $\mathcal{C}_{\text{lat}} = 0.0$ (Zero adaptation overhead).
  * For $\mathbf{a}_1$: $\mathcal{C}_{\text{lat}} = \frac{t_{\text{SIMD\_quant}}}{t_{\text{token\_decode}}} \approx 0.05$ (Sub-millisecond SIMD insertion).
  * For $\mathbf{a}_2$: $\mathcal{C}_{\text{lat}} = \frac{t_{\text{requant}} + t_{\text{io\_stall}}}{t_{\text{token\_decode}}} \approx 0.20$ (Cost of in-place vector bit-shifting and single-layer storage IO).

* **$\mathcal{L}_{\text{qual}}(\mathbf{a}_i)$ (Representation Quality Loss)**:
  * For $\mathbf{a}_0$: $\mathcal{L}_{\text{qual}} = 0.0$ (Exact FP16 attention preservation).
  * For $\mathbf{a}_1$: $\mathcal{L}_{\text{qual}} = 0.14$ (Expected 4-bit Lloyd-Max distortion $\approx 1 - \text{CosineScore} = 1 - 0.86$).
  * For $\mathbf{a}_2$: $\mathcal{L}_{\text{qual}} = 0.20$ (Expected 2-bit Lloyd-Max distortion $\approx 1 - 0.80$).

The optimal arbitration decision is chosen as:
$$ \mathbf{a}^* = \arg\min_{\mathbf{a} \in \mathcal{A}} \mathcal{J}(\mathbf{a}) $$

---

## 5. Formal Undergraduate Thesis Defense Hypothesis

For the academic examination committee (Sidang Tugas Akhir S1), the contribution is stated with exact scientific rigor:

> **Thesis Statement**:  
> *"In autoregressive Large Language Model inference under dynamic host memory contention, a Predictive Cost-Benefit Multi-Action Arbitration Policy (PCBM-Policy) that jointly evaluates projected kernel memory exhaustion, adaptation latency overhead, and vector quantization distortion achieves provable Zero-OOM completion and superior throughput parity compared to static threshold-based finite state machines and host-blind attention-eviction heuristics."*

### Key Testable Sub-Hypotheses:
1. **Hypothesis 1 (Predictive Adaptation Superiority)**: Evaluating growth velocity $\dot{M}$ over horizon $\tau_{\text{horizon}}$ triggers pre-emptive mitigation before kernel swap page-thrashing occurs, avoiding throughput collapse compared to reactive threshold baselines.
2. **Hypothesis 2 (Multi-Action Cost Trade-Off)**: Optimizing $\mathcal{J}(\mathbf{a})$ dynamically suppresses unnecessary vector requantization and maintains maximal prefetch depth during transient non-critical spikes, maximizing throughput while guaranteeing survival.
3. **Hypothesis 3 (Zero Context Loss via Transactional Commit)**: Enforcing a batch commit barrier on episodic vector memory guarantees zero unarchived token loss during eviction, preserving $>85\%$ semantic context recall and 100% factual needle recovery.
