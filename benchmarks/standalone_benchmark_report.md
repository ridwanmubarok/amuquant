# AMU-QUANT Reproducible Benchmark Report

- **Platform**: POSIX / macOS Apple Silicon & Linux
- **Target Architecture**: Qwen 2.5 3B / Llama Transformer Spec
- **Embedding Dimension**: 1024 dim (FP16 token slots = 2 KB/token)
- **Transformer Layers**: 32 layers (32 MB physical anonymous mmap allocation)
- **Working Set Bounds**: Attention Sink $S = 16$, Recent Window $W = 64$, Minimum Survival Window $W_{\text{min}} = 32$
- **Stress Condition**: Deterministic Host Starvation & Swap Trajectory

## 1. Empirical Results Across Context Horizons

| Horizon ($N$) | Controller Policy | Peak KV (KB) | Final KV (KB) | KV Reclaimed | Active Slots | Operational Regime | Memory Complexity |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **64** | llama.cpp Baseline (Unmanaged) | 128 KB | 128 KB | 0 KB | 64 tokens | Regime 1 (Nominal) | O(N) Linear Growth |
| **64** | Static Threshold Controller | 102 KB | 102 KB | 26 KB | 51 tokens | Regime 3 (Survival) | O(1) Bounded (Reactive) |
| **64** | Predictive Cost-Benefit Controller | 102 KB | 102 KB | 26 KB | 51 tokens | Regime 3 (Survival) | O(1) Bounded (Proactive) |
| **256** | llama.cpp Baseline (Unmanaged) | 512 KB | 512 KB | 0 KB | 256 tokens | Regime 1 (Nominal) | O(N) Linear Growth |
| **256** | Static Threshold Controller | 166 KB | 102 KB | 410 KB | 51 tokens | Regime 3 (Survival) | O(1) Bounded (Reactive) |
| **256** | Predictive Cost-Benefit Controller | 310 KB | 102 KB | 410 KB | 51 tokens | Regime 3 (Survival) | O(1) Bounded (Proactive) |
| **512** | llama.cpp Baseline (Unmanaged) | 1024 KB | 1024 KB | 0 KB | 512 tokens | Regime 1 (Nominal) | O(N) Linear Growth |
| **512** | Static Threshold Controller | 256 KB | 102 KB | 922 KB | 51 tokens | Regime 3 (Survival) | O(1) Bounded (Reactive) |
| **512** | Predictive Cost-Benefit Controller | 526 KB | 102 KB | 922 KB | 51 tokens | Regime 3 (Survival) | O(1) Bounded (Proactive) |
| **1024** | llama.cpp Baseline (Unmanaged) | 2048 KB | 2048 KB | 0 KB | 1024 tokens | Regime 1 (Nominal) | O(N) Linear Growth |
| **1024** | Static Threshold Controller | 512 KB | 102 KB | 1946 KB | 51 tokens | Regime 3 (Survival) | O(1) Bounded (Reactive) |
| **1024** | Predictive Cost-Benefit Controller | 950 KB | 102 KB | 1946 KB | 51 tokens | Regime 3 (Survival) | O(1) Bounded (Proactive) |

## 2. Mathematical Complexity Analysis

- **Unmanaged Baseline ($llama.cpp$)**: Exhibits strictly monotonic $\mathcal{O}(N)$ memory scaling. At horizon $N=1024$, physical KV consumption reaches $2,048\text{ KB}$, with zero tokens pruned and zero memory reclaimed.
- **Static Threshold Controller**: Enforces $\mathcal{O}(1)$ working set bounds ($W_{\text{active}} \le S + W = 80\text{ tokens} = 160\text{ KB}$), but triggers reactively only after memory pressure crosses static thresholds ($\theta_{\text{low}} = 0.70$, $\theta_{\text{high}} = 0.88$).
- **Predictive Cost-Benefit Controller**: Proactively antedates pressure spikes via differential velocity $\frac{dM}{dt}$ and lookahead horizon $\tau = 2.0\text{ s}$, arbitrating optimal action regimes before physical memory exhaustion occurs.
- **Bounded Working Set**: Across all extended horizons ($N \in \{256, 512, 1024\}$), AMU-QUANT bounds active KV slots to $\le 80$ tokens (Regime 2) or $\le 48$ tokens (Regime 3), achieving $\mathcal{O}(1)$ complexity relative to sequence length.

Generated automatically by `amuquant_benchmark`.
