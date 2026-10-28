# AMU-QUANT: Host-Aware Co-Adaptive Memory Management for Edge Large Language Model Inference

**An Undergraduate Thesis Research Project in Computer Science / Informatics (Skripsi S1)**

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![C++17](https://img.shields.io/badge/C%2B%2B-17-00599C.svg?logo=c%2B%2B)](CMakeLists.txt)
[![CMake](https://img.shields.io/badge/CMake-3.20+-green.svg?logo=cmake)](CMakeLists.txt)

---

## 1. Thesis Overview & Academic Positioning

### 1.1 Thesis Title
**"Host-Aware Co-Adaptive Memory Management for Edge Large Language Model Inference: Achieving Zero-OOM Resilience and Real-Time Throughput on Resource-Constrained Hardware"**

### 1.2 Problem Formulation (Latar Belakang & Rumusan Masalah)
Deploying modern autoregressive Large Language Models (3B to 8B parameters) directly on consumer edge hardware (such as 8 GB unified memory laptops or constrained personal computers) encounters two major architectural failure modes in existing state-of-the-art inference engines:

1. **Unbounded KV-Cache Growth and Passive Virtual Memory in llama.cpp**:
   - `llama.cpp` relies on static memory-mapping (`mmap`) and passive operating system kernel page replacement.
   - Its Key-Value (KV) cache grows monotonically ($O(N)$) with context length.
   - The engine is completely blind to external host memory pressure: when background applications (e.g., web browsers, compiler jobs, operating system daemons) or extended multi-turn conversations saturate physical RAM, the operating system kernel either collapses into catastrophic swap thrashing or invokes the Out-Of-Memory (OOM) killer, terminating the inference process with `SIGKILL` (Exit Code 137).
2. **Severe Disk I/O Latency Penalties in AirLLM**:
   - AirLLM avoids physical memory exhaustion by streaming model layers sequentially from secondary storage (SSD/disk) into RAM/VRAM on every forward pass.
   - While this prevents OOM crashes, it introduces an insurmountable disk I/O bottleneck, reducing generation throughput to less than 0.5 to 1.0 tokens per second.
   - Furthermore, AirLLM is memory-unaware: it executes slow disk streaming indiscriminately even when gigabytes of physical RAM are completely free.

### 1.3 The AMU-QUANT Solution
AMU-QUANT bridges the gap between the speed of `llama.cpp` and the crash resilience of `AirLLM`. Implemented in native C++17 with zero external language dependencies, AMU-QUANT introduces **Host-Aware Co-Adaptive Memory Management**:
- **Closed-Loop OS Telemetry**: Continuously samples Darwin/Linux kernel metrics (`phys_footprint`, `available_ram`, swap usage velocity $\frac{dM}{dt}$) with sub-millisecond non-blocking overhead.
- **Predictive Multi-Budget Arbitration ($\Phi_t$ and $J(a)$)**: Antedates physical memory exhaustion by evaluating differential expansion velocity over a lookahead horizon ($\tau = 2.0\text{ s}$).
- **Dynamic Tri-Modal Operational Regimes**:
  - **Regime 1 (High-Throughput Nominal)**: Aggressive weight prefetching (depth 3), unconstrained decoding, achieving ~25-27 tokens/second on Apple Silicon.
  - **Regime 2 (Balanced Offload)**: Controlled prefetch depth (depth 1), bounded dynamic physical KV-cache sliding window with attention-sink shielding, enforcing a constant $O(1)$ working set.
  - **Regime 3 (Ultra-Low Survival / Anti-Crash)**: Coordinated single-layer execution with immediate kernel page advisories (`posix_madvise(DONTNEED)` on inactive layer pages) and compressed KV working set, guaranteeing zero-OOM survival under extreme host memory contention without dropping into disk-bound stall speeds.
- **Physical Page Verification (`mincore(2)`)**: Directly queries kernel page tables to verify physical page frame residency, establishing an empirical distinction between virtual scheduling intent and physical memory footprint.

---

## 2. Comparative Analysis: AMU-QUANT vs. Existing Baselines

| Dimension / Metric | **llama.cpp** (Baseline 1) | **AirLLM** (Baseline 2) | **AMU-QUANT** (Proposed Runtime) |
| :--- | :--- | :--- | :--- |
| **Execution Architecture** | Monolithic `mmap` with passive OS page cache | Serial layer-by-layer disk loading in Python | Host-Aware Co-Adaptive Memory Manager (C++17) |
| **Host System Awareness** | None (Blind to system RAM and swap pressure) | None (Blind to available host RAM) | Closed-Loop Telemetry: Samples OS kernel page stats & $\frac{dM}{dt}$ |
| **Memory Pressure Metric** | None (Static allocation) | None (Static disk-stream) | Mathematical Formulation: Composite score $\Phi_t \in [0, 1+]$ |
| **Weight Paging Strategy** | Relies entirely on passive OS kernel page replacement | Synchronous disk file reads per layer every forward pass | Tri-modal adaptive paging: `posix_madvise` WILLNEED / DONTNEED |
| **Residency Verification** | None | Manual layer allocation | Kernel page table audit via `mincore(2)` |
| **KV-Cache Complexity** | Linear growth $O(N)$ with sequence length | Full sequence KV-cache | Bounded $O(1)$ working set with attention sink shielding |
| **Throughput (Healthy RAM)** | High (~25.0 tok/s on Qwen 3B Apple Silicon) | Extremely low (< 0.5 - 1.0 tok/s) | High (~26.4 tok/s on Qwen 3B Apple Silicon) |
| **Throughput (Contended RAM)** | Collapses due to swap thrashing or fatal OOM | Constant low (< 0.5 tok/s) | Adaptive (~18-24 tok/s in Regime 2; resilient in Regime 3) |
| **Crash Resilience (Zero-OOM)**| Fragile: Fatal SIGKILL (137) under RAM pressure | Resistant to OOM, but unusable latency | **100% Anti-Crash Zero-OOM Guarantee** via predictive mitigation |
| **Runtime Stack** | C / C++ | Python + PyTorch + Safetensors | Native C++17 (Zero Rust, Zero Python overhead) |

---

## 3. System Architecture

```
+-----------------------------------------------------------------------------------+
|                           HOST OPERATING SYSTEM KERNEL                            |
|             (macOS Darwin / Linux Virtual Memory Subsystem & Page Tables)         |
+-----------------------------------------------------------------------------------+
       | (phys_footprint)               | (available_ram)             | (vm.swapusage)
       v                                v                             v
+-----------------------------------------------------------------------------------+
|                        AMU-QUANT SYSTEM TELEMETRY SAMPLER                         |
|                 (Non-blocking 50ms sampling of kernel physical state)             |
+-----------------------------------------------------------------------------------+
                                        |
                                        v
+-----------------------------------------------------------------------------------+
|                     PREDICTIVE CO-ADAPTIVE CONTROLLER (FSM)                       |
|           Evaluates Pressure Metric Phi_t and Minimizes Utility J(a)              |
+-----------------------------------------------------------------------------------+
         |                                |                               |
         | Regime 1                       | Regime 2                      | Regime 3
         v (Nominal)                      v (Balanced Offload)            v (Survival Anti-Crash)
+------------------------+  +-------------------------------+  +-------------------------------+
| Prefetch Depth: 3      |  | Prefetch Depth: 1             |  | Prefetch Depth: 0 (Single)   |
| Zero KV Eviction       |  | KV Sliding Window (Sink + Win)|  | POSIX_MADV_DONTNEED Inactive  |
| Full Throughput Speed  |  | RoPE Realignment (seq_add)    |  | Min KV Window (O(1) Bounds)   |
+------------------------+  +-------------------------------+  +-------------------------------+
                                        |
                                        v
+-----------------------------------------------------------------------------------+
|                  DYNAMIC KV SURGEON & ATTENTION SINK SHIELD                       |
|      Preserves initial prompt tokens [0, S) and recent context [N - W, N)         |
|      Prunes middle slots [S, N - W) via seq_rm and shifts RoPE positions         |
+-----------------------------------------------------------------------------------+
```

---

## 4. Mathematical Formulation

### 4.1 Predictive Host-Aware Memory Pressure Metric ($\Phi_t$)
Memory exhaustion cannot be avoided reliably by checking static thresholds after memory is already allocated. AMU-QUANT calculates a composite predictive score $\Phi_t$:

$$\Phi_t = \max(U_{\text{proc}}(t), U_{\text{host}}(t)) + \beta \cdot P_{\text{growth}}(t) + \gamma \cdot P_{\text{swap}}(t)$$

Where:
- **Process Saturation**: $U_{\text{proc}}(t) = \frac{M_{\text{process\_rss}}(t)}{M_{\text{limit}} - M_{\text{os\_reserved}}}$
- **Host System Depletion**: $U_{\text{host}}(t) = 1.0 + \frac{M_{\text{os\_reserved}} - M_{\text{available}}}{M_{\text{os\_reserved}}}$ when available RAM breaches the safety reserve floor.
- **Projected Expansion Velocity**: $P_{\text{growth}}(t) = \frac{1}{M_{\text{safe}}} \left(\frac{dM_{\text{rss}}}{dt} \cdot \tau\right)$, with lookahead horizon $\tau = 2.0\text{ s}$.
- **Kernel Swap Thrashing Penalty**: $P_{\text{swap}}(t)$ scales with active swap space utilization modulated by remaining host headroom.

### 4.2 Dynamic KV-Cache Working Set Sizing
For any transformer architecture, the exact physical memory footprint of a single token's KV-cache across all layers is computed directly from model hyperparameters:

$$\text{Bytes Per Token} = 2 \times N_{\text{layers}} \times N_{\text{kv\_heads}} \times D_{\text{head}} \times B_{\text{element}}$$

For Qwen 2.5 3B ($N_{\text{layers}} = 36$, $N_{\text{kv\_heads}} = 2$, $D_{\text{head}} = 128$, $B_{\text{element}} = 2\text{ bytes}$ for FP16):
$$\text{Bytes Per Token} = 2 \times 36 \times 2 \times 128 \times 2 = 36,864\text{ bytes (36 KB / token)}$$

Without AMU-QUANT, context expansion to 4,096 tokens consumes **147.45 MB**, and 32,768 tokens consumes **1.18 GB** of additional physical RAM per user session. AMU-QUANT bounds active slots to a constant window ($S + W \le 64-128$ tokens), reducing KV-cache working memory to **2.3 - 4.6 MB** ($O(1)$ constant complexity).

### 4.3 Rotary Position Embedding (RoPE) Realignment
When middle tokens $[p_0, p_1)$ are excised from physical memory, remaining tokens at physical positions $p \ge p_1$ must be realigned to prevent attention coordinate drift. AMU-QUANT shifts remaining token keys and values:

$$\Delta p = -(p_1 - p_0)$$

This is committed directly through `llama_memory_seq_add(mem, 0, p1, -1, delta)`, guaranteeing that relative rotary distances between recent tokens remain mathematically coherent without recomputing previous keys and values.

---

## 5. Repository Structure

```
aegis/
├── CMakeLists.txt                 # Modern native C++17 build configuration
├── README.md                      # Academic thesis documentation and guide
├── include/
│   └── amuquant/
│       ├── types.hpp              # Core structures, Regime enum, and StepDecision
│       ├── telemetry.hpp          # Real-time macOS Darwin / Linux memory sampler
│       ├── layer_manager.hpp      # OS advisory paging and mincore(2) page audit
│       ├── kv_surgeon.hpp         # Dynamic physical sliding-window and RoPE realignment
│       ├── controller.hpp         # Predictive FSM and utility optimization engine
│       ├── engine.hpp             # Unified AMUQuantEngine runtime coordinator
│       ├── canonical_history.hpp  # Verbatim conversational dialogue store
│       └── gguf_parser.hpp        # Direct binary parser for GGUF model headers
├── src/
│   ├── telemetry.cpp              # Kernel mach/sysinfo telemetry implementation
│   ├── layer_manager.cpp          # posix_madvise and mincore physical verification
│   ├── kv_surgeon.cpp             # KV sequence pruning and token map tracking
│   ├── controller.cpp             # Phi_t and cost-benefit utility implementation
│   ├── amuquant_engine.cpp        # Central engine coordination
│   ├── canonical_history.cpp      # Conversation history tracking
│   └── gguf_parser.cpp            # GGUF tensor and hyperparameter reader
├── examples/
│   ├── amuquant_chat.cpp          # Interactive terminal chat with hardware HUD
│   └── simple_inference.cpp       # Minimal programmatic integration example
├── benchmarks/
│   ├── standalone_benchmark.cpp   # Zero-dependency reproducible memory benchmark
│   └── live_benchmark.cpp         # Empirical evaluation suite with libllama
└── tests/
    ├── test_controller.cpp        # Unit tests for FSM transitions and hysteresis
    └── test_amuquant_pipeline.cpp # End-to-end integration and mincore(2) verification
```

---

## 6. Build and Verification Instructions

### 6.1 Prerequisites
- Operating System: macOS (Apple Silicon / Intel) or Linux (x86_64 / aarch64).
- Compiler: Clang or GCC supporting C++17.
- Build System: CMake 3.20+.
- No Rust toolchain or external package managers required.

### 6.2 Compilation
```bash
# Configure build directory
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release

# Build all libraries, benchmarks, and interactive executables
cmake --build build -j
```

### 6.3 Running Unit Tests
```bash
ctest --test-dir build --output-on-failure
```
Expected output:
```
    Start 1: test_controller
1/2 Test #1: test_controller ..................   Passed    0.01 sec
    Start 2: test_amuquant_pipeline
2/2 Test #2: test_amuquant_pipeline ...........   Passed    0.01 sec

100% tests passed, 0 tests failed out of 2
```

### 6.4 Running the Standalone Memory Benchmark (Zero Dependencies)
```bash
./build/amuquant_benchmark --tokens 96
```
This evaluates physical KV memory complexity under identical sequence horizons:
- **Unmanaged Baseline (`llama.cpp` model)**: Linear growth $O(N)$ with unbounded memory accumulation.
- **AMU-QUANT Co-Adaptive Engine**: Strict $O(1)$ bounded working set with physical memory reclamation.

### 6.5 Interactive Terminal Chat (`amuquant_chat`)
To execute real-time, hardware-accelerated chat with the Qwen 2.5 3B model:
```bash
./build/amuquant_chat models/qwen2.5-3b-instruct-q4_k_m.gguf
```
During conversation, the status bar displays real-time telemetry:
```
[REGIME_1_HIGH_THROUGHPUT | Metal: 99/36L | P-Cores: 4T | FlashAttn: ON | RSS: 258 MB | Throughput: 26.4 tok/s | Context: 333 tokens]
```

Available interactive commands:
- `/stats`: Displays comprehensive telemetry, active regime, RSS, and page fault counters.
- `/regime`: Displays current operational regime and exact pressure score $\Phi_t$.
- `/clear`: Clears conversation history and resets the KV-cache.
- `/exit`: Exits the program cleanly.

---

## 7. License

This research codebase is distributed under the Apache License 2.0. See [LICENSE](LICENSE) for details.
