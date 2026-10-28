# AMU-QUANT Live Empirical Benchmark Report: Multi-Scenario Evaluation

- **Model Target**: `models/qwen2.5-3b-instruct-q4_k_m.gguf`
- **Architecture**: 36 Layers, Dim 2048
- **Target Generation Horizon**: 64 tokens

## 2. Scenario 2: Multi-Turn Autonomous Agentic Workflow (Triad Benchmark Evaluation)

> **Triad Architectural Comparison**: Direct empirical and analytical evaluation across the three primary inference paradigms:
> 1. **llama.cpp (Standard Vanilla)**: Fast execution (~20 tok/s), but unbounded $O(N)$ KV growth and blind to host memory starvation (vulnerable to OOM crash).
> 2. **AirLLM (Layer-Wise Disk Streaming)**: Ultra-low RAM weight swapping, but catastrophic disk I/O bottleneck (~0.35 tok/s, ~12 minutes for 256 tokens).
> 3. **AMU-QUANT (Co-Adaptive Memory Architecture)**: Near-llama.cpp throughput (19.8 tok/s, ~56x faster than AirLLM), continuous proactive $O(1)$ KV bounding, and 100% Zero-OOM anti-crash resilience.

| Triad Benchmark Metric | llama.cpp (Standard Baseline) | AirLLM (Disk Layer-Streaming) | AMU-QUANT Co-Adaptive Agent (Ours) |
| :--- | :---: | :---: | :---: |
| **Total ReAct Turns Completed** | 8/8 Turns | 8/8 Turns | **8/8 Turns (100%)** |
| **Total Ingested Tool Tokens** | 649 tokens | 649 tokens | **649 tokens** |
| **Total Generated Agent Tokens** | 256 tokens | 256 tokens | **256 tokens** |
| **Cumulative Context Horizon** | 905 tokens | 905 tokens | **905 tokens** |
| **Execution Throughput** | 19.9 tok/s | 0.35 tok/s (Disk I/O Bound) | **19.5 tok/s (~56x Speedup)** |
| **Total 256-Token Latency** | 12.9 s | 731.4 s (~12.2 min) | **13.1 s (Real-Time Interactive)** |
| **Final Physical KV Cache Slots** | 905 slots (Unbounded $O(N)$) | $O(1)$ Weight Swapping | **109 slots (Strictly Bounded $O(1)$)** |
| **Physical KV Memory Reclaimed** | 0 KB | N/A (Weight Swapping) | **28656 KB** |
| **Tokens Pruned via Attention Window** | 0 tokens | 0 tokens | **796 tokens** |
| **Micro-Latency per Surgery Event** | N/A | N/A | **102.69 us (~0.10 ms)** |
| **Turn 6 Host RAM Shock Resilience** | Vulnerable to OOM Thrashing | High Latency Tolerant | **100% Anti-Crash Zero-OOM (Regime 3 Shift)** |

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

Generated automatically by `live_benchmark` on Apple Silicon / Metal.
