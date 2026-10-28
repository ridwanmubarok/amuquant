# AMU-QUANT Live Empirical Benchmark Report: Multi-Scenario Evaluation

- **Model Target**: `models/qwen2.5-3b-instruct-q4_k_m.gguf`
- **Architecture**: 36 Layers, Dim 2048
- **Target Generation Horizon**: 64 tokens

## 2. Scenario 2: Multi-Turn Autonomous Agentic Workflow (ReAct Tool Loop)

| Agent Workflow Metric | Unmanaged Baseline Agent | AMU-QUANT Co-Adaptive Agent |
| :--- | :--- | :--- |
| **Total ReAct Turns Completed** | 8/8 Turns | **8/8 Turns (100%)** |
| **Total Ingested Tool Tokens** | 649 tokens | **649 tokens** |
| **Total Generated Agent Tokens** | 256 tokens | **256 tokens** |
| **Cumulative Context Horizon** | 905 tokens | **905 tokens** |
| **Final Physical KV Cache Slots** | 905 slots (Unbounded Expansion) | **273 slots (Strictly Bounded $O(1)$)** |
| **Physical KV Memory Reclaimed** | 0 KB | **22752 KB** |
| **Tokens Pruned via Attention Window** | 0 tokens | **632 tokens** |
| **Turn 6 Host Memory Shock Resilience** | Vulnerable to OOM Thrashing | **100% Anti-Crash Zero-OOM (Regime 3 Shift)** |
| **Generation Throughput** | 20.1 tok/s | **19.8 tok/s** |

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
