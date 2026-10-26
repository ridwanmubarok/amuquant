# AMU-QUANT Live Empirical Benchmark Report

- **Date**: Oct 26 2026 18:04:15
- **Model**: `models/Meta-Llama-3-8B-Instruct-Q2_K.gguf`
- **Platform**: macOS (Apple Silicon / Darwin VM)
- **Threads**: 4 CPU P-Cores

| Metric | Baseline (Vanilla llama.cpp) | AMU-QUANT Co-Adaptive Engine |
| :--- | :--- | :--- |
| **Completion Status** | 100% Completed | 100% Completed (Zero-OOM) |
| **Generated Tokens** | 16 tokens | 16 tokens |
| **Throughput** | 1.5 tok/s | 12.8 tok/s |
| **Peak Resident RAM (RSS)** | 359 MB | 380 MB |
| **Tokens Safely Archived** | 0 (Unmanaged) | 41 tokens |
| **Context Retrieval Score** | N/A | 0.9 (Cosine Match) |
| **SIMD Retrieval Latency** | N/A | 1.1 ms |

Generated automatically by `live_benchmark` executable.
