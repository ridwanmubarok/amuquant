# AMU-QUANT Live Empirical Benchmark Report

- **Date**: Oct 26 2026 18:04:15
- **Model**: `models/Meta-Llama-3-8B-Instruct-Q2_K.gguf`
- **Platform**: macOS (Apple Silicon / Darwin VM)
- **Threads**: 4 CPU P-Cores

| Metric | Baseline (Vanilla llama.cpp) | AMU-QUANT Co-Adaptive Engine |
| :--- | :--- | :--- |
| **Completion Status** | 100% Completed | 100% Completed (Zero-OOM) |
| **Generated Tokens** | 32 tokens | 32 tokens |
| **Throughput** | 2.5 tok/s | 8.0 tok/s |
| **Peak Resident RAM (RSS)** | 361 MB | 383 MB |
| **Tokens Safely Archived** | 0 (Unmanaged) | 57 tokens |
| **Context Retrieval Score** | N/A | 0.9 (Cosine Match) |
| **SIMD Retrieval Latency** | N/A | 0.6 ms |

Generated automatically by `live_benchmark` executable.
