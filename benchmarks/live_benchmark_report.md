# AMU-QUANT Live Empirical Benchmark Report (3-Way Methodological Parity)

- **Model**: `models/Meta-Llama-3-8B-Instruct-Q2_K.gguf`
- **Architecture**: 32 Layers, Dim 4096
- **Target Generation Horizon**: 32 tokens

| Evaluation Metric | Baseline 1 (Raw Vanilla) | Baseline 2 (With Embeddings) | AMU-QUANT Co-Adaptive Engine |
| :--- | :--- | :--- | :--- |
| **Target Horizon Completed** | Yes (100%) | Yes (100%) | Yes (100%) |
| **Tokens Generated** | 32/32 | 32/32 | 32/32 |
| **Peak Physical RSS** | 353 MB | 377 MB | 380 MB |
| **Inference Throughput** | 8.93 tok/s | 14.34 tok/s | 13.04 tok/s |
| **KV Memory Reclaimed** | 0 KB | 0 KB | 5632 KB |
| **Tokens Safely Archived** | 0 | 0 | 44 |
| **Semantic Recall Score** | N/A | N/A | 0.8468 |
| **SIMD Retrieval Latency** | N/A | N/A | 1.508 ms |

Generated automatically by `live_benchmark` on Apple Silicon / Metal.
