# AMU-QUANT Live Empirical Benchmark Report (3-Way Methodological Parity)

- **Model**: `models/Meta-Llama-3-8B-Instruct-Q2_K.gguf`
- **Architecture**: 32 Layers, Dim 4096
- **Target Generation Horizon**: 32 tokens

| Evaluation Metric | Baseline 1 (Raw Vanilla) | Baseline 2 (With Embeddings) | AMU-QUANT Co-Adaptive Engine |
| :--- | :--- | :--- | :--- |
| **Target Horizon Completed** | Yes (100%) | Yes (100%) | Yes (100%) |
| **Tokens Generated** | 32/32 | 32/32 | 32/32 |
| **Peak Physical RSS** | 370 MB | 394 MB | 397 MB |
| **Inference Throughput** | 2.71 tok/s | 11.45 tok/s | 11.22 tok/s |
| **KV Memory Reclaimed** | 0 KB | 0 KB | 7680 KB |
| **Tokens Safely Archived** | 0 | 0 | 60 |
| **Semantic Recall Score** | N/A | N/A | 0.8468 |
| **SIMD Retrieval Latency** | N/A | N/A | 0.510 ms |

Generated automatically by `live_benchmark` on Apple Silicon / Metal.
