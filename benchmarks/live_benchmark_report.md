# AMU-QUANT Live Empirical Benchmark Report (3-Way Methodological Parity)

- **Model**: `models/Meta-Llama-3-8B-Instruct-Q2_K.gguf`
- **Architecture**: 32 Layers, Dim 4096
- **Target Generation Horizon**: 64 tokens

| Evaluation Metric | Baseline 1 (Raw Vanilla) | Baseline 2 (With Embeddings) | AMU-QUANT Co-Adaptive Engine |
| :--- | :--- | :--- | :--- |
| **Target Horizon Completed** | Yes (100%) | Yes (100%) | Yes (100%) |
| **Tokens Generated** | 64/64 | 64/64 | 64/64 |
| **Peak Physical RSS** | 367 MB | 392 MB | 382 MB |
| **Inference Throughput** | 12.69 tok/s | 13.61 tok/s | 12.42 tok/s |
| **KV Memory Reclaimed** | 0 KB | 0 KB | 8704 KB |
| **Tokens Safely Archived** | 0 | 0 | 68 |
| **Semantic Recall Score** | N/A | N/A | 0.8468 |
| **SIMD Retrieval Latency** | N/A | N/A | 1.109 ms |

Generated automatically by `live_benchmark` on Apple Silicon / Metal.
