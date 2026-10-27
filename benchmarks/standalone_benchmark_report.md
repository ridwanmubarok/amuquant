# AMU-QUANT Standalone Benchmark Report (Zero External Dependencies)

- **Platform**: POSIX / macOS Apple Silicon
- **Embedding Dimension**: 1024 dim
- **Transformer Layers**: 32 layers (32 MB physical mmap allocation)

| Metric | Unmanaged Baseline | AMU-QUANT Co-Adaptive Engine |
| :--- | :--- | :--- |
| **Tokens Processed** | 96 tokens | 96 tokens |
| **Peak KV-Cache Memory** | 192 KB | 80 KB |
| **Final KV-Cache Memory** | 192 KB | 38 KB |
| **Physical RAM Reclaimed** | 0 KB | 154 KB |
| **Tokens Safely Archived** | 0 tokens | 77 tokens |
| **Needle Retrieval Score** | N/A | 0.864226 (Cosine Match) |
| **SIMD Retrieval Latency** | N/A | 126.208 µs |

Generated automatically by `amuquant_benchmark`.
