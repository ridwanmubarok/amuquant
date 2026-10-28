# AMU-QUANT Standalone Benchmark Report (Zero External Dependencies)

- **Platform**: POSIX / macOS Apple Silicon
- **Embedding Dimension**: 1024 dim
- **Transformer Layers**: 32 layers (32 MB physical mmap allocation)

| Metric | Unmanaged Baseline | AMU-QUANT Co-Adaptive Engine |
| :--- | :--- | :--- |
| **Tokens Processed** | 128 tokens | 128 tokens |
| **Peak KV-Cache Memory** | 256 KB | 80 KB |
| **Final KV-Cache Memory** | 256 KB | 38 KB |
| **Physical RAM Reclaimed** | 0 KB | 218 KB |
| **Tokens Pruned via Window** | 0 tokens | 109 tokens |
| **KV Complexity Bound** | O(N) Linear Growth | O(1) Bounded Working Set |

Generated automatically by `amuquant_benchmark`.
