# AMU-QUANT: Co-Adaptive Layer-wise Weight Streaming and TurboQuant Context Compression for On-Device Large Language Model Inference

---

## 1. Intisari Ilmiah (Abstract)

Menjalankan model bahasa berskala besar (*Large Language Model* / LLM) secara lokal pada perangkat komputer jinjing (*laptop*) standar dengan kapasitas RAM terbatas (misalnya 8 GB–16 GB) menghadapi satu kendala mendasar: **memori cepat habis saat teks atau penalaran yang dihasilkan semakin panjang**. Di dalam arsitektur komputer, alokasi memori fisik (RAM) oleh LLM terbagi menjadi dua komponen utama yang saling berkompetisi memperebutkan ruang memori:

1. **Bobot Model (*Model Weights*):** Ukuran file model yang dimuat ke dalam memori (misalnya dalam format terkuantisasi GGUF) memakan porsi besar RAM fisik sejak inisialisasi awal. Digabungkan dengan pemakaian memori kernel sistem operasi, sisa ruang memori bebas pada perangkat menjadi sangat sempit bahkan sebelum proses generasi teks dimulai.
2. **Memori Konteks (*Key-Value Cache* / KV-Cache):** Saat model menghasilkan kata demi kata (*autoregressive decoding*), sistem harus menyimpan riwayat status representasi tersembunyi di dalam KV-Cache. Semakin panjang teks atau penalaran bertingkat yang dihasilkan (mencapai ribuan token), ukuran KV-Cache membengkak secara linier dan memakan tambahan ruang RAM yang besar.

Ketika ukuran KV-Cache membesar atau ada aplikasi lain (seperti peramban web atau editor kode) yang dibuka bersamaan, total kebutuhan memori melewati batas fisik RAM perangkat. Akibatnya, sistem operasi memindahkan paksa sebagian memori bobot model dari RAM ke penyimpanan SSD (*swapping / page thrashing*). Hal ini membuat kecepatan generasi token turun drastis atau proses dihentikan mendadak oleh sistem operasi (*Out-of-Memory / OOM Killer*). Masalah ini bersifat umum dan terdistribusi di berbagai skala model: dari model kelas kompak (sub-3B), menengah (7B–8B), besar (14B–32B), hingga sangat besar (70B).

Pendekatan terdahulu memiliki empat kelemahan sistemik:
* **Sistem Layer-Streaming Tidak Mengurus Pembengkakan KV-Cache:** Metode seperti AirLLM ([Lyogavin, 2024](https://github.com/lyogavin/airllm)) atau fitur `mmap` pada `llama.cpp` ([Gerganov et al., 2026](https://github.com/ggml-org/llama.cpp)) mampu membaca bobot bertahap per layer dari SSD, namun membiarkan KV-Cache terus membesar di RAM hingga akhirnya tetap kehabisan memori pada obrolan panjang.
* **Pemotongan Konteks Konvensional Menimbulkan Kehilangan Informasi (*Evict-to-Trash*):** Metode seperti StreamingLLM ([Xiao et al., ICLR 2024](https://arxiv.org/abs/2309.17453)), H2O ([Zhang et al., NeurIPS 2023](https://arxiv.org/abs/2306.14048)), ThinKV ([ICLR 2026](https://arxiv.org/abs/2510.01290)), dan Lethe ([AAAI 2026](https://arxiv.org/abs/2511.12612)) menghapus token lama dari KV-Cache secara permanen (*evict-to-trash*) sehingga informasi penting hilang selamanya, serta sepenuhnya buta terhadap sisa ketersediaan RAM sistem operasi.
* **Kompresi TurboQuant di llama.cpp Masih Bersifat Statis:** Penerapan kuantisasi vektor tanpa pelatihan TurboQuant ([Zandieh et al., ICLR 2026](https://arxiv.org/abs/2504.19874)) pada cabang komunitas `llama.cpp` ([TheTom, 2026](https://github.com/TheTom/llama-cpp-turboquant); [AmesianX, 2026](https://github.com/AmesianX/TurboQuant); [Issue #20977](https://github.com/ggml-org/llama.cpp/issues/20977); [PR #21038](https://www.google.com/search?q=https://github.com/ggml-org/llama.cpp/pull/21038)) hanya diatur di awal sebelum model berjalan, belum mampu menyesuaikan diri secara otomatis saat RAM tiba-tiba penuh di tengah generasi.
* **Pustaka turbovec Belum Terhubung ke Mesin C++ (llama.cpp):** Pustaka `turbovec` ([Codrai, 2026](https://github.com/RyanCodrai/turbovec)) dalam bahasa Rust mampu memadatkan vektor hingga $7{,}8\times$ (mode 4-bit) dan $15{,}5\times$ (mode 2-bit) dalam mikrodetik via SIMD CPU, namun baru mendukung Rust dan Python, serta belum pernah dihubungkan ke `llama.cpp` (C++) untuk menampung KV-Cache yang terpotong.

**AMU-QUANT** mengusulkan algoritma inferensi ko-adaptif berbasis C++17 dan Rust yang memantau sisa RAM perangkat saat LLM menghasilkan teks. Jika RAM mulai menipis, AMU-QUANT memotong token lama dari KV-Cache di `llama.cpp` agar ruang RAM langsung lega, lalu memindahkan representasi vektor token tadi ke dalam indeks terkompresi `turbovec` (4-bit atau 2-bit) via jembatan C-ABI FFI (`turbovec-ffi`) secara *zero-copy*. Pada saat yang sama, AMU-QUANT mengatur pelepasan halaman memori layer model yang sudah selesai dihitung via panggilan kernel `posix_madvise(..., POSIX_MADV_DONTNEED / WILLNEED)` agar alokasi bobot model dan KV-Cache tidak saling berebut RAM.

Dari sudut pandang **Sains Data (*Data Science*)** dan **Pemrosesan Bahasa Alami (*Natural Language Processing*)**, penelitian ini berfokus pada:
1. **Optimasi representasi data vektor berdimensi tinggi** melalui kuantisasi skalar Lloyd-Max (4-bit dan 2-bit).
2. **Pemodelan prediktif deret waktu (*time-series forecasting*)** pada data telemetri memori sistem untuk mendeteksi akselerasi konsumsi memori sebelum terjadi *swapping*.
3. **Evaluasi statistik terhadap kualitas distribusi probabilitas token (*Perplexity* / PPL)** dan akurasi temu kembali informasi (*Information Retrieval* / Recall@K).

---

## 2. Landasan Teoretis dan Model Matematika

### 2.1 Persamaan Ekuilibrium Memori Ganda (Dual-Budget Constraint)
Pada perangkat dengan batas aman alokasi RAM fisik $M_{\text{limit}}$, total alokasi memori fisik (*Resident Set Size* / RSS) pada langkah pembuatan token ke-$t$ dimodelkan sebagai pertidaksamaan ekuilibrium:

$$M_{\text{ext}}(t) + \sum_{l \in \mathcal{L}_{\text{resident}}(t)} M_{\text{weight}}(l) + M_{\text{KV}}(W_t) + M_{\text{turbovec}}(C_t, b_t) \le M_{\text{limit}}$$

Keterangan variabel:
* $M_{\text{ext}}(t)$: Alokasi memori proses eksternal di luar engine inferensi pada waktu $t$.
* $\mathcal{L}_{\text{resident}}(t) \subseteq \{0, 1, \dots, L-1\}$: Himpunan indeks layer model yang ditahan halamannya di RAM via `mmap`.
* $M_{\text{weight}}(l)$: Beban memori bobot layer ke-$l$ ($\approx 305\text{ MB}$ per layer untuk model 32B Q4_K_M).
* $W_t$: Jumlah token aktif pada KV-cache fisik `llama.cpp`.
* $C_t$: Jumlah potongan konteks (*context chunks*) yang diarsipkan ke indeks `turbovec`.
* $b_t \in \{4, 2\}$: Presisi kuantisasi vektor TurboQuant pada langkah $t$.

### 2.2 Skor Tekanan Memori Prediktif ($\Phi_t$)
Untuk mengantisipasi lonjakan beban memori sebelum OOM Killer terpicu, kontroler menghitung skor tekanan prediktif setiap interval $K$ token:

$$\Phi_t = \alpha \left( \frac{M_{\text{total}}(t)}{M_{\text{limit}}} \right) + \beta \left( \frac{\widehat{\Delta M}_t}{\Delta t} \cdot \frac{\tau_{\text{horizon}}}{M_{\text{limit}}} \right)$$

Nilai nominal parameter: $\alpha = 0{,}65$, $\beta = 0{,}35$, dan $\tau_{\text{horizon}} = 2{,}0\text{ detik}$.

### 2.3 Kebijakan Tiga Rejim Adaptasi
* **Regime 1: High-Throughput Caching ($\Phi_t \le \theta_{\text{low}}$)**  
  Memaksimalkan jendela *prefetching* bobot layer di RAM ($|\mathcal{L}_{\text{resident}}| \gg 1$). KV-Cache penuh dipertahankan di `llama.cpp` tanpa pemangkasan.
* **Regime 2: Balanced Offloading ($\theta_{\text{low}} < \Phi_t \le \theta_{\text{high}}$)**  
  Token tengah dipangkas dari KV-Cache `llama.cpp` menggunakan `llama_kv_cache_seq_rm()` dan diarsipkan ke `turbovec` mode 4-bit (menghemat RAM $\approx 7{,}8\times$). Sisa ruang RAM dialihkan sebagai buffer *prefetching* 1 layer berikutnya dari SSD. Layer yang selesai dieksekusi dilepas halamannya via `posix_madvise(..., POSIX_MADV_DONTNEED)`.
* **Regime 3: Ultra-Low Memory Survival Mode ($\Phi_t > \theta_{\text{high}}$)**  
  Menjalankan kuantisasi ulang *in-place* pada indeks `turbovec` dari 4-bit ke 2-bit (menghemat RAM $\approx 15{,}5\times$). Menerapkan *Strict Layer-by-Layer Streaming* (hanya 1 layer aktif di RAM). Menjalankan pemulihan konteks selektif (*selective context recall*) berbasis pencarian kemiripan kosinus SIMD (<0,25 ms).

---

## 3. Arsitektur Sistem

```mermaid
flowchart TD
    subgraph OS_Layer ["Lapisan Sistem Operasi (Kernel dan Virtual Memory)"]
        OS_Mem["Kernel Virtual Memory (/proc/meminfo / mach_vm_info)"]
        Sys_Madvise["Kernel Paging: posix_madvise(DONTNEED / WILLNEED)"]
        GGUF_File["File Model Tunggal .gguf pada NVMe SSD"]
    end

    subgraph AMUQuant_Controller ["AMU-QUANT Co-Adaptive Controller (C++17)"]
        TelemetrySampler["Memory Sampler (Setiap K Token)"]
        PredictiveScore["Kalkulasi Skor Prediktif: Φ_t"]
        RegimeMachine{"Evaluasi Rejim (Histeresis δ_hyst)"}
        R1["Regime 1: High-Throughput (Max Prefetch)"]
        R2["Regime 2: Balanced Offload (turbovec 4-bit)"]
        R3["Regime 3: Ultra-Low Survival (turbovec 2-bit + Strict Streaming)"]
    end

    subgraph Llama_Engine ["Engine Inferensi: llama.cpp"]
        WeightRegistry["Mmap Layer Address Space Registry"]
        KVCache["KV-Cache Buffer Aktif"]
        KVSurgery["KV Surgery: seq_rm & seq_add (RoPE Alignment)"]
    end

    subgraph Turbovec_FFI ["Episodic Vector Memory: turbovec (Rust C-ABI)"]
        FFI_Layer["turbovec-ffi (Zero-Copy Raw Pointer Bridge)"]
        TurboQuantIndex["TurboQuant Index (Random Orthogonal + Lloyd-Max)"]
        SIMDSearch["AVX2 / AVX-512 / NEON Cosine Search (<0.25 ms)"]
        Requantizer["In-Place 4-bit to 2-bit Requantizer"]
    end

    OS_Mem --> TelemetrySampler
    TelemetrySampler --> PredictiveScore
    PredictiveScore --> RegimeMachine
    RegimeMachine -->|Φ_t ≤ θ_low| R1
    RegimeMachine -->|θ_low < Φ_t ≤ θ_high| R2
    RegimeMachine -->|Φ_t > θ_high| R3

    R1 -.->|Prefetch Layer Mendatang| Sys_Madvise
    R2 -->|Evict Token Tengah| KVSurgery
    R3 -->|Trigger Kompresi 2-bit & Strict Streaming| KVSurgery

    GGUF_File -->|mmap read-only| WeightRegistry
    Sys_Madvise -->|Evict / Prefetch Halaman| WeightRegistry
    KVSurgery -->|Pointer Float Mentah (Zero-Copy)| FFI_Layer
    FFI_Layer --> TurboQuantIndex
    TurboQuantIndex --> Requantizer
    SIMDSearch -->|Selective Context Recall| KVCache
```

---

## 4. Matriks Posisi Kebaruan (Novelty Matrix)

Perbandingan komprehensif posisi AMU-QUANT terhadap penelitian dan implementasi mutakhir (*state-of-the-art*):

| Metode / Sistem | Referensi Utama | Bahasa | Layer Streaming (Model > RAM) | Kompresi Konteks TurboQuant | Kesadaran Tekanan Memori OS | Mekanisme Evict-to-Index & SIMD Recall | Beban Runtime Eksternal |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **AirLLM** | [Lyogavin (2024)](https://github.com/lyogavin/airllm) | Python | Ya (File Splitting) | Tidak | Tidak | Tidak | Python Runtime (Lambat) |
| **StreamingLLM** | [ICLR 2024](https://arxiv.org/abs/2309.17453) | Python | Tidak | Tidak | Tidak | Tidak (Permanen Dihapus) | Python Runtime |
| **H2O** | [NeurIPS 2023](https://arxiv.org/abs/2306.14048) | Python | Tidak | Tidak | Tidak | Tidak (Permanen Dihapus) | Python Runtime |
| **ThinKV** | [ICLR 2026](https://arxiv.org/abs/2510.01290) | Python | Tidak | Tidak | Tidak | Tidak (Permanen Dihapus) | Python Runtime |
| **Lethe** | [AAAI 2026](https://arxiv.org/abs/2511.12612) | Python | Tidak | Tidak | Tidak | Tidak (Permanen Dihapus) | Python Runtime |
| **llama.cpp (mmap)**| [Gerganov et al. (2026)](https://github.com/ggml-org/llama.cpp) | C++ | Pasif (Demand Paging) | Tidak | Tidak | Tidak | Nihil (Native C++) |
| **TurboQuant Fork** | [TheTom / AmesianX (2026)](https://github.com/TheTom/llama-cpp-turboquant) | C++ | Tidak | Ya (Statis Inisial) | Tidak | Tidak | Nihil (Native C++) |
| **turbovec** | [Codrai (2026)](https://github.com/RyanCodrai/turbovec) | Rust | Tidak | Ya (4-bit & 2-bit) | Tidak | Ya (Khusus Vektor DB RAG) | Nihil (Native Rust) |
| **AMU-QUANT** *(Usulan)* | **Tugas Akhir (2026)** | **C++ & Rust FFI** | **Ya (Aktif madvise / Prefetch)** | **Ya (Dinamis Mid-Gen)** | **Ya (Dual-Budget Controller)** | **Ya (turbovec SIMD Recall <0.25 ms)** | **Nihil (Native Unified Binary)** |

---

## 5. Struktur Dokumentasi Proyek

Dokumentasi akademik dan spesifikasi teknis lengkap tersedia di direktori [`docs/`](docs/):

* [**docs/01_PROPOSAL_TUGAS_AKHIR.md**](docs/01_PROPOSAL_TUGAS_AKHIR.md): Naskah Usulan Tugas Akhir Lengkap (Bab I Pendahuluan, Bab II Tinjauan Pustaka, Bab III Metodologi, Bab IV Jadwal dan Target Luaran).
* [**docs/02_ARSITEKTUR_SISTEM.md**](docs/02_ARSITEKTUR_SISTEM.md): Arsitektur Teknis Subsistem, Diagram Sekuensial Autoregressive Decoding Loop, dan Model Kepemilikan Memori *Zero-Copy*.
* [**docs/03_MODEL_MATEMATIKA_DAN_KONTROLER.md**](docs/03_MODEL_MATEMATIKA_DAN_KONTROLER.md): Model Matematis *Dual-Budget Equilibrium*, Formula Skor Tekanan Prediktif $\Phi_t$, *Hysteresis State Machine*, dan Pseudocode Algoritma.
* [**docs/04_DESAIN_TURBOVEC_FFI.md**](docs/04_DESAIN_TURBOVEC_FFI.md): Spesifikasi Antarmuka C-ABI `turbovec_ffi.h`, Implementasi Rust `ffi.rs`, Rekuantisasi *In-Place* 4-bit $\to$ 2-bit, dan Optimasi SIMD CPU.
* [**docs/05_METODOLOGI_DAN_RENCANA_EKSPERIMEN.md**](docs/05_METODOLOGI_DAN_RENCANA_EKSPERIMEN.md): Setup Lingkungan Uji (Node 8GB & 16GB RAM), Model Uji (8B, 14B, 32B), 4 Metode Baseline, 3 Skenario Eksperimen, dan Metrik Kuantitatif.
* [**docs/06_TINJAUAN_PUSTAKA_DAN_REFERENSI.md**](docs/06_TINJAUAN_PUSTAKA_DAN_REFERENSI.md): Kajian Pustaka Kritis, Komparasi State-of-the-Art, dan Sitasi Ilmiah Lengkap.

---

## 6. Spesifikasi Antarmuka C-ABI (`turbovec_ffi.h`)

Jembatan integrasi native antara C++ (`llama.cpp`) dan Rust (`turbovec`) didefinisikan dalam [**`docs/04_DESAIN_TURBOVEC_FFI.md`**](docs/04_DESAIN_TURBOVEC_FFI.md) dengan ringkasan tanda tangan fungsi:

```c
// Alokasi indeks vektor TurboQuant di heap Rust (mode 4-bit atau 2-bit)
TurbovecStatus turbovec_init(uint32_t dimension, uint8_t bit_width, TurbovecIndexHandle** out_handle);

// Penyisipan pointer float langsung tanpa penyalinan perantara (Zero-Copy)
TurbovecStatus turbovec_insert_zero_copy(TurbovecIndexHandle* handle, uint64_t chunk_id, const float* float_ptr, size_t len);

// Pencarian k-tetangga terdekat kemiripan kosinus SIMD (< 0.25 ms)
TurbovecStatus turbovec_search_simd(const TurbovecIndexHandle* handle, const float* query_ptr, uint32_t top_k, uint64_t* out_ids, float* out_scores, uint32_t* out_count);

// Rekuantisasi in-place dari presisi 4-bit ke 2-bit (Regime 3 Survival Mode)
TurbovecStatus turbovec_requantize_to_2bit(TurbovecIndexHandle* handle);

// Deallokasi instance indeks dan pembersihan heap memory
TurbovecStatus turbovec_free(TurbovecIndexHandle* handle);
```

---

## 7. Rencana Eksperimen dan Metrik Evaluasi

### 7.1 Matriks Skenario Uji
1. **Skenario 1 — Skalabilitas Ukuran Model dan Konteks (Clean Run):**  
   Pengujian spektrum model lintas-skala (model kompak sub-3B, menengah 7B/8B, besar 14B/32B, hingga 70B dalam kuantisasi Q4_K_M) pada variasi panjang konteks 2K, 4K, 8K, dan 16K token di lingkungan bersih untuk mengevaluasi titik saturasi OOM.
2. **Skenario 2 — Injeksi Tekanan Memori OS Dinamis (Stress Test):**  
   Pemberian beban alokasi memori eksternal sintetik (2 GB–6 GB) secara mendadak saat fase autoregressive decoding berlangsung untuk menguji respons adaptif rejim AMU-QUANT dan pencegahan OOM.
3. **Skenario 3 — Ketahanan Konteks (Needle-in-a-Haystack dan PPL):**  
   Pengujian ekstraksi fakta tersembunyi pada variasi kedalaman dokumen 0%, 25%, 50%, 75%, 100% serta evaluasi Perplexity (PPL) pada teks benchmark WikiText-2.

### 7.2 Metrik Kuantitatif
* **Peak Resident Set Size (Peak RSS):** Pengukuran memori fisik puncak via `getrusage(ru_maxrss)` dan `/proc/[pid]/status` (`VmHWM`).
* **Survival Rate (%):** Rasio penyelesaian eksekusi hingga tuntas tanpa terminasi OOM Killer (`SIGKILL`).
* **Inter-Token Latency (ITL) dan Time-to-First-Token (TTFT):** Latensi generasi per token dan komputasi prompt awal dalam milidetik.
* **Major Page Faults Rate:** Frekuensi pemanggilan blok data dari SSD akibat halaman terdepak dari RAM (`ru_majflt`).
* **Retrieval Accuracy (Recall@K):** Akurasi pemulihan informasi dari blok konteks yang dipangkas via SIMD search.

---

## 8. Daftar Referensi Ilmiah

1. **Zandieh, A., Daliri, M., Hadian, M., & Mirrokni, V. (2026).** *TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate*. In *Proceedings of the International Conference on Learning Representations (ICLR 2026)*.  
   Paper arXiv: [https://arxiv.org/abs/2504.19874](https://arxiv.org/abs/2504.19874)  
   Google Research Blog: [https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/)

2. **Codrai, R. (2026).** *turbovec: High-performance, in-process vector search index in Rust implementing Google Research's TurboQuant (ICLR 2026)*.  
   Repositori GitHub: [https://github.com/RyanCodrai/turbovec](https://github.com/RyanCodrai/turbovec)

3. **Gerganov, G., et al. (2026).** *llama.cpp: Inference of LLaMA model in pure C/C++*.  
   Repositori Utama: [https://github.com/ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)  
   Diskusi & PR TurboQuant: [Issue #20977](https://github.com/ggml-org/llama.cpp/issues/20977) & [Pull Request #21038](https://www.google.com/search?q=https://github.com/ggml-org/llama.cpp/pull/21038)

4. **Lyogavin. (2023/2024).** *AirLLM: 70B inference with single 4GB GPU (Layer-wise LLM Inference Optimization)*.  
   Repositori GitHub: [https://github.com/lyogavin/airllm](https://github.com/lyogavin/airllm)

5. **Xiao, G., Tian, Y., Chen, B., Han, S., & Lewis, M. (2024).** *Efficient Streaming Language Models with Attention Sinks (StreamingLLM)*. In *Proceedings of the International Conference on Learning Representations (ICLR 2024)*.  
   Paper arXiv: [https://arxiv.org/abs/2309.17453](https://arxiv.org/abs/2309.17453)

6. **Zhang, Z., Sheng, Y., Zhou, T., Chen, T., Zheng, L., Cai, R., Song, Z., Tian, Y., Ré, C., Barrett, C., Wang, Z., & Chen, B. (2023).** *H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models*. In *Advances in Neural Information Processing Systems (NeurIPS 2023)*, Vol. 36.  
   Paper arXiv: [https://arxiv.org/abs/2306.14048](https://arxiv.org/abs/2306.14048)

7. **ThinKV Authors. (2026).** *ThinKV: Thought-Adaptive KV Cache Compression for Reasoning Models*. In *Proceedings of the International Conference on Learning Representations (ICLR 2026)*.  
   Paper arXiv: [https://arxiv.org/abs/2510.01290](https://arxiv.org/abs/2510.01290)

8. **Lethe Authors. (2026).** *Lethe: Layer-Adaptive KV Cache Eviction with Dynamic Sparsity Allocation*. In *Proceedings of the AAAI Conference on Artificial Intelligence (AAAI 2026)*.  
   Paper arXiv: [https://arxiv.org/abs/2511.12612](https://arxiv.org/abs/2511.12612)

9. **Liu, Z., Desai, A., Liao, F., Shen, W., Xie, Z., Xu, K., Pavone, M., & Ermon, S. (2024).** *Scissorhands: Exploiting the Persistence of Importance Hypothesis for LLM KV Cache Compression*. In *Advances in Neural Information Processing Systems (NeurIPS 2024)*.  
   Paper arXiv: [https://arxiv.org/abs/2305.17118](https://arxiv.org/abs/2305.17118)

10. **TheTom. (2026).** *llama-cpp-turboquant: Experimental TurboQuant integration in llama.cpp*.  
    Repositori GitHub: [https://github.com/TheTom/llama-cpp-turboquant](https://github.com/TheTom/llama-cpp-turboquant)

11. **AmesianX. (2026).** *TurboQuant: C++ Implementation and Exploration for Quantized Inference*.  
    Repositori GitHub: [https://github.com/AmesianX/TurboQuant](https://github.com/AmesianX/TurboQuant)

---

## 9. Penulis dan Afiliasi Akademik

* **Nama Pengusul:** Mahasiswa Tugas Akhir  
* **Program Studi:** Sains Data / Ilmu Komputer / Teknik Informatika  
* **Fokus Penelitian:** *Data Science, Natural Language Processing, & High-Performance Systems for Deep Learning*  
* **Tahun Akademik:** 2026/2027
