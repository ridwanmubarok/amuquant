# Pengembangan Algoritma Adaptif Berbasis Data Kondisi Memori untuk Optimasi Inferensi Large Language Model pada Perangkat dengan Sumber Daya Terbatas

**Tugas Akhir / Skripsi S1 — Peminatan Data Science**  
*Host-Aware Adaptive Algorithm Based on Memory Condition Data for Large Language Model Inference Optimization on Resource-Constrained Devices*

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![C++17](https://img.shields.io/badge/C%2B%2B-17-00599C.svg?logo=c%2B%2B)](CMakeLists.txt)
[![CMake](https://img.shields.io/badge/CMake-3.20+-green.svg?logo=cmake)](CMakeLists.txt)
[![Data Science](https://img.shields.io/badge/Data_Science-Telemetry_Time__Series-purple.svg)](scripts/analyze_telemetry.py)

---

## 1. Latar Belakang & Posisi Akademis Tugas Akhir

### 1.1 Judul Penelitian
**"Pengembangan Algoritma Adaptif Berbasis Data Kondisi Memori untuk Optimasi Inferensi Large Language Model pada Perangkat dengan Sumber Daya Terbatas"**

### 1.2 Rumusan Masalah
Penerapan Large Language Model (LLM) autoregresif pada perangkat lokal (*edge computing* / PC dengan kapasitas RAM terbatas seperti 8 GB Unified Memory) menghadapi dua kendala utama:
1. **Pertumbuhan Monotonik KV Cache ($\mathcal{O}(N)$)**:
   Pada mesin inferensi standar seperti `llama.cpp`, ukuran Key-Value (KV) cache bertambah linier seiring bertambahnya panjang konteks percakapan multi-turn. Alokasi ini bersaing langsung dengan aplikasi host lain (browser, sistem operasi, compiler).
2. **Kebutaan Runtime terhadap Tekanan Memori Host (*Host-Memory Blindness*)**:
   Mesin inferensi standar tidak membaca kondisi memori fisik host secara real-time. Ketika RAM fisik saturasi, kernel sistem operasi akan mengalami *swap thrashing* parah atau memanggil *Out-Of-Memory (OOM) Killer* (`SIGKILL 137`), yang menghentikan proses inferensi secara tiba-tiba.

### 1.3 Kontribusi & Pendekatan Data Science
AMU-QUANT mengintegrasikan metodologi **Data Science** langsung ke dalam *runtime inference loop* menggunakan C++17 native:
- **Ekstraksi Fitur Telemetri Time-Series Host**: Mengambil metrik fisik kernel secara non-blocking setiap interval token ($\Delta t \approx 50\text{ ms}$): *Process RSS*, *Physical Footprint* Darwin/Linux, *Available RAM*, dan laju perubahan swap ($\frac{dM}{dt}$).
- **Skor Tekanan Statistik Prediktif ($\Phi_t$)**: Menghitung skor komposit prediktif yang memproyeksikan risiko kejenuhan memori pada horizon masa depan ($\tau = 2.0\text{ s}$), mendahului terjadinya OOM fisik.
- **Arbitrase Multi-Objektif Terkalibrasi ($J(a)$)**: Mengambil keputusan adaptasi (*Regime 1: Nominal*, *Regime 2: Balanced*, *Regime 3: Survival*) berdasarkan minimasi fungsi biaya yang menyeimbangkan risiko OOM, penalti latensi, dan retensi kualitas konteks.
- **Restriksi Working Set Logis $\mathcal{O}(1)$**: Membatasi ukuran aktif KV Cache menggunakan *Attention Sink* dan *Sliding Window* dengan translasi posisi RoPE secara matematis, mencegah lonjakan kebutuhan alokasi heap baru.

---

## 2. Perbandingan Karakteristik Runtime

| Dimensi Evaluasi | **llama.cpp** (Baseline Tanpa Manajemen) | **Disk-Streaming Statis** (AirLLM Paradigm) | **AMU-QUANT** (Algoritma Adaptif) |
| :--- | :--- | :--- | :--- |
| **Arsitektur Eksekusi** | Monolitik `mmap` dengan paging pasif kernel OS | Pembacaan layer sinkron dari disk per forward pass | Host-Aware Co-Adaptive Controller (C++17) |
| **Kesadaran Memori Host**| Tidak ada (statis tanpa membaca RAM tersedia) | Tidak ada (selalu streaming dari disk) | Telemetri Time-Series Tertutup: RSS, Footprint, Swap, $\frac{dM}{dt}$ |
| **Metrik Tekanan Memori**| Tidak ada | Tidak ada | Formulasi statistik prediktif $\Phi_t \in [0, 1+]$ |
| **Manajemen Bobot Mmap** | Pasif menyerahkan seluruhnya ke kernel OS | Swap disk eksplisit per-layer | *Advisory Kernel Paging*: `posix_madvise` & `msync(MS_INVALIDATE)` |
| **Verifikasi Residensi** | Tidak ada | Alokasi layer manual | Audit fisik tabel halaman kernel via `mincore(2)` |
| **Kompleksitas KV Cache** | Linier $\mathcal{O}(N)$ terhadap panjang sekuens | Penuh tanpa pemangkasan | Bounded $\mathcal{O}(1)$ working set ($S + W_{\text{recent}}$) |
| **Mitigasi Risiko OOM**   | Rentan terhadap `SIGKILL (137)` saat RAM host penuh | Tahan OOM namun latensi sangat tinggi (< 0.5 tok/s) | Proaktif membatasi working set & menurunkan beban memori |
| **Logging Data Science**  | Log teks standar | Log teks standar | Ekspor dataset time-series 16 variabel ke CSV |

> [!NOTE]
> **Catatan Ilmiah Mengenai Alokasi Memori Fisik vs Logis**:
> Pemangkasan slot KV Cache (`llama_memory_seq_rm`) membebaskan slot pada *cell table* internal `llama.cpp` sehingga token baru dapat memakai kembali slot tersebut tanpa memerlukan alokasi heap/VRAM baru. Hal ini membatasi *working set* secara ketat pada kompleksitas $\mathcal{O}(1)$ dan mencegah ekspansi memori tak berhingga. Namun, alokator memori `llama.cpp` mengelola memori tersebut dalam bentuk pool memori yang tidak serta-merta mengembalikan halaman heap fisik ke OS seketika itu juga.

---

## 3. Arsitektur Sistem

```
+-----------------------------------------------------------------------------------+
|                           HOST OPERATING SYSTEM KERNEL                            |
|             (macOS Darwin / Linux Virtual Memory Subsystem & Page Tables)         |
+-----------------------------------------------------------------------------------+
       | (phys_footprint)               | (available_ram)             | (vm.swapusage)
       v                                v                             v
+-----------------------------------------------------------------------------------+
|                        AMU-QUANT SYSTEM TELEMETRY SAMPLER                         |
|                 (Non-blocking sampling of kernel physical state)                  |
+-----------------------------------------------------------------------------------+
                                        |
                                        v
+-----------------------------------------------------------------------------------+
|                     PREDICTIVE CO-ADAPTIVE CONTROLLER (FSM)                       |
|           Evaluates Pressure Metric Phi_t and Minimizes Utility J(a)              |
+-----------------------------------------------------------------------------------+
         |                                |                               |
         | Regime 1                       | Regime 2                      | Regime 3
         v (Nominal)                      v (Balanced Offload)            v (Survival Anti-Crash)
+------------------------+  +-------------------------------+  +-------------------------------+
| Prefetch Depth: 3      |  | Prefetch Depth: 1             |  | Prefetch Depth: 0 (Single)   |
| Zero KV Eviction       |  | KV Sliding Window (Sink + Win)|  | POSIX_MADV_DONTNEED Inactive  |
| Full Generation Speed  |  | RoPE Realignment (seq_add)    |  | Min KV Window (O(1) Bounds)   |
+------------------------+  +-------------------------------+  +-------------------------------+
                                        |
                                        v
+-----------------------------------------------------------------------------------+
|                  DYNAMIC KV SURGEON & ATTENTION SINK SHIELD                       |
|      Preserves initial prompt tokens [0, S) and recent context [N - W, N)         |
|      Prunes middle slots [S, N - W) via seq_rm and shifts RoPE positions         |
+-----------------------------------------------------------------------------------+
                                        |
                                        v
+-----------------------------------------------------------------------------------+
|                    DATA SCIENCE TELEMETRY DATASET EXPORTER                        |
|       Logs 16 metrics per-token to CSV -> Analyzed via scripts/analyze_telemetry.py|
+-----------------------------------------------------------------------------------+
```

---

## 4. Landasan Teori & Formulasi Matematis

### 4.1 Skor Tekanan Memori Prediktif ($\Phi_t$)
Penilaian kondisi memori host dilakukan melalui perpaduan rasio saturasi statis dan turunan laju pertumbuhan memori (*time-series slope*):

$$\Phi_t = \max(U_{\text{proc}}(t), U_{\text{host}}(t)) + \beta \cdot P_{\text{growth}}(t) + \gamma \cdot P_{\text{swap}}(t)$$

Di mana:
- **Saturasi Proses**: $U_{\text{proc}}(t) = \frac{M_{\text{process\_rss}}(t)}{M_{\text{limit}} - M_{\text{os\_reserved}}}$
- **Depresi Memori Host**: $U_{\text{host}}(t) = 1.0 + \frac{M_{\text{os\_reserved}} - M_{\text{available}}}{M_{\text{os\_reserved}}}$ saat RAM tersedia berada di bawah batas cadangan aman.
- **Laju Pertumbuhan Dinamis**: $P_{\text{growth}}(t) = \frac{1}{M_{\text{safe}}} \left(\frac{dM_{\text{rss}}}{dt} \cdot \tau\right)$, dengan lookahead horizon $\tau = 2.0\text{ s}$.
- **Penalti Swap Host**: $P_{\text{swap}}(t)$ mengukur akselerasi swap yang memicu thrashing kernel.

### 4.2 Batas Matematika Working Set KV Cache ($\mathcal{O}(1)$)
Ukuran memori logis satu slot token KV Cache dihitung dari arsitektur model:

$$\text{Bytes Per Token} = 2 \times N_{\text{layers}} \times N_{\text{kv\_heads}} \times D_{\text{head}} \times B_{\text{element}}$$

Untuk arsitektur Qwen 2.5 3B ($N_{\text{layers}} = 36$, $N_{\text{kv\_heads}} = 2$, $D_{\text{head}} = 128$, $B_{\text{element}} = 2\text{ bytes}$ FP16):
$$\text{Bytes Per Token} = 2 \times 36 \times 2 \times 128 \times 2 = 36,864\text{ bytes (36 KB / token)}$$

AMU-QUANT membatasi slot aktif $W_{\text{active}}$ berdasarkan rejim yang dipilih:

$$W_{\text{active}}(R_t) = S + W_{\text{recent}}(R_t)$$

- **Attention Sink ($S$)**: $S = 64$ token permanen di awal konteks.
- **Regime 2 (Balanced)**: $W_{\text{recent}} = 512$ token $\implies W_{\text{active}} = 64 + 512 = 576$ token ($\approx 20.7\text{ MB}$).
- **Regime 3 (Survival)**: $W_{\text{recent}} = W_{\text{min}} = 128$ token $\implies W_{\text{active}} = 64 + 128 = 192$ token ($\approx 6.9\text{ MB}$).

Untuk panjang sekuens tak berhingga $N \to \infty$, kapasitas aktif KV Cache tetap terikat secara asimtotik:

$$\lim_{N \to \infty} M_{\text{KV\_active}}(N) \le W_{\text{active}}(R_t) \times \text{BytesPerToken} \in \mathcal{O}(1)$$

---

## 5. Struktur Repositori

```
aegis/
├── CMakeLists.txt                 # Konfigurasi build native C++17
├── README.md                      # Dokumentasi akademik Tugas Akhir
├── include/
│   └── amuquant/
│       ├── types.hpp              # Struktur metrik telemetri, enum Regime, dan StepDecision
│       ├── telemetry.hpp          # Sampler memori kernel real-time (macOS Darwin & Linux)
│       ├── layer_manager.hpp      # Manajemen penasihat halaman kernel & mincore(2) audit
│       ├── kv_surgeon.hpp         # Pemangkasan KV Cache dinamis & translasi RoPE
│       ├── controller.hpp         # Pengontrol FSM prediktif dan optimasi fungsi biaya J(a)
│       ├── engine.hpp             # Koordinator runtime AMUQuantEngine terpadu
│       ├── canonical_history.hpp  # Penyimpanan riwayat faktual & intent-aware abstention
│       └── gguf_parser.hpp        # Parser biner langsung untuk metadata model GGUF
├── src/
│   ├── telemetry.cpp              # Implementasi telemetri kernel mach/sysinfo
│   ├── layer_manager.cpp          # Implementasi posix_madvise, msync, dan mincore
│   ├── kv_surgeon.cpp             # Eksekusi seq_rm, seq_add, dan rollback transaksi
│   ├── controller.cpp             # Implementasi evaluasi Phi_t dan arbitrase utilitas
│   ├── amuquant_engine.cpp        # Integrasi koordinasi engine pusat
│   ├── canonical_history.cpp      # Klasifikasi intent query dan verifikasi grounding
│   └── gguf_parser.cpp            # Ekstraksi tensor dan arsitektur file GGUF
├── examples/
│   ├── amuquant_chat.cpp          # Terminal interaktif chat dengan HUD hardware & ekspor CSV
│   └── simple_inference.cpp       # Contoh integrasi minimal API
├── benchmarks/
│   ├── standalone_benchmark.cpp   # Benchmark multi-horizon reproducible dengan ekspor CSV & kalibrasi
│   ├── live_benchmark.cpp         # Benchmark empiris integrasi libllama
│   └── data/                      # Direktori output dataset eksperimen CSV & grafik analisis
├── scripts/
│   └── analyze_telemetry.py       # Skrip Python pipeline analisis statistik Data Science
└── tests/
    ├── test_controller.cpp        # Unit test transisi FSM dan histeresis
    ├── test_amuquant_pipeline.cpp # Unit test verifikasi residensi mincore(2) & alokasi
    └── test_reliable_memory.cpp   # Unit test intent-aware abstention & rolling window
```

---

## 6. Petunjuk Kompilasi & Pengujian

### 6.1 Kompilasi Proyek
```bash
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

### 6.2 Menjalankan Unit Test Terotomatisasi
```bash
ctest --test-dir build --output-on-failure
```

### 6.3 Menjalankan Benchmark Reproducible & Ekspor CSV
```bash
# Menjalankan evaluasi 4 horizon (64, 256, 512, 1024 token) dan ekspor dataset CSV:
./build/amuquant_benchmark --export-csv benchmarks/data/standalone_benchmark_results.csv
```

### 6.4 Melakukan Kalibrasi Parameter Controller (Sensitivitas $\alpha$ vs $\beta$)
```bash
# Melakukan grid-search sensitivitas bobot latensi vs kualitas:
./build/amuquant_benchmark --calibrate
```

### 6.5 Menganalisis Dataset Menggunakan Pipeline Data Science
```bash
# Menghitung statistik deskriptif dan menghasilkan grafik time-series:
python3 scripts/analyze_telemetry.py --input benchmarks/data/standalone_benchmark_results.csv --plot benchmarks/data/telemetry_analysis.png
```

### 6.6 Menjalankan Chat Interaktif dengan Logging Dataset
```bash
./build/amuquant_chat models/qwen2.5-3b-instruct-q4_k_m.gguf --export-csv benchmarks/data/chat_telemetry.csv
```

---

## 7. Lisensi

Penelitian dan basis kode ini didistribusikan di bawah lisensi Apache License 2.0. Lihat file [LICENSE](LICENSE) untuk informasi lebih lanjut.
