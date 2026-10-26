# MODEL MATEMATIKA DAN KONTROLER KO-ADAPTIF

Dokumen ini menjelaskan formalisasi matematis dari sistem alokasi memori ganda (*dual-budget equilibrium*) dan desain kontroler ko-adaptif **AMU-QUANT**.

---

## 1. Formulasi Batas Memori Sistem Ganda (Dual-Budget Equilibrium)

Pada perangkat komputasi personal dengan batas aman alokasi RAM fisik $M_{\text{limit}}$, total alokasi memori fisik (*Resident Set Size* / RSS) pada langkah pembuatan token ke-$t$ terdiri dari empat komponen:

$$\underbrace{M_{\text{ext}}(t)}_{\text{Alokasi OS & Aplikasi Luar}} + \underbrace{\sum_{l \in \mathcal{L}_{\text{resident}}(t)} M_{\text{weight}}(l)}_{\text{Kuota Buffer Bobot Layer di RAM}} + \underbrace{M_{\text{KV}}(W_t)}_{\text{KV-Cache Aktif di llama.cpp}} + \underbrace{M_{\text{turbovec}}(C_t, b_t)}_{\text{Indeks Terkompresi turbovec}} \le M_{\text{limit}}$$

### Penjelasan Variabel:
* $M_{\text{limit}}$: Ambang batas aman memori fisik yang dialokasikan untuk sistem, biasanya diset sebesar:
  $$M_{\text{limit}} = M_{\text{total\_RAM}} - M_{\text{OS\_reserved}}$$
  (Misalnya pada laptop RAM 16 GB, $M_{\text{limit}} \approx 13{,}5\text{ GB}$ dengan asumsi $2{,}5\text{ GB}$ dicadangkan untuk kernel dan UI dasar).
* $M_{\text{ext}}(t)$: Alokasi memori oleh proses latar belakang di luar mesin inferensi pada waktu $t$ (misal: browser, IDE, aplikasi sistem).
* $\mathcal{L}_{\text{resident}}(t) \subseteq \{0, 1, \dots, L-1\}$: Himpunan indeks layer model yang halaman memorinya saat ini ditahan (*pinned/cached*) di dalam RAM fisik melalui `mmap`.
* $M_{\text{weight}}(l)$: Ukuran memori bobot untuk satu layer ke-$l$:
  $$M_{\text{weight}}(l) \approx \frac{\text{Total Ukuran File GGUF}}{L}$$
  Untuk model 32B Q4_K_M ($19{,}5\text{ GB}$) dengan $L=64$ layer, setiap layer berukuran $\approx 305\text{ MB}$.
* $W_t$: Jumlah token aktif yang tersimpan dalam KV-cache fisik `llama.cpp` pada langkah $t$. Ukuran memorinya:
  $$M_{\text{KV}}(W_t) = 2 \times L \times n_{\text{kv\_heads}} \times d_{\text{head}} \times \text{sizeof(fp16)} \times W_t$$
* $C_t$: Jumlah potongan konteks (*context chunks*) yang telah dipangkas dari KV-cache fisik dan diarsipkan ke dalam memori `turbovec`.
* $b_t \in \{4, 2\}$: Presisi kuantisasi yang digunakan pada indeks `turbovec` pada langkah $t$. Ukuran memorinya:
  $$M_{\text{turbovec}}(C_t, b_t) = C_t \times \left( \frac{b_t \cdot d}{8} + \text{Metadata Overhead} \right)$$
  Pada presisi 4-bit, rasio penghematan adalah $\approx 7{,}8\times$; pada presisi 2-bit, penghematan mencapai $\approx 15{,}5\times$ dibandingkan representasi FP16.

---

## 2. Skor Tekanan Memori Prediktif ($\Phi_t$)

Untuk mencegah keterlambatan respon adaptasi yang dapat berujung pada terminasi seketika oleh kernel OS (*OOM Killer*), AMU-QUANT menggunakan **Skor Tekanan Memori Prediktif ($\Phi_t$)** yang menggabungkan pembacaan memori absolut dan laju perubahan diferensial (derivatif pertumbuhan memori).

Skor $\Phi_t$ dihitung setiap interval $K$ token generasi:

$$\Phi_t = \alpha \left( \frac{M_{\text{total}}(t)}{M_{\text{limit}}} \right) + \beta \left( \frac{\widehat{\Delta M}_t}{\Delta t} \cdot \frac{\tau_{\text{horizon}}}{M_{\text{limit}}} \right)$$

### Komponen Formula:
1. **Rasio Utilisasi Memori Saat Ini:**  
   $$\frac{M_{\text{total}}(t)}{M_{\text{limit}}}$$  
   Mengukur seberapa dekat pemakaian memori total saat ini terhadap batas toleransi fisik.
2. **Laju Pertumbuhan Memori Diferensial:**  
   $$\frac{\widehat{\Delta M}_t}{\Delta t} = \frac{M_{\text{total}}(t) - M_{\text{total}}(t - K)}{\Delta t_{\text{elapsed}}}$$  
   Mengukur kecepatan kenaikan konsumsi memori per satuan waktu (akibat ekspansi konteks atau aplikasi eksternal yang tiba-tiba mengalokasikan RAM).
3. **Horizon Prediktif ($\tau_{\text{horizon}}$):**  
   Konstanta waktu masa depan (misal $\tau_{\text{horizon}} = 2{,}0\text{ detik}$) untuk memproyeksikan konsumsi memori beberapa langkah ke depan.
4. **Faktor Pembobot ($\alpha, \beta$):**  
   Konstanta kalibrasi yang memenuhi $\alpha + \beta = 1{,}0$ (nilai nominal yang direkomendasikan: $\alpha = 0{,}65$, $\beta = 0{,}35$).

---

## 3. Tiga Rejim Kebijakan Adaptasi (Adaptation Regimes)

Berdasarkan nilai $\Phi_t$, kontroler AMU-QUANT membagi strategi alokasi ke dalam 3 rejim operasional:

```mermaid
stateDiagram-v2
    [*] --> Regime1: Inisialisasi

    Regime1 --> Regime2: Φ_t > θ_low
    Regime2 --> Regime1: Φ_t < (θ_low - δ_hyst)

    Regime2 --> Regime3: Φ_t > θ_high
    Regime3 --> Regime2: Φ_t < (θ_high - δ_hyst)

    state Regime1 {
        description: "Regime 1: High-Throughput Caching"
        Weights: "Jendela Bobot Maksimal di RAM"
        KV_Cache: "Konteks Penuh di llama.cpp"
        Turbovec: "Indeks Bersifat Siaga (Idle)"
    }

    state Regime2 {
        description: "Regime 2: Balanced Offloading"
        Weights: "Prefetch 1-2 Layer di Depan"
        KV_Cache: "Pruning Token Tengah"
        Turbovec: "Arsip Konteks Terkompresi 4-bit (~7.8x)"
    }

    state Regime3 {
        description: "Regime 3: Ultra-Low Survival Mode"
        Weights: "Strict Single Layer Execution (madvise DONTNEED)"
        KV_Cache: "Persempit Sliding Window Ekstrem"
        Turbovec: "Rekuantisasi In-Place ke 2-bit (~15.5x)"
    }
```

### 3.1 Regime 1: High-Throughput Caching ($\Phi_t \le \theta_{\text{low}}$)
* **Kondisi:** Tekanan memori rendah (misal $\Phi_t \le 0{,}65$). Tersedia ruang RAM yang cukup luas.
* **Strategi Bobot:** Sistem operasi diizinkan menahan sebanyak mungkin halaman bobot layer di dalam *page cache* RAM ($|\mathcal{L}_{\text{resident}}| \gg 1$). Sinyal `posix_madvise(..., POSIX_MADV_WILLNEED)` dapat dipanggil secara agresif untuk beberapa layer ke depan.
* **Strategi KV-Cache:** Seluruh konteks dipertahankan dalam KV-cache fisik `llama.cpp` tanpa pemangkasan.
* **Target Kinerja:** Throughput maksimal dengan latensi antartoken minimal karena nihilnya *page fault*.

### 3.2 Regime 2: Balanced Offloading ($\theta_{\text{low}} < \Phi_t \le \theta_{\text{high}}$)
* **Kondisi:** Tekanan memori moderat (misal $0{,}65 < \Phi_t \le 0{,}85$).
* **Strategi KV-Cache:**  
  * Token *attention sinks* (token 0–3) dipertahankan.
  * Token terbaru ($N_{\text{recent}}$ token) dipertahankan.
  * Token-token di tengah dievakuasi dari KV-cache `llama.cpp` menggunakan `llama_kv_cache_seq_rm()` dan diarsipkan ke dalam indeks `turbovec` pada presisi **4-bit** (menghemat ruang $\approx 7{,}8\times$).
* **Strategi Bobot:** Penghematan ruang RAM dari KV-cache dialihkan sebagai *page cache buffer* untuk melakukan *prefetching* 1 layer berikutnya secara sekuensial. Layer yang baru saja dieksekusi segera dilepas halamannya melalui `POSIX_MADV_DONTNEED`.

### 3.3 Regime 3: Ultra-Low Memory Survival Mode ($\Phi_t > \theta_{\text{high}}$)
* **Kondisi:** Tekanan memori kritis (misal $\Phi_t > 0{,}85$). Sistem berada di ambang batas bahaya terminasi OOM.
* **Strategi KV-Cache & turbovec:**  
  * Memanggil fungsi `turbovec_requantize_to_2bit()` untuk mengompresi seluruh vektor yang ada di dalam indeks `turbovec` secara *in-place* dari 4-bit ke **2-bit** (menggandakan penghematan memori menjadi $\approx 15{,}5\times$).
  * Mempersempit ukuran jendela *recent token* ($N_{\text{recent}}$) ke nilai batas minimum.
* **Strategi Bobot:** Menjalankan **Strict Layer-by-Layer Streaming**. Hanya satu layer yang aktif yang diizinkan berada di RAM fisik. Segera setelah perkalian matriks selesai, halaman layer langsung dilepas dengan `POSIX_MADV_DONTNEED`.
* **Mekanisme Recall:** Jika model membutuhkan informasi dari konteks masa lalu, pencarian cepat SIMD dijalankan di `turbovec` (<0,25 ms) untuk memulihkan hanya beberapa token yang paling relevan (*selective retrieval*).

---

## 4. Mekanisme Histeresis Kontrol (Anti-Thrashing)

Untuk mencegah terjadinya perpindahan rejim yang berosilasi secara cepat bolak-balik (*regime flapping/thrashing*) akibat fluktuasi kecil pemakaian RAM, kontroler menerapkan ambang histeresis $\delta_{\text{hyst}}$ (misal $\delta_{\text{hyst}} = 0{,}05$):

* **Kenaikan Rejim:**  
  * Pindah dari Regime 1 ke Regime 2 jika: $\Phi_t > \theta_{\text{low}}$
  * Pindah dari Regime 2 ke Regime 3 jika: $\Phi_t > \theta_{\text{high}}$
* **Penurunan Rejim:**  
  * Pindah dari Regime 3 kembali ke Regime 2 hanya jika: $\Phi_t \le (\theta_{\text{high}} - \delta_{\text{hyst}})$
  * Pindah dari Regime 2 kembali ke Regime 1 hanya jika: $\Phi_t \le (\theta_{\text{low}} - \delta_{\text{hyst}})$

---

## 5. Algoritma & Pseudocode Lengkap Penjadwal AMU-QUANT

Berikut adalah algoritma formal penjadwalan ko-adaptif AMU-QUANT:

```python
# Algoritma 1: AMU-QUANT Co-Adaptive Memory Scheduler
Input:
    ctx             : Pointer ke llama_context
    model           : Pointer ke llama_model
    turbovec_idx    : Pointer ke TurbovecIndex (C-ABI)
    theta_low       : Ambang batas tekanan bawah (default = 0.65)
    theta_high      : Ambang batas tekanan atas (default = 0.85)
    delta_hyst      : Margin histeresis (default = 0.05)
    K               : Frekuensi evaluasi memori (setiap K token)

State Variables:
    current_regime  : Current operational regime (default = REGIME_1)
    quant_bit_width : Current turbovec quantization mode (4 or 2)
    last_eval_time  : Waktu evaluasi terakhir
    last_rss        : Pemakaian RSS memori pada evaluasi terakhir

Procedure Evaluate_And_Adapt(token_step t):
    If (t % K != 0):
        Return  # Lewati jika belum interval evaluasi

    # 1. Akuisisi Telemetri Sistem Operasi
    rss_now = Get_Process_RSS_Bytes()
    mem_total, mem_available = Get_OS_Memory_Telemetry()
    t_now = Get_Current_Time_Seconds()

    # 2. Hitung Laju Pertumbuhan Memori Diferensial
    dt = max(t_now - last_eval_time, 1e-4)
    dM_dt = (rss_now - last_rss) / dt

    # 3. Hitung Skor Tekanan Prediktif Phi_t
    utilization_ratio = rss_now / (mem_total - OS_RESERVED_BYTES)
    projected_growth = (dM_dt * TAU_HORIZON) / (mem_total - OS_RESERVED_BYTES)
    Phi_t = (ALPHA * utilization_ratio) + (BETA * projected_growth)

    # Simpan state untuk iterasi berikutnya
    last_eval_time = t_now
    last_rss = rss_now

    # 4. Transisi Rejim dengan Histeresis
    If current_regime == REGIME_1:
        If Phi_t > theta_high:
            Transition_To_Regime(REGIME_3)
        Else If Phi_t > theta_low:
            Transition_To_Regime(REGIME_2)

    Else If current_regime == REGIME_2:
        If Phi_t > theta_high:
            Transition_To_Regime(REGIME_3)
        Else If Phi_t <= (theta_low - delta_hyst):
            Transition_To_Regime(REGIME_1)

    Else If current_regime == REGIME_3:
        If Phi_t <= (theta_low - delta_hyst):
            Transition_To_Regime(REGIME_1)
        Else If Phi_t <= (theta_high - delta_hyst):
            Transition_To_Regime(REGIME_2)

Procedure Transition_To_Regime(target_regime):
    current_regime = target_regime

    Match target_regime:
        Case REGIME_1:
            # Mode Throughput Penuh
            Set_Prefetch_Depth(layers_ahead = 4)
            Set_Aggressive_Page_Eviction(False)

        Case REGIME_2:
            # Mode Seimbang: Pangkas token tengah & arsipkan ke turbovec 4-bit
            Set_Prefetch_Depth(layers_ahead = 1)
            Set_Aggressive_Page_Eviction(True)
            If quant_bit_width != 4:
                quant_bit_width = 4
            Perform_KV_Eviction_To_Index(bit_width = 4)

        Case REGIME_3:
            # Mode Darurat: Kuantisasi ke 2-bit & Strict Layer-wise Streaming
            Set_Prefetch_Depth(layers_ahead = 0)  # Tanpa prefetch
            Set_Aggressive_Page_Eviction(True)
            If quant_bit_width == 4:
                turbovec_requantize_to_2bit(turbovec_idx)
                quant_bit_width = 2
            Perform_KV_Eviction_To_Index(bit_width = 2)

Procedure Perform_KV_Eviction_To_Index(bit_width):
    p_start, p_end = Identify_Middle_Token_Range(ctx)
    If (p_end - p_start) <= 0:
        Return

    # Ekstraksi representasi float pointer langsung (Zero-Copy)
    For pos from p_start to p_end:
        raw_vec_ptr = llama_get_kv_embedding_ptr(ctx, pos)
        turbovec_insert_zero_copy(turbovec_idx, chunk_id = pos,
                                  float_ptr = raw_vec_ptr,
                                  len = EMBEDDING_DIM)

    # Eksekusi Pemangkasan Fisik pada llama.cpp
    llama_kv_cache_seq_rm(ctx, 0, p_start, p_end)
    llama_kv_cache_seq_add(ctx, 0, p_end, -1, -(p_end - p_start))
```
