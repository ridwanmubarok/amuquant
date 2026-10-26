# DESAIN JEMBATAN FOREIGN FUNCTION INTERFACE (turbovec-ffi)

Dokumen ini memaparkan spesifikasi teknis antarmuka *Foreign Function Interface* (C-ABI FFI) yang menjembatani struktur data C++ pada `llama.cpp` dengan pustaka kompresi vektor berkecepatan tinggi `turbovec` yang ditulis dalam Rust.

---

## 1. Landasan Algoritma: TurboQuant (Google Research, ICLR 2026)

Algoritma **TurboQuant** ([Zandieh et al., ICLR 2026](https://arxiv.org/abs/2504.19874)) dirancang untuk mengatasi kelemahan metode kuantisasi vektor konvensional seperti *Product Quantization* (PQ), yang membutuhkan tahap pelatihan K-Means yang lambat dan sensitif terhadap pergeseran distribusi data (*distribution shift*).

```
Vektor Asli (FP16/FP32, d=4096)
               │
               ▼
   [ Rotasi Ortogonal Acak Q ]  ───> Mentransformasikan vektor ke distribusi Gaussian N(0, 1/d)
               │
               ▼
[ Lloyd-Max Scalar Quantizer ] ───> Memetakan komponen ke level diskrit optimal
               │
       ┌───────┴───────┐
       ▼               ▼
Mode 4-bit (16 level)  Mode 2-bit (4 level)
(Kompresi ~7,8x)       (Kompresi ~15,5x)
```

### Tahapan Matematis Kompresi TurboQuant:
1. **Rotasi Ortogonal Acak:**  
   Vektor masukan $\mathbf{x} \in \mathbb{R}^d$ dikalikan dengan matriks ortogonal acak $\mathbf{Q} \in \mathbb{R}^{d \times d}$ yang memenuhi $\mathbf{Q}^T \mathbf{Q} = \mathbf{I}$:
   $$\mathbf{y} = \mathbf{Q} \mathbf{x}$$
   Berdasarkan Teorema Batas Pusat (*Central Limit Theorem*), perkalian ini meratakan komponen vektor sehingga setiap elemen $y_i$ terdistribusi identik dan independen mengikuti distribusi normal Gaussian:
   $$y_i \sim \mathcal{N}\left(0, \frac{\|\mathbf{x}\|_2^2}{d}\right)$$
2. **Kuantisasi Skalar Lloyd-Max:**  
   Karena distribusi telah menjadi Gaussian standar, titik-titik representasi kuantisasi skalar (*codebook centroids*) dapat dihitung secara analitik satu kali di awal (*zero-training*).
   * Pada **Mode 4-bit**: Setiap komponen $y_i$ dipetakan ke dalam salah satu dari 16 interval batas, lalu dikodekan menjadi nilai integer 4-bit ($0 \dots 15$).
   * Pada **Mode 2-bit**: Setiap komponen $y_i$ dipetakan ke dalam 4 interval batas, lalu dikodekan menjadi nilai integer 2-bit ($0 \dots 3$).
3. **Pencarian Kemiripan Kosinus Berbasis SIMD:**  
   Untuk menghitung kemiripan kosinus antara vektor kueri $\mathbf{q}$ dan kumpulan vektor terindeks $\mathbf{x}^{(j)}$, TurboQuant memanfaatkan dekompresi cepat tervektorisasi menggunakan register CPU SIMD (AVX2, AVX-512, atau ARM NEON).

---

## 2. Spesifikasi Antarmuka C-ABI (`turbovec_ffi.h`)

Berikut adalah definisi header formal C/C++ yang diekspos oleh modul jembatan `turbovec-ffi`:

```c
/**
 * @file turbovec_ffi.h
 * @brief Foreign Function Interface (C-ABI) for turbovec TurboQuant Vector Index.
 * 
 * Jembatan antarmuka C/C++ untuk pustaka Rust turbovec.
 * Mendukung pengiriman pointer zero-copy, kuantisasi dinamis 4-bit & 2-bit,
 * serta pencarian cepat SIMD.
 */

#ifndef TURBOVEC_FFI_H
#define TURBOVEC_FFI_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Kode status eksekusi FFI.
 */
typedef enum {
    TURBOVEC_STATUS_OK                 =  0,
    TURBOVEC_ERR_NULL_POINTER          = -1,
    TURBOVEC_ERR_INVALID_DIMENSION     = -2,
    TURBOVEC_ERR_UNSUPPORTED_BIT_WIDTH = -3,
    TURBOVEC_ERR_OUT_OF_MEMORY         = -4,
    TURBOVEC_ERR_PANIC_CAUGHT          = -5,
    TURBOVEC_ERR_INDEX_NOT_FOUND       = -6
} TurbovecStatus;

/**
 * @brief Opaque pointer ke struktur indeks TurboVec di dalam heap Rust.
 */
typedef struct TurbovecIndexHandle TurbovecIndexHandle;

/**
 * @brief Menginisialisasi indeks vektor TurboQuant baru di dalam memori Rust.
 * 
 * @param dimension Dimensi vektor (contoh: 4096 atau 5120).
 * @param bit_width Mode kuantisasi awal (harus bernilai 4 atau 2).
 * @param out_handle Pointer keluaran untuk menyimpan handle instance indeks.
 * @return TurbovecStatus Kode status keberhasilan atau jenis error.
 */
TurbovecStatus turbovec_init(
    uint32_t dimension,
    uint8_t bit_width,
    TurbovecIndexHandle** out_handle
);

/**
 * @brief Menyisipkan vektor konteks langsung dari pointer C++ tanpa duplikasi memory (Zero-Copy).
 * 
 * @param handle Handle indeks yang valid.
 * @param chunk_id ID unik potongan konteks (biasanya nomor urut posisi token awal).
 * @param float_ptr Pointer langsung ke buffer array float kontinu (32-bit float).
 * @param len Panjang elemen float (harus sama persis dengan dimension indeks).
 * @return TurbovecStatus Kode status operasi penyisipan.
 */
TurbovecStatus turbovec_insert_zero_copy(
    TurbovecIndexHandle* handle,
    uint64_t chunk_id,
    const float* float_ptr,
    size_t len
);

/**
 * @brief Menjalankan pencarian k-tetangga terdekat (Top-K) berbasis kemiripan kosinus SIMD.
 * 
 * @param handle Handle indeks yang valid.
 * @param query_ptr Pointer ke buffer vektor kueri float kontinu.
 * @param top_k Jumlah potongan konteks paling relevan yang ingin diambil.
 * @param out_ids Array yang dialokasikan pemanggil untuk menampung ID chunk hasil.
 * @param out_scores Array yang dialokasikan pemanggil untuk menampung skor kemiripan.
 * @param out_count Jumlah hasil aktual yang ditemukan (bisa <= top_k).
 * @return TurbovecStatus Kode status operasi pencarian (<0.25 ms).
 */
TurbovecStatus turbovec_search_simd(
    const TurbovecIndexHandle* handle,
    const float* query_ptr,
    uint32_t top_k,
    uint64_t* out_ids,
    float* out_scores,
    uint32_t* out_count
);

/**
 * @brief Mengompresi ulang seluruh indeks dari mode 4-bit ke 2-bit secara in-place.
 * 
 * Dipanggil oleh kontroler AMU-QUANT saat sistem memasuki Regime 3 (Survival Mode).
 * 
 * @param handle Handle indeks yang valid.
 * @return TurbovecStatus Kode status rekuantisasi.
 */
TurbovecStatus turbovec_requantize_to_2bit(
    TurbovecIndexHandle* handle
);

/**
 * @brief Mendapatkan statistik penggunaan memori riil indeks dalam satuan byte.
 * 
 * @param handle Handle indeks yang valid.
 * @param out_bytes_used Pointer untuk menampung total konsumsi byte di heap Rust.
 * @return TurbovecStatus Kode status.
 */
TurbovecStatus turbovec_get_memory_usage(
    const TurbovecIndexHandle* handle,
    size_t* out_bytes_used
);

/**
 * @brief Menghancurkan indeks dan membebaskan seluruh alokasi memori di heap Rust.
 * 
 * @param handle Handle indeks yang akan dibebaskan.
 * @return TurbovecStatus Kode status pembebasan memori.
 */
TurbovecStatus turbovec_free(
    TurbovecIndexHandle* handle
);

#ifdef __cplusplus
}
#endif

#endif /* TURBOVEC_FFI_H */
```

---

## 3. Implementasi Sisi Rust (`ffi.rs`)

Struktur internal pada modul Rust dibungkus dalam tipe `#[repr(C)]` dengan penanganan pengecualian ketat:

```rust
// File: turbovec-ffi/src/lib.rs

use std::panic::catch_unwind;
use std::slice;
use std::sync::RwLock;

// Asumsi: crate turbovec mengekspos TurboIndex core
use turbovec::{TurboIndex, QuantBitWidth};

#[repr(C)]
#[derive(Debug, PartialEq, Eq)]
pub enum TurbovecStatus {
    Ok = 0,
    ErrNullPointer = -1,
    ErrInvalidDimension = -2,
    ErrUnsupportedBitWidth = -3,
    ErrOutOfMemory = -4,
    ErrPanicCaught = -5,
    ErrIndexNotFound = -6,
}

pub struct TurbovecIndexHandle {
    inner: RwLock<TurboIndex>,
}

#[no_mangle]
pub extern "C" fn turbovec_init(
    dimension: u32,
    bit_width: u8,
    out_handle: *mut *mut TurbovecIndexHandle,
) -> TurbovecStatus {
    if out_handle.is_null() {
        return TurbovecStatus::ErrNullPointer;
    }

    if dimension == 0 {
        return TurbovecStatus::ErrInvalidDimension;
    }

    let q_mode = match bit_width {
        4 => QuantBitWidth::Bits4,
        2 => QuantBitWidth::Bits2,
        _ => return TurbovecStatus::ErrUnsupportedBitWidth,
    };

    let result = catch_unwind(|| {
        let index = TurboIndex::new(dimension as usize, q_mode);
        let boxed = Box::new(TurbovecIndexHandle {
            inner: RwLock::new(index),
        });
        unsafe {
            *out_handle = Box::into_raw(boxed);
        }
    });

    match result {
        Ok(_) => TurbovecStatus::Ok,
        Err(_) => TurbovecStatus::ErrPanicCaught,
    }
}

#[no_mangle]
pub unsafe extern "C" fn turbovec_insert_zero_copy(
    handle: *mut TurbovecIndexHandle,
    chunk_id: u64,
    float_ptr: *const f32,
    len: usize,
) -> TurbovecStatus {
    if handle.is_null() || float_ptr.is_null() {
        return TurbovecStatus::ErrNullPointer;
    }

    let result = catch_unwind(|| {
        let handle_ref = &*handle;
        // Zero-copy view dari pointer C++
        let slice_view = slice::from_raw_parts(float_ptr, len);

        let mut index = handle_ref.inner.write().unwrap();
        index.insert_slice(chunk_id, slice_view);
    });

    match result {
        Ok(_) => TurbovecStatus::Ok,
        Err(_) => TurbovecStatus::ErrPanicCaught,
    }
}

#[no_mangle]
pub unsafe extern "C" fn turbovec_search_simd(
    handle: *const TurbovecIndexHandle,
    query_ptr: *const f32,
    top_k: u32,
    out_ids: *mut u64,
    out_scores: *mut f32,
    out_count: *mut u32,
) -> TurbovecStatus {
    if handle.is_null() || query_ptr.is_null() || out_ids.is_null() || out_scores.is_null() || out_count.is_null() {
        return TurbovecStatus::ErrNullPointer;
    }

    let result = catch_unwind(|| {
        let handle_ref = &*handle;
        let index = handle_ref.inner.read().unwrap();
        let query_slice = slice::from_raw_parts(query_ptr, index.dimension());

        // Eksekusi pencarian kemiripan kosinus berbasis SIMD
        let search_results = index.search_simd(query_slice, top_k as usize);
        let count = search_results.len().min(top_k as usize);

        for i in 0..count {
            *out_ids.add(i) = search_results[i].id;
            *out_scores.add(i) = search_results[i].score;
        }
        *out_count = count as u32;
    });

    match result {
        Ok(_) => TurbovecStatus::Ok,
        Err(_) => TurbovecStatus::ErrPanicCaught,
    }
}

#[no_mangle]
pub unsafe extern "C" fn turbovec_requantize_to_2bit(
    handle: *mut TurbovecIndexHandle,
) -> TurbovecStatus {
    if handle.is_null() {
        return TurbovecStatus::ErrNullPointer;
    }

    let result = catch_unwind(|| {
        let handle_ref = &*handle;
        let mut index = handle_ref.inner.write().unwrap();
        index.requantize_in_place(QuantBitWidth::Bits2);
    });

    match result {
        Ok(_) => TurbovecStatus::Ok,
        Err(_) => TurbovecStatus::ErrPanicCaught,
    }
}

#[no_mangle]
pub unsafe extern "C" fn turbovec_free(
    handle: *mut TurbovecIndexHandle,
) -> TurbovecStatus {
    if handle.is_null() {
        return TurbovecStatus::ErrNullPointer;
    }

    let result = catch_unwind(|| {
        // Mengonversi kembali raw pointer ke Box dan menjatuhkannya (drop)
        let _ = Box::from_raw(handle);
    });

    match result {
        Ok(_) => TurbovecStatus::Ok,
        Err(_) => TurbovecStatus::ErrPanicCaught,
    }
}
```

---

## 4. Mekanisme Kuantisasi Ulang In-Place (4-bit $\to$ 2-bit)

Ketika sistem mengalami tekanan memori ekstrim (Regime 3), waktu merupakan parameter krusial. Melakukan kompresi ulang dari vektor float asli tidak memungkinkan karena representasi float asli telah dibuang untuk menghemat RAM.

Oleh karena itu, modul `turbovec` mengimplementasikan algoritma **Direct Bit-Truncation & Codebook Remapping**:
* Indeks kuantisasi 4-bit ($q_4 \in \{0, 1, \dots, 15\}$) merepresentasikan 16 interval nilai Gaussian.
* Mode 2-bit membagi distribusi Gaussian yang sama menjadi 4 interval kuantil.
* Fungsi rekuantisasi memetakan 4 kode 4-bit berurutan ke dalam 1 kode 2-bit:
  $$q_2 = \left\lfloor \frac{q_4}{4} \right\rfloor$$
* Dua kode 4-bit yang sebelumnya menempati 1 byte ($8\text{ bit}$) dipadatkan menjadi empat kode 2-bit dalam 1 byte yang sama, memangkas ukuran indeks hingga $50\%$ secara instan tanpa re-alokasi buffer baru.

---

## 5. Validasi Efisiensi SIMD & Overhead Budget

Untuk memenuhi target latensi antartoken yang kompetitif, jembatan FFI dirancang dengan batas toleransi latensi (*budget overhead*) yang ketat:

| Operasi | Target Latensi | Strategi Optimasi |
|:---|:---:|:---|
| **Zero-copy insertion per chunk** | $< 15\,\mu\text{s}$ | Nihil alokasi malloc per token; rotasi matriks SIMD langsung pada register AVX/NEON |
| **SIMD Search Top-5 (Kueri 4096-dim)** | $< 250\,\mu\text{s}$ ($0{,}25\text{ ms}$) | Dot-product akumulasi integer tervektorisasi penuh tanpa unboxing float |
| **Requantize to 2-bit (10.000 token)** | $< 2{,}5\text{ ms}$ | Operasi bit-shift tervektorisasi paralel multi-threading |
| **Batas FFI Call Boundary Overhead** | $< 80\text{ ns}$ | Penggunaan konvensi panggilan `extern "C"` standar tanpa layer serialization |
