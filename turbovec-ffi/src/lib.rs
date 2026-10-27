pub mod turboquant;

use std::panic::catch_unwind;
use std::slice;
use std::sync::RwLock;

use turboquant::{TurboIndex, QuantBitWidth};

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
    pub inner: RwLock<TurboIndex>,
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
pub unsafe extern "C" fn turbovec_remove(
    handle: *mut TurbovecIndexHandle,
    chunk_id: u64,
) -> TurbovecStatus {
    if handle.is_null() {
        return TurbovecStatus::ErrNullPointer;
    }

    let result = catch_unwind(|| {
        let handle_ref = &*handle;
        let mut index = handle_ref.inner.write().unwrap();
        index.remove(chunk_id);
    });

    match result {
        Ok(_) => TurbovecStatus::Ok,
        Err(_) => TurbovecStatus::ErrPanicCaught,
    }
}

#[no_mangle]
pub unsafe extern "C" fn turbovec_remove_batch(
    handle: *mut TurbovecIndexHandle,
    chunk_ids: *const u64,
    count: usize,
) -> TurbovecStatus {
    if handle.is_null() {
        return TurbovecStatus::ErrNullPointer;
    }
    if chunk_ids.is_null() && count > 0 {
        return TurbovecStatus::ErrNullPointer;
    }

    let result = catch_unwind(|| {
        let handle_ref = &*handle;
        let id_slice = slice::from_raw_parts(chunk_ids, count);
        let mut index = handle_ref.inner.write().unwrap();
        index.remove_batch(id_slice);
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
pub unsafe extern "C" fn turbovec_get_memory_usage(
    handle: *const TurbovecIndexHandle,
    out_bytes_used: *mut usize,
) -> TurbovecStatus {
    if handle.is_null() || out_bytes_used.is_null() {
        return TurbovecStatus::ErrNullPointer;
    }

    let result = catch_unwind(|| {
        let handle_ref = &*handle;
        let index = handle_ref.inner.read().unwrap();
        *out_bytes_used = index.memory_usage_bytes();
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
        let _ = Box::from_raw(handle);
    });

    match result {
        Ok(_) => TurbovecStatus::Ok,
        Err(_) => TurbovecStatus::ErrPanicCaught,
    }
}
