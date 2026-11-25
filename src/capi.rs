//! C ABI used by the Python bindings (`python/ane_unleashed`, via ctypes).
//! All functions return 0 on success, -1 on error (see `aneu_last_error`).
use crate::dsp::AneFir;
use crate::grep::{AneGrep, Match};
use crate::query::{AneTable, CmpOp, Filter};
use std::cell::RefCell;
use std::ffi::{c_char, CString};
use std::slice;

thread_local! { static LAST: RefCell<CString> = RefCell::new(CString::default()); }

fn fail<E: std::fmt::Display>(e: E) -> i32 {
    LAST.with(|l| *l.borrow_mut() = CString::new(e.to_string().replace('\0', " ")).unwrap_or_default());
    -1
}

#[no_mangle]
pub extern "C" fn aneu_last_error() -> *const c_char {
    LAST.with(|l| l.borrow().as_ptr())
}

#[no_mangle]
pub extern "C" fn aneu_available() -> i32 {
    match crate::available() {
        Ok(()) => 0,
        Err(e) => fail(e),
    }
}

#[no_mangle]
pub extern "C" fn aneu_ane_power_state() -> i32 {
    crate::telemetry::ane_power_state()
}

// ---------------------------------------------------------------- query
#[repr(C)]
pub struct CFilter {
    pub col: u32,
    /// 0 >, 1 >=, 2 <, 3 <=, 4 ==, 5 !=
    pub op: u32,
    pub value: f32,
}

/// # Safety
/// `cols` must point to `ncols` valid arrays of `rows` f32.
#[no_mangle]
pub unsafe extern "C" fn aneu_table_new_f32(cols: *const *const f32, ncols: usize, rows: usize, out: *mut *mut AneTable) -> i32 {
    let c: Vec<&[f32]> = (0..ncols).map(|i| slice::from_raw_parts(*cols.add(i), rows)).collect();
    match AneTable::from_f32(&c) {
        Ok(t) => {
            *out = Box::into_raw(Box::new(t));
            0
        }
        Err(e) => fail(e),
    }
}

/// # Safety
/// `t` must come from `aneu_table_new_f32`.
#[no_mangle]
pub unsafe extern "C" fn aneu_table_free(t: *mut AneTable) {
    if !t.is_null() {
        drop(Box::from_raw(t));
    }
}

/// # Safety
/// `filters` points to `nf` entries; `mask_out` is NULL or has room for `rows` bytes.
#[no_mangle]
pub unsafe extern "C" fn aneu_table_query(
    t: *const AneTable,
    filters: *const CFilter,
    nf: usize,
    sum_col: i64,
    count: *mut u64,
    sum: *mut f64,
    mask_out: *mut u8,
    ane_seconds: *mut f64,
) -> i32 {
    let t = &*t;
    let fs: Vec<Filter> = match slice::from_raw_parts(filters, nf)
        .iter()
        .map(|f| {
            let op = match f.op {
                0 => CmpOp::Gt,
                1 => CmpOp::Ge,
                2 => CmpOp::Lt,
                3 => CmpOp::Le,
                4 => CmpOp::Eq,
                5 => CmpOp::Ne,
                x => return Err(format!("bad op {x}")),
            };
            Ok(Filter { col: f.col as usize, op, value: f.value })
        })
        .collect::<Result<_, _>>()
    {
        Ok(v) => v,
        Err(e) => return fail(e),
    };
    let sc = if sum_col >= 0 { Some(sum_col as usize) } else { None };
    match t.query(&fs, sc, !mask_out.is_null()) {
        Ok(r) => {
            *count = r.count;
            if !sum.is_null() {
                *sum = r.sum.unwrap_or(0.0);
            }
            if let Some(m) = r.mask {
                std::ptr::copy_nonoverlapping(m.as_ptr(), mask_out, m.len());
            }
            if !ane_seconds.is_null() {
                *ane_seconds = r.ane_s;
            }
            0
        }
        Err(e) => fail(e),
    }
}

// ---------------------------------------------------------------- grep
/// # Safety
/// `pats[i]` points to `lens[i]` bytes.
#[no_mangle]
pub unsafe extern "C" fn aneu_grep_new(
    pats: *const *const u8,
    lens: *const usize,
    n: usize,
    projections: usize,
    rows_per_dispatch: usize,
    out: *mut *mut AneGrep,
) -> i32 {
    let p: Vec<&[u8]> = (0..n).map(|i| slice::from_raw_parts(*pats.add(i), *lens.add(i))).collect();
    match AneGrep::new(&p, projections, rows_per_dispatch) {
        Ok(g) => {
            *out = Box::into_raw(Box::new(g));
            0
        }
        Err(e) => fail(e),
    }
}

/// # Safety
/// `g` from `aneu_grep_new`.
#[no_mangle]
pub unsafe extern "C" fn aneu_grep_free(g: *mut AneGrep) {
    if !g.is_null() {
        drop(Box::from_raw(g));
    }
}

pub struct MatchList(Vec<Match>);

/// Scan `text`; returns a match list handle and its length.
/// # Safety
/// `text` points to `len` bytes.
#[no_mangle]
pub unsafe extern "C" fn aneu_grep_scan(g: *const AneGrep, text: *const u8, len: usize, out: *mut *mut MatchList, n: *mut usize) -> i32 {
    match (*g).scan(slice::from_raw_parts(text, len)) {
        Ok((m, _)) => {
            *n = m.len();
            *out = Box::into_raw(Box::new(MatchList(m)));
            0
        }
        Err(e) => fail(e),
    }
}

/// Copy matches out and free the list.
/// # Safety
/// `pos`/`pat` have room for the length returned by `aneu_grep_scan`.
#[no_mangle]
pub unsafe extern "C" fn aneu_matches_take(l: *mut MatchList, pos: *mut u64, pat: *mut u32) {
    let l = Box::from_raw(l);
    for (i, m) in l.0.iter().enumerate() {
        *pos.add(i) = m.pos as u64;
        *pat.add(i) = m.pattern;
    }
}

// ---------------------------------------------------------------- fir
/// # Safety
/// `irs[c]` points to `taps` f32 for each of `nch` channels.
#[no_mangle]
pub unsafe extern "C" fn aneu_fir_new(irs: *const *const f32, nch: usize, taps: usize, block: usize, frames: usize, out: *mut *mut AneFir) -> i32 {
    let h: Vec<&[f32]> = (0..nch).map(|i| slice::from_raw_parts(*irs.add(i), taps)).collect();
    match AneFir::new(&h, if block == 0 { None } else { Some(block) }, frames) {
        Ok(f) => {
            *out = Box::into_raw(Box::new(f));
            0
        }
        Err(e) => fail(e),
    }
}

/// # Safety
/// `f` from `aneu_fir_new`.
#[no_mangle]
pub unsafe extern "C" fn aneu_fir_free(f: *mut AneFir) {
    if !f.is_null() {
        drop(Box::from_raw(f));
    }
}

/// # Safety
/// `x[c]` and `y[c]` point to `n` f32 for each channel.
#[no_mangle]
pub unsafe extern "C" fn aneu_fir_process(f: *const AneFir, x: *const *const f32, y: *const *mut f32, n: usize) -> i32 {
    let f = &*f;
    let xs: Vec<&[f32]> = (0..f.channels).map(|i| slice::from_raw_parts(*x.add(i), n)).collect();
    match f.process(&xs) {
        Ok((out, _)) => {
            for (c, o) in out.iter().enumerate() {
                std::ptr::copy_nonoverlapping(o.as_ptr(), *y.add(c), n);
            }
            0
        }
        Err(e) => fail(e),
    }
}
