//! ANE-Query: columnar predicate filtering and COUNT / SUM aggregation.
//!
//! `SELECT COUNT(*), SUM(c) FROM t WHERE a > X AND b <= Y ...`
//!
//! Each column chunk is an IOSurface tensor `[1, 1024, 8, 2048]` (16,777,216
//! rows); rows of 2048 fp32 elements are 64-byte aligned, so a dense column
//! slice maps onto the surface with no padding. The generated MIL program casts
//! inputs to fp16, evaluates every predicate as an elementwise compare, ANDs
//! them, and reduces along W, producing one partial COUNT and SUM per 2048 rows.
//!
//! ## Exactness contract (verified in tests against `reference_query`)
//! * Predicates are evaluated on values rounded to fp16 **with ties away from
//!   zero** (that is what the ANE's fp32->fp16 conversion does — not IEEE RNE).
//!   For fp16 columns, or fp32 columns holding fp16-representable values, the
//!   selection is therefore identical to an fp32 evaluation.
//! * COUNT is exact (partials <= 2048 are exactly representable in fp16).
//! * SUM is approximate: values are fp16-rounded and each 2048-row partial sum is
//!   rounded once to fp16 (relative error per partial <= 2^-11).
use crate::iosurface_tensor::{AneTensor, DType};
use crate::mil_codegen::{MilBuilder, Var};
use crate::runtime::AneProgram;
use crate::{Error, Result};
use half::f16;
use rayon::prelude::*;
use std::sync::Arc;
use std::time::Instant;

pub const CHUNK_C: usize = 1024;
pub const CHUNK_H: usize = 8;
pub const CHUNK_W: usize = 2048;
pub const CHUNK_ROWS: usize = CHUNK_C * CHUNK_H * CHUNK_W;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CmpOp {
    Gt,
    Ge,
    Lt,
    Le,
    Eq,
    Ne,
}

impl CmpOp {
    fn mil(self) -> &'static str {
        match self {
            CmpOp::Gt => "greater",
            CmpOp::Ge => "greater_equal",
            CmpOp::Lt => "less",
            CmpOp::Le => "less_equal",
            CmpOp::Eq => "equal",
            CmpOp::Ne => "not_equal",
        }
    }
    pub fn eval(self, v: f32, k: f32) -> bool {
        match self {
            CmpOp::Gt => v > k,
            CmpOp::Ge => v >= k,
            CmpOp::Lt => v < k,
            CmpOp::Le => v <= k,
            CmpOp::Eq => v == k,
            CmpOp::Ne => v != k,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Filter {
    pub col: usize,
    pub op: CmpOp,
    pub value: f32,
}

/// fp32 -> fp16 rounding with ties away from zero (the ANE's conversion rule).
pub fn f16_round_away(x: f32) -> f32 {
    if !x.is_finite() || x == 0.0 {
        return f16::from_f32(x).to_f32();
    }
    let a = x.abs() as f64;
    if a >= 65520.0 {
        return f32::INFINITY.copysign(x);
    }
    let e2 = (a.log2().floor() as i32).max(-14);
    let ulp = 2f64.powi(e2 - 10);
    let r = ((a / ulp) + 0.5).floor() * ulp;
    (r as f32).copysign(x)
}

/// Rewrite a constant so that comparing fp16 data against an fp16 constant is
/// equivalent to comparing the same fp16 data against the original fp32 constant.
fn fp16_constant(op: CmpOp, k: f32) -> f32 {
    let h = f16::from_f32(k);
    let hv = h.to_f32();
    if hv == k || !k.is_finite() {
        return hv;
    }
    // fp16 neighbours of k
    let down = if hv < k { hv } else { prev_f16(h).to_f32() };
    let up = if hv > k { hv } else { next_f16(h).to_f32() };
    match op {
        CmpOp::Gt | CmpOp::Le => down, // v > k  <=> v > down ;  v <= k <=> v <= down
        CmpOp::Ge | CmpOp::Lt => up,   // v >= k <=> v >= up  ;  v < k  <=> v < up
        CmpOp::Eq | CmpOp::Ne => k,    // never equal for fp16 data; caller handles
    }
}
/// Predicate actually sent to the ANE: (op, finite fp16 constant). Constants that
/// overflow fp16 become ±inf above; since no finite fp16 lies beyond ±65504,
/// `v < +inf` is rewritten `v <= 65504`, `v >= +inf` -> `v > 65504`, etc.
pub fn fp16_predicate(op: CmpOp, k: f32) -> (CmpOp, f32) {
    const MAX: f32 = 65504.0;
    let kk = fp16_constant(op, k);
    if kk.is_finite() {
        return (op, kk);
    }
    match (op, kk > 0.0) {
        (CmpOp::Lt, true) => (CmpOp::Le, MAX),
        (CmpOp::Le, true) => (CmpOp::Le, MAX), // v <= +inf excluding +inf itself is unreachable: data inf only from overflow
        (CmpOp::Gt, true) | (CmpOp::Ge, true) => (CmpOp::Gt, MAX),
        (CmpOp::Gt, false) => (CmpOp::Ge, -MAX),
        (CmpOp::Ge, false) => (CmpOp::Ge, -MAX),
        (CmpOp::Lt, false) | (CmpOp::Le, false) => (CmpOp::Lt, -MAX),
        (CmpOp::Eq, _) => (CmpOp::Gt, MAX), // equality with an out-of-range constant: matches only overflowed values
        (CmpOp::Ne, _) => (CmpOp::Le, MAX),
    }
}

fn next_f16(h: f16) -> f16 {
    let b = h.to_bits();
    if h.to_f32() >= 0.0 { f16::from_bits(b + 1) } else if b == 0x8000 { f16::from_bits(1) } else { f16::from_bits(b - 1) }
}
fn prev_f16(h: f16) -> f16 {
    let b = h.to_bits();
    if h.to_f32() > 0.0 { f16::from_bits(b - 1) } else if b == 0 { f16::from_bits(0x8001) } else { f16::from_bits(b + 1) }
}

/// A table whose columns live in ANE-ready IOSurfaces (ingested once).
pub struct AneTable {
    pub rows: usize,
    pub dtype: DType,
    /// columns[c][chunk]
    columns: Vec<Vec<AneTensor>>,
    pub ingest_s: f64,
}

#[derive(Clone, Debug, Default)]
pub struct QueryResult {
    pub count: u64,
    /// SUM(sum_col) over selected rows (fp16-approximate), if requested
    pub sum: Option<f64>,
    /// per-row selection (1/0), if requested
    pub mask: Option<Vec<u8>>,
    pub compile_ms: f64,
    pub ane_s: f64,
    pub total_s: f64,
}

struct Plan {
    prog: Arc<AneProgram>,
    cols: Vec<usize>,
    has_sum: bool,
    want_mask: bool,
}

impl AneTable {
    /// Ingest fp32 columns (all the same length) into IOSurface chunks.
    pub fn from_f32(cols: &[&[f32]]) -> Result<Self> {
        Self::ingest(cols, DType::F32)
    }
    /// Ingest fp16 columns (half the bytes; exact predicate semantics).
    pub fn from_f16(cols: &[&[f16]]) -> Result<Self> {
        Self::ingest(cols, DType::F16)
    }

    fn ingest<T: Copy + Send + Sync + Default>(cols: &[&[T]], dtype: DType) -> Result<Self> {
        let t0 = Instant::now();
        let rows = cols.first().map(|c| c.len()).unwrap_or(0);
        if rows == 0 || cols.iter().any(|c| c.len() != rows) {
            return Err(Error::Shape("columns must be non-empty and equal length".into()));
        }
        let nchunks = rows.div_ceil(CHUNK_ROWS);
        let mut columns = Vec::new();
        for col in cols {
            let chunks: Vec<AneTensor> = (0..nchunks)
                .into_par_iter()
                .map(|k| {
                    let t = AneTensor::new([1, CHUNK_C, CHUNK_H, CHUNK_W], dtype)?;
                    let lo = k * CHUNK_ROWS;
                    let hi = (lo + CHUNK_ROWS).min(rows);
                    t.write_dense(&col[lo..hi]); // zero-fills the tail
                    Ok(t)
                })
                .collect::<Result<_>>()?;
            columns.push(chunks);
        }
        Ok(Self { rows, dtype, columns, ingest_s: t0.elapsed().as_secs_f64() })
    }

    pub fn ncols(&self) -> usize {
        self.columns.len()
    }

    fn plan(&self, filters: &[Filter], sum_col: Option<usize>, want_mask: bool) -> Result<Plan> {
        let mut cols: Vec<usize> = filters.iter().map(|f| f.col).chain(sum_col).collect();
        cols.sort();
        cols.dedup();
        if let Some(&c) = cols.iter().find(|&&c| c >= self.ncols()) {
            return Err(Error::Shape(format!("column {c} out of range")));
        }
        let shape = [1, CHUNK_C, CHUNK_H, CHUNK_W];
        let mut b = MilBuilder::new();
        let mut vars: Vec<(usize, Var)> = Vec::new();
        for &c in &cols {
            let x = b.input(&format!("col{c}"), self.dtype, &shape);
            let x16 = if self.dtype == DType::F16 { x } else { b.cast(&x, DType::F16) };
            vars.push((c, x16));
        }
        let get = |c: usize| vars.iter().find(|(k, _)| *k == c).unwrap().1.clone();
        let mut mask: Option<Var> = None;
        for f in filters {
            let (op, k) = fp16_predicate(f.op, f.value);
            let kv = b.scalar_f16(k);
            let p = b.binary(op.mil(), &get(f.col), &kv);
            mask = Some(match mask {
                None => p,
                Some(m) => b.binary("logical_and", &m, &p),
            });
        }
        let mask = mask.ok_or_else(|| Error::Unsupported("query needs at least one filter".into()))?;
        let m16 = b.cast(&mask, DType::F16);
        let cnt = b.reduce("sum", &m16, &[3]); // [1,1024,8,1], each <= 2048: exact
        let mut outs = vec![cnt];
        if let Some(sc) = sum_col {
            let prod = b.binary("mul", &m16, &get(sc));
            outs.push(b.reduce("sum", &prod, &[3]));
        }
        if want_mask {
            outs.push(b.cast(&mask, DType::U8));
        }
        let refs: Vec<&Var> = outs.iter().collect();
        let (mil, blob) = b.finish(&refs);
        let prog = AneProgram::compile(&mil, &blob)?;
        Ok(Plan { prog, cols, has_sum: sum_col.is_some(), want_mask })
    }

    /// Run `COUNT(*) [, SUM(sum_col)] WHERE AND(filters)` on the ANE.
    pub fn query(&self, filters: &[Filter], sum_col: Option<usize>, want_mask: bool) -> Result<QueryResult> {
        let t0 = Instant::now();
        let plan = self.plan(filters, sum_col, want_mask)?;
        let compile_ms = t0.elapsed().as_secs_f64() * 1e3;
        let part_shape = [1, CHUNK_C, CHUNK_H, 1];
        let cnt_t = AneTensor::new(part_shape, DType::F16)?;
        let sum_t = if plan.has_sum { Some(AneTensor::new(part_shape, DType::F16)?) } else { None };
        let mask_t = if plan.want_mask { Some(AneTensor::new([1, CHUNK_C, CHUNK_H, CHUNK_W], DType::U8)?) } else { None };
        let mut outs: Vec<&AneTensor> = vec![&cnt_t];
        if let Some(s) = &sum_t {
            outs.push(s);
        }
        if let Some(m) = &mask_t {
            outs.push(m);
        }
        let nchunks = self.columns[0].len();
        let mut count = 0u64;
        let mut sum = 0f64;
        let mut mask = if want_mask { Some(vec![0u8; self.rows]) } else { None };
        let mut ane_s = 0.0;
        for k in 0..nchunks {
            let ins: Vec<&AneTensor> = plan.cols.iter().map(|&c| &self.columns[c][k]).collect();
            let req = plan.prog.bind(&ins, &outs)?;
            let ta = Instant::now();
            req.run()?;
            ane_s += ta.elapsed().as_secs_f64();
            let valid = (self.rows - k * CHUNK_ROWS).min(CHUNK_ROWS);
            let cnts: Vec<f16> = cnt_t.read_dense();
            let sums: Option<Vec<f16>> = sum_t.as_ref().map(|s| s.read_dense());
            // partial i covers rows [i*2048, (i+1)*2048)
            let full = valid / CHUNK_W;
            for i in 0..full {
                count += cnts[i].to_f32() as u64;
                if let Some(s) = &sums {
                    sum += s[i].to_f64();
                }
            }
            let tail = valid % CHUNK_W;
            if tail > 0 {
                // the tail partial includes zero-filled padding rows (0.0 in every column);
                // remove them from COUNT if 0.0 satisfies the predicate. SUM is unaffected.
                let pad_matches = filters.iter().all(|f| {
                    let (op, k) = fp16_predicate(f.op, f.value);
                    op.eval(0.0, k)
                });
                let c = cnts[full].to_f32() as u64;
                let pad = (CHUNK_W - tail) as u64;
                count += if pad_matches { c - pad } else { c };
                if let Some(s) = &sums {
                    sum += s[full].to_f64();
                }
            }
            if let (Some(m), Some(mt)) = (mask.as_mut(), &mask_t) {
                mt.read_with(|src| m[k * CHUNK_ROWS..k * CHUNK_ROWS + valid].copy_from_slice(&src[..valid]));
            }
        }
        Ok(QueryResult {
            count,
            sum: if sum_col.is_some() { Some(sum) } else { None },
            mask,
            compile_ms,
            ane_s,
            total_s: t0.elapsed().as_secs_f64(),
        })
    }
}

/// Bit-exact CPU model of the ANE semantics (fp16 ties-away rounding), used for
/// verification. Returns (count, exact f64 sum of fp16-rounded values, mask).
pub fn reference_query(cols: &[&[f32]], filters: &[Filter], sum_col: Option<usize>) -> (u64, f64, Vec<u8>) {
    let rows = cols[0].len();
    let ks: Vec<(CmpOp, f32)> = filters.iter().map(|f| fp16_predicate(f.op, f.value)).collect();
    let mask: Vec<u8> = (0..rows)
        .into_par_iter()
        .map(|r| filters.iter().zip(&ks).all(|(f, &(op, k))| op.eval(f16_round_away(cols[f.col][r]), k)) as u8)
        .collect();
    let count = mask.iter().map(|&m| m as u64).sum();
    let sum = sum_col
        .map(|c| (0..rows).into_par_iter().filter(|&r| mask[r] == 1).map(|r| f16_round_away(cols[c][r]) as f64).sum())
        .unwrap_or(0.0);
    (count, sum, mask)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn round_away() {
        // 0.5 + 2^-12 is the exact midpoint between fp16 0.5 and 0.5 + 2^-11
        assert_eq!(f16_round_away(0.5 + 2f32.powi(-12)), 0.5 + 2f32.powi(-11));
        assert_eq!(f16_round_away(-(0.5 + 2f32.powi(-12))), -(0.5 + 2f32.powi(-11)));
        assert_eq!(f16_round_away(1.0), 1.0);
        assert_eq!(f16_round_away(65519.0), 65504.0);
        assert!(f16_round_away(65520.0).is_infinite());
    }
    #[test]
    fn constants() {
        let k = 0.1f32; // not representable
        for op in [CmpOp::Gt, CmpOp::Ge, CmpOp::Lt, CmpOp::Le] {
            let kk = fp16_constant(op, k);
            for bits in 0u16..0x7c00 {
                let v = f16::from_bits(bits).to_f32();
                assert_eq!(op.eval(v, k), op.eval(v, kk), "{op:?} v={v}");
                assert_eq!(op.eval(-v, k), op.eval(-v, kk), "{op:?} v={}", -v);
            }
        }
    }
}
