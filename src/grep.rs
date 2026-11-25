//! ANE-Grep: multi-pattern byte scanning with an exact ANE prefilter.
//!
//! Each dispatch takes `C` consecutive 16 KiB segments of the text as the
//! channels of a `[1, C, 1, 16384]` uint8 tensor (a plain contiguous copy of the
//! bytes). For every pattern `j` and projection `q` a grouped conv computes
//!
//! ```text
//!   h[j,q](i) = sum_k w[j,q,k] * x[i+k]  -  sum_k w[j,q,k] * p_j[k]      (fused fp16-exact bias)
//! ```
//!
//! with small integer weights. The conv accumulates exactly and rounds once
//! (verified: identical to exact-integer-then-round on 100% of outputs), so on a
//! true occurrence every `h[j,q]` is exactly 0 — **no false negatives**.
//! Zeros are counted per position with a second grouped conv; `count >= m`
//! marks a candidate, and a 1x64 max-pool reduces candidates to one flag per
//! 64-byte block. The CPU confirms full patterns only inside flagged blocks (and
//! at the <= 14 positions per segment whose window crosses into the next
//! segment), so the final match set is exact.
//!
//! Limits: the ANE conv kernel is at most 15 wide, so the prefilter uses the first
//! `min(len, 15)` bytes of each pattern; longer patterns are confirmed on the CPU.
use crate::iosurface_tensor::{AneTensor, DType};
use crate::mil_codegen::{ConvSpec, MilBuilder};
use crate::runtime::AneProgram;
use crate::{Error, Result};
use half::f16;
use rayon::prelude::*;
use std::sync::Arc;
use std::time::Instant;

pub const ROW: usize = 16384;
pub const BLOCK: usize = 64;
pub const MAX_PREFIX: usize = 15;
/// Segments (16 KiB each) per dispatch; further capped so segments * patterns *
/// projections <= 16384 channels. (An earlier rows-on-H layout sent
/// ANECompilerService into a multi-hour compile; see HARDWARE_NOTES.md.)
pub const MAX_ROWS_PER_DISPATCH: usize = 16384;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Match {
    pub pos: usize,
    pub pattern: u32,
}

#[derive(Clone, Debug, Default)]
pub struct ScanStats {
    pub bytes: usize,
    pub flagged_blocks: usize,
    pub total_blocks: usize,
    pub copy_s: f64,
    pub ane_s: f64,
    pub confirm_s: f64,
    pub total_s: f64,
}

pub struct AneGrep {
    patterns: Vec<Vec<u8>>,
    prefix: usize,
    projections: usize,
    rows_per_dispatch: usize,
    prog: Arc<AneProgram>,
    /// first byte -> pattern ids
    by_first: Vec<Vec<u32>>,
    pub compile_ms: f64,
}

/// Small deterministic PRNG (splitmix64) so weights are reproducible per pattern set.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
}

impl AneGrep {
    /// `projections`: hash projections per pattern (2 is a good default; more = fewer
    /// false-positive blocks, more ANE work). `rows_per_dispatch`: 16 KiB rows per
    /// ANE call (<= 16384).
    pub fn new(patterns: &[&[u8]], projections: usize, rows_per_dispatch: usize) -> Result<Self> {
        if patterns.is_empty() || patterns.iter().any(|p| p.is_empty()) {
            return Err(Error::Unsupported("need >= 1 non-empty pattern".into()));
        }
        if !(1..=MAX_ROWS_PER_DISPATCH).contains(&rows_per_dispatch) || projections == 0 {
            return Err(Error::Shape(format!("rows_per_dispatch must be 1..={MAX_ROWS_PER_DISPATCH}, projections >= 1")));
        }
        let prefix = patterns.iter().map(|p| p.len().min(MAX_PREFIX)).max().unwrap();
        let np = patterns.len();
        let m = projections;
        // conv weights [np*m, 1, 1, prefix] and fp16-exact integer biases
        let mut w = vec![f16::ZERO; np * m * prefix];
        let mut bias = vec![f16::ZERO; np * m];
        let mut rng = Rng(0xA4E_u64 ^ (np as u64) << 32);
        for (j, p) in patterns.iter().enumerate() {
            let lp = p.len().min(MAX_PREFIX);
            for r in 0..m {
                let ch = j * m + r;
                let mut tries = 0;
                loop {
                    tries += 1;
                    let mut acc: i64 = 0;
                    for k in 0..prefix {
                        let v = if k < lp {
                            let mag = 1 + (rng.next() % 3) as i64; // 1..=3
                            if rng.next() & 1 == 0 { mag } else { -mag }
                        } else {
                            0
                        };
                        w[ch * prefix + k] = f16::from_f32(v as f32);
                        acc += v * p.get(k).copied().unwrap_or(0) as i64;
                    }
                    // integers with |v| <= 2048 are exact in fp16
                    if acc.abs() <= 2048 || tries > 10_000 {
                        let b = f16::from_f32(-acc as f32);
                        if b.to_f32() as i64 != -acc {
                            return Err(Error::Unsupported("could not find fp16-exact projection bias".into()));
                        }
                        bias[ch] = b;
                        break;
                    }
                }
            }
        }

        // Channel-major layout: segment c of the dispatch lives in channel c, so every
        // per-pattern channel is computed with grouped convs (groups = segments).
        let r = rows_per_dispatch.min((crate::iosurface_tensor::ANE_MAX_DIM / (np * m)).max(1));
        let g = np * m;
        let mut wt = Vec::with_capacity(r * g * prefix);
        let mut bt = Vec::with_capacity(r * g);
        for _ in 0..r {
            wt.extend_from_slice(&w);
            bt.extend_from_slice(&bias);
        }
        let mut b = MilBuilder::new();
        let x = b.input("text", DType::U8, &[1, r, 1, ROW]);
        let xh = b.cast(&x, DType::F16);
        let wv = b.tensor_f16(&[r * g, 1, 1, prefix], &wt);
        let bv = b.tensor_f16(&[r * g], &bt);
        // h[c, j*m+q](i) = sum_k w[j,q,k] * (x_c[i+k] - p_j[k])  (exact integer, rounded once)
        let h = b.conv(
            &xh,
            &wv,
            Some(&bv),
            &ConvSpec { kernel: [r * g, 1, 1, prefix], groups: r, strides: [1, 1], pad: [0, 0, 0, prefix - 1], dilations: [1, 1] },
        );
        let zero = b.scalar_f16(0.0);
        let z = b.binary("equal", &h, &zero);
        let z16 = b.cast(&z, DType::F16);
        // zeros per segment position (a true match contributes m zeros)
        let ones = b.tensor_f16(&[r, g, 1, 1], &vec![f16::ONE; r * g]);
        let cnt = b.conv(&z16, &ones, None, &ConvSpec { kernel: [r, g, 1, 1], groups: r, strides: [1, 1], pad: [0; 4], dilations: [1, 1] });
        let thr = b.scalar_f16(m as f32);
        let hit = b.binary("greater_equal", &cnt, &thr);
        let hit16 = b.cast(&hit, DType::F16);
        let pooled = b.max_pool_w(&hit16, BLOCK);
        let flags = b.cast(&pooled, DType::U8);
        let (mil, blob) = b.finish(&[&flags]);
        let t0 = Instant::now();
        let prog = AneProgram::compile(&mil, &blob)?;
        let compile_ms = t0.elapsed().as_secs_f64() * 1e3;

        let mut by_first = vec![Vec::new(); 256];
        for (j, p) in patterns.iter().enumerate() {
            by_first[p[0] as usize].push(j as u32);
        }
        Ok(Self {
            patterns: patterns.iter().map(|p| p.to_vec()).collect(),
            prefix,
            projections,
            rows_per_dispatch: r,
            prog,
            by_first,
            compile_ms,
        })
    }

    pub fn prefix_len(&self) -> usize {
        self.prefix
    }
    pub fn projections(&self) -> usize {
        self.projections
    }
    /// 16 KiB segments processed per ANE dispatch.
    pub fn segments_per_dispatch(&self) -> usize {
        self.rows_per_dispatch
    }

    #[inline]
    fn confirm_at(&self, text: &[u8], pos: usize, out: &mut Vec<Match>) {
        for &j in &self.by_first[text[pos] as usize] {
            let p = &self.patterns[j as usize];
            if pos + p.len() <= text.len() && &text[pos..pos + p.len()] == p.as_slice() {
                out.push(Match { pos, pattern: j });
            }
        }
    }

    /// Find every (overlapping) occurrence of every pattern. Matches are sorted by
    /// (position, pattern id).
    pub fn scan(&self, text: &[u8]) -> Result<(Vec<Match>, ScanStats)> {
        let t_all = Instant::now();
        let r = self.rows_per_dispatch;
        let bytes_per = r * ROW;
        let blocks_per_row = ROW / BLOCK;
        let input = AneTensor::new([1, r, 1, ROW], DType::U8)?;
        let flags = AneTensor::new([1, r, 1, blocks_per_row], DType::U8)?;
        let req = self.prog.bind(&[&input], &[&flags])?;
        let mut st = ScanStats { bytes: text.len(), ..Default::default() };
        let edge = self.prefix - 1; // positions per segment whose window crosses into the next one

        // Pipeline: a producer thread copies + dispatches chunk k+1 on the ANE while
        // this thread confirms chunk k's flagged blocks in parallel (rayon).
        let (tx, rx) = std::sync::mpsc::sync_channel::<(usize, usize, Vec<u8>)>(2);
        let mut per_row: Vec<Vec<Match>> = Vec::new();
        let (input_r, flags_r) = (&input, &flags);
        let producer_result = std::thread::scope(|sc| {
            // `move` hands the (Send, !Sync) bound request to the producer thread
            let producer = sc.spawn(move || -> Result<(f64, f64)> {
                let (mut copy_s, mut ane_s) = (0.0, 0.0);
                for (d, chunk) in text.chunks(bytes_per).enumerate() {
                    let t = Instant::now();
                    input_r.write_with(|dst| {
                        dst[..chunk.len()].copy_from_slice(chunk);
                        dst[chunk.len()..].fill(0);
                    });
                    copy_s += t.elapsed().as_secs_f64();
                    let t = Instant::now();
                    req.run()?;
                    ane_s += t.elapsed().as_secs_f64();
                    let fl = flags_r.read_with(|src| src[..r * blocks_per_row].to_vec());
                    if tx.send((d, chunk.len(), fl)).is_err() {
                        break;
                    }
                }
                drop(tx);
                Ok((copy_s, ane_s))
            });
            for (d, len, fl) in rx {
                let t = Instant::now();
                let base = d * bytes_per;
                let rows_used = len.div_ceil(ROW);
                let res: Vec<(Vec<Match>, usize)> = (0..rows_used)
                    .into_par_iter()
                    .map(|row| {
                        let mut out = Vec::new();
                        let row_base = base + row * ROW;
                        let row_end = (row_base + ROW).min(text.len());
                        let f = &fl[row * blocks_per_row..(row + 1) * blocks_per_row];
                        let mut flagged = 0;
                        for (bk, &v) in f.iter().enumerate() {
                            if v == 0 {
                                continue;
                            }
                            flagged += 1;
                            let lo = row_base + bk * BLOCK;
                            for pos in lo..(lo + BLOCK).min(row_end) {
                                self.confirm_at(text, pos, &mut out);
                            }
                        }
                        // windows crossing into the next segment were computed against zero padding
                        if edge > 0 && row_end == row_base + ROW && row_end < text.len() && f[blocks_per_row - 1] == 0 {
                            for pos in row_end - edge..row_end {
                                self.confirm_at(text, pos, &mut out);
                            }
                        }
                        (out, flagged)
                    })
                    .collect();
                for (m, f) in res {
                    st.flagged_blocks += f;
                    per_row.push(m);
                }
                st.total_blocks += rows_used * blocks_per_row;
                st.confirm_s += t.elapsed().as_secs_f64();
            }
            producer.join().expect("grep producer panicked")
        });
        let (copy_s, ane_s) = producer_result?;
        st.copy_s = copy_s;
        st.ane_s = ane_s;
        let mut matches: Vec<Match> = per_row.into_iter().flatten().collect();
        matches.sort_unstable();
        st.total_s = t_all.elapsed().as_secs_f64();
        Ok((matches, st))
    }
}
