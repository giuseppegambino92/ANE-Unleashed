//! Multi-core CPU baselines for the three engines (what a well-written CPU
//! implementation would do — not strawmen).
use crate::grep::Match;
use crate::query::Filter;
use aho_corasick::{AhoCorasick, MatchKind};
use rayon::prelude::*;
use realfft::RealFftPlanner;

#[inline(always)]
fn apply<F: Fn(f32) -> bool>(mask: &mut [u8], col: &[f32], f: F) {
    for (m, &v) in mask.iter_mut().zip(col) {
        *m &= f(v) as u8;
    }
}

/// COUNT / SUM with native fp32 semantics: column-at-a-time over 16K-row chunks
/// (branch-free, auto-vectorised compare loops), rayon across chunks.
pub fn query(cols: &[&[f32]], filters: &[Filter], sum_col: Option<usize>) -> (u64, f64) {
    use crate::query::CmpOp::*;
    let rows = cols[0].len();
    const CH: usize = 1 << 14;
    (0..rows.div_ceil(CH))
        .into_par_iter()
        .map_init(
            || vec![0u8; CH],
            |mask, k| {
                let lo = k * CH;
                let hi = (lo + CH).min(rows);
                let mask = &mut mask[..hi - lo];
                mask.fill(1);
                for f in filters {
                    let col = &cols[f.col][lo..hi];
                    let kv = f.value;
                    match f.op {
                        Gt => apply(mask, col, |v| v > kv),
                        Ge => apply(mask, col, |v| v >= kv),
                        Lt => apply(mask, col, |v| v < kv),
                        Le => apply(mask, col, |v| v <= kv),
                        Eq => apply(mask, col, |v| v == kv),
                        Ne => apply(mask, col, |v| v != kv),
                    }
                }
                let c: u64 = mask.iter().map(|&m| m as u64).sum();
                let s = sum_col
                    .map(|sc| {
                        let col = &cols[sc][lo..hi];
                        mask.chunks(1024)
                            .zip(col.chunks(1024))
                            .map(|(m, v)| m.iter().zip(v).map(|(&m, &v)| if m != 0 { v } else { 0.0 }).sum::<f32>() as f64)
                            .sum::<f64>()
                    })
                    .unwrap_or(0.0);
                (c, s)
            },
        )
        .reduce(|| (0, 0.0), |a, b| (a.0 + b.0, a.1 + b.1))
}

/// All overlapping occurrences of all patterns (Aho-Corasick, the engine behind
/// ripgrep's multi-literal search), parallel over chunks with pattern-length overlap.
pub fn grep(text: &[u8], patterns: &[&[u8]]) -> Vec<Match> {
    let ac = AhoCorasick::builder().match_kind(MatchKind::Standard).build(patterns).expect("aho-corasick");
    let maxlen = patterns.iter().map(|p| p.len()).max().unwrap_or(1);
    const CH: usize = 4 << 20;
    let mut v: Vec<Match> = (0..text.len().div_ceil(CH))
        .into_par_iter()
        .flat_map_iter(|k| {
            let lo = k * CH;
            let hi = (lo + CH + maxlen - 1).min(text.len());
            let own_end = (lo + CH).min(text.len());
            ac.find_overlapping_iter(&text[lo..hi])
                .filter(move |m| lo + m.start() < own_end)
                .map(move |m| Match { pos: lo + m.start(), pattern: m.pattern().as_u32() })
                .collect::<Vec<_>>()
        })
        .collect();
    v.sort_unstable();
    v
}

/// Count-only variant (no allocation of the match list) — the fastest fair CPU scan.
pub fn grep_count(text: &[u8], patterns: &[&[u8]]) -> usize {
    let ac = AhoCorasick::builder().match_kind(MatchKind::Standard).build(patterns).expect("aho-corasick");
    let maxlen = patterns.iter().map(|p| p.len()).max().unwrap_or(1);
    const CH: usize = 4 << 20;
    (0..text.len().div_ceil(CH))
        .into_par_iter()
        .map(|k| {
            let lo = k * CH;
            let hi = (lo + CH + maxlen - 1).min(text.len());
            let own_end = (lo + CH).min(text.len());
            ac.find_overlapping_iter(&text[lo..hi]).filter(|m| lo + m.start() < own_end).count()
        })
        .sum()
}

extern "C" {
    fn vDSP_conv(a: *const f32, ia: isize, f: *const f32, if_: isize, c: *mut f32, ic: isize, n: usize, p: usize);
}

/// Direct-form FIR with Accelerate `vDSP_conv`, one channel per core.
pub fn fir_direct(x: &[&[f32]], h: &[&[f32]]) -> Vec<Vec<f32>> {
    x.par_iter()
        .zip(h.par_iter())
        .map(|(x, h)| {
            let t = h.len();
            let n = x.len();
            let mut padded = vec![0f32; n + t - 1];
            padded[t - 1..].copy_from_slice(x);
            let mut y = vec![0f32; n];
            // vDSP_conv correlates; a negative filter stride (pointing at the last tap) convolves
            unsafe { vDSP_conv(padded.as_ptr(), 1, h.as_ptr().add(t - 1), -1, y.as_mut_ptr(), 1, n, t) };
            y
        })
        .collect()
}

/// FFT overlap-save FIR (realfft / rustfft NEON), one channel per core.
pub fn fir_fft(x: &[&[f32]], h: &[&[f32]]) -> Vec<Vec<f32>> {
    x.par_iter()
        .zip(h.par_iter())
        .map(|(x, h)| {
            let t = h.len();
            let n = x.len();
            let fft_n = (4 * t).next_power_of_two();
            let hop = fft_n - (t - 1);
            let mut planner = RealFftPlanner::<f32>::new();
            let fwd = planner.plan_fft_forward(fft_n);
            let inv = planner.plan_fft_inverse(fft_n);
            let mut hbuf = vec![0f32; fft_n];
            hbuf[..t].copy_from_slice(h);
            let mut hspec = fwd.make_output_vec();
            fwd.process(&mut hbuf, &mut hspec).unwrap();
            let scale = 1.0 / fft_n as f32;
            let mut y = vec![0f32; n];
            let mut buf = vec![0f32; fft_n];
            let mut spec = fwd.make_output_vec();
            let mut s = 0usize;
            while s < n {
                // buf covers x[s - (t-1) .. s - (t-1) + fft_n)
                for (i, v) in buf.iter_mut().enumerate() {
                    let idx = s as isize - (t as isize - 1) + i as isize;
                    *v = if idx >= 0 && (idx as usize) < n { x[idx as usize] } else { 0.0 };
                }
                fwd.process(&mut buf, &mut spec).unwrap();
                for (a, b) in spec.iter_mut().zip(&hspec) {
                    *a = *a * *b * scale;
                }
                inv.process(&mut spec, &mut buf).unwrap();
                let take = hop.min(n - s);
                y[s..s + take].copy_from_slice(&buf[t - 1..t - 1 + take]);
                s += hop;
            }
            y
        })
        .collect()
}
