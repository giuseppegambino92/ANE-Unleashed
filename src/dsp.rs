//! ANE-DSP: multi-channel long FIR / impulse-response convolution.
//!
//! The ANE's conv kernel is at most 15 wide, so a T-tap FIR cannot be a single
//! 1-D conv. Instead the signal is cut into frames of B samples and the FIR is
//! rewritten as a block-Toeplitz convolution over frames:
//!
//! ```text
//!   y_frame[m] = sum_{j=0..J} H_j · x_frame[m - j],   H_j[b, b'] = h[j·B + b - b'],   J = ceil((T-1)/B)
//! ```
//!
//! In MIL: input `[1, C, M, B]` (natural sample order, fp32) -> cast fp16 ->
//! transpose to `[1, C, B, M]` -> reshape `[1, C·B, 1, M]` -> grouped conv
//! (groups = C, B -> B channels, kernel width J+1 <= 15, causal left pad J) ->
//! inverse reshape/transpose -> fp32 output. All on the ANE; the CPU only memcpys
//! per-channel segments in and out. Successive dispatches overlap by J frames
//! (overlap-save) so arbitrarily long signals are filtered seamlessly.
//!
//! Precision: inputs, taps and intermediates are fp16 (accumulation is wider);
//! expect ~55-60 dB SNR against an fp64 reference.
use crate::iosurface_tensor::{AneTensor, DType, ANE_MAX_DIM};
use crate::mil_codegen::{ConvSpec, MilBuilder};
use crate::runtime::AneProgram;
use crate::{Error, Result};
use half::f16;
use rayon::prelude::*;
use std::sync::Arc;
use std::time::Instant;

pub struct AneFir {
    pub channels: usize,
    pub taps: usize,
    pub block: usize,
    pub frames: usize,
    j: usize,
    groups: Vec<FirGroup>,
    pub compile_ms: f64,
    pub weight_bytes: usize,
}

struct FirGroup {
    ch0: usize,
    nch: usize,
    prog: Arc<AneProgram>,
}

#[derive(Clone, Debug, Default)]
pub struct FirStats {
    pub copy_s: f64,
    pub ane_s: f64,
    pub total_s: f64,
    pub dispatches: usize,
}

/// Smallest power-of-two block size keeping the frame kernel <= 15 wide.
pub fn choose_block(taps: usize) -> usize {
    let mut b = 16;
    while taps.saturating_sub(1).div_ceil(b) + 1 > 15 {
        b *= 2;
    }
    b
}

impl AneFir {
    /// `irs[c]` is the impulse response of channel c (all the same length).
    /// `frames` = frames of `block` samples per dispatch (<= 16384).
    pub fn new(irs: &[&[f32]], block: Option<usize>, frames: usize) -> Result<Self> {
        let channels = irs.len();
        let taps = irs.first().map(|h| h.len()).unwrap_or(0);
        if channels == 0 || taps == 0 || irs.iter().any(|h| h.len() != taps) {
            return Err(Error::Shape("need >= 1 channel and equal-length impulse responses".into()));
        }
        let b = block.unwrap_or_else(|| choose_block(taps));
        let j = taps.saturating_sub(1).div_ceil(b);
        if j + 1 > 15 {
            return Err(Error::Unsupported(format!("block {b} too small for {taps} taps (frame kernel {} > 15)", j + 1)));
        }
        if b > ANE_MAX_DIM || frames > ANE_MAX_DIM || frames <= j {
            return Err(Error::Shape(format!("need block <= 16384 and {j} < frames <= 16384")));
        }
        let per_group = (ANE_MAX_DIM / b).max(1);
        let t0 = Instant::now();
        let mut groups = Vec::new();
        let mut weight_bytes = 0;
        let mut ch0 = 0;
        while ch0 < channels {
            let nch = per_group.min(channels - ch0);
            let (mil, blob) = Self::program(&irs[ch0..ch0 + nch], b, j, frames);
            weight_bytes += blob.len();
            let prog = AneProgram::compile(&mil, &blob)?;
            groups.push(FirGroup { ch0, nch, prog });
            ch0 += nch;
        }
        Ok(Self {
            channels,
            taps,
            block: b,
            frames,
            j,
            groups,
            compile_ms: t0.elapsed().as_secs_f64() * 1e3,
            weight_bytes,
        })
    }

    fn program(irs: &[&[f32]], b: usize, j: usize, m: usize) -> (String, Vec<u8>) {
        let c = irs.len();
        let k = j + 1;
        let t = irs[0].len();
        // weight [c*b, b, 1, k]; conv is cross-correlation with left pad j, so tap
        // index kk multiplies frame m + kk - j  => block lag jj = j - kk
        let mut w = vec![f16::ZERO; c * b * b * k];
        w.par_chunks_mut(b * b * k).enumerate().for_each(|(cc, wc)| {
            let h = irs[cc];
            for bo in 0..b {
                for bi in 0..b {
                    for kk in 0..k {
                        let jj = j - kk;
                        let tap = (jj * b + bo) as isize - bi as isize;
                        if tap >= 0 && (tap as usize) < t {
                            wc[(bo * b + bi) * k + kk] = f16::from_f32(h[tap as usize]);
                        }
                    }
                }
            }
        });
        let mut mb = MilBuilder::new();
        let x = mb.input("x", DType::F32, &[1, c, m, b]);
        let xh = mb.cast(&x, DType::F16);
        let xt = mb.transpose(&xh, [0, 1, 3, 2]);
        let xr = mb.reshape(&xt, &[1, c * b, 1, m]);
        let wv = mb.tensor_f16(&[c * b, b, 1, k], &w);
        let yr = mb.conv(&xr, &wv, None, &ConvSpec { kernel: [c * b, b, 1, k], groups: c, strides: [1, 1], pad: [0, 0, j, 0], dilations: [1, 1] });
        let yt = mb.reshape(&yr, &[1, c, b, m]);
        let yh = mb.transpose(&yt, [0, 1, 3, 2]);
        let y = mb.cast(&yh, DType::F32);
        mb.finish(&[&y])
    }

    /// Filter every channel (`x[c]`, equal lengths). Output has the input length
    /// (causal FIR, zero initial state).
    pub fn process(&self, x: &[&[f32]]) -> Result<(Vec<Vec<f32>>, FirStats)> {
        if x.len() != self.channels {
            return Err(Error::Shape(format!("expected {} channels", self.channels)));
        }
        let n = x[0].len();
        if x.iter().any(|c| c.len() != n) {
            return Err(Error::Shape("channels must have equal length".into()));
        }
        let t_all = Instant::now();
        let (b, m, j) = (self.block, self.frames, self.j);
        let new_frames = m - j;
        let seg = new_frames * b; // fresh output samples per dispatch
        let mut out: Vec<Vec<f32>> = (0..self.channels).map(|_| vec![0f32; n]).collect();
        let mut st = FirStats::default();
        for g in &self.groups {
            let xin = AneTensor::new([1, g.nch, m, b], DType::F32)?;
            let yout = AneTensor::new([1, g.nch, m, b], DType::F32)?;
            let req = g.prog.bind(&[&xin], &[&yout])?;
            let mut s = 0usize;
            while s < n {
                let t = Instant::now();
                // input frames cover samples [s - j*b, s - j*b + m*b)
                xin.write_with(|dst| {
                    let dst = unsafe { std::slice::from_raw_parts_mut(dst.as_mut_ptr() as *mut f32, dst.len() / 4) };
                    dst.par_chunks_mut(m * b).enumerate().for_each(|(ci, d)| {
                        let src = x[g.ch0 + ci];
                        // d[i] = src[start + i] where in range, else 0 (left history / right tail)
                        let start = s as isize - (j * b) as isize;
                        let lo = (-start).max(0) as usize; // first d index with a valid sample
                        let src_lo = (start.max(0)) as usize;
                        let len = (d.len() - lo.min(d.len())).min(n.saturating_sub(src_lo));
                        d[..lo].fill(0.0);
                        d[lo..lo + len].copy_from_slice(&src[src_lo..src_lo + len]);
                        d[lo + len..].fill(0.0);
                    });
                });
                st.copy_s += t.elapsed().as_secs_f64();
                let t = Instant::now();
                req.run()?;
                st.ane_s += t.elapsed().as_secs_f64();
                st.dispatches += 1;
                let t = Instant::now();
                let take = seg.min(n - s);
                yout.read_with(|src| {
                    let src = unsafe { std::slice::from_raw_parts(src.as_ptr() as *const f32, src.len() / 4) };
                    out[g.ch0..g.ch0 + g.nch].par_iter_mut().enumerate().for_each(|(ci, o)| {
                        let base = ci * m * b + j * b;
                        o[s..s + take].copy_from_slice(&src[base..base + take]);
                    });
                });
                st.copy_s += t.elapsed().as_secs_f64();
                s += seg;
            }
        }
        st.total_s = t_all.elapsed().as_secs_f64();
        Ok((out, st))
    }
}
