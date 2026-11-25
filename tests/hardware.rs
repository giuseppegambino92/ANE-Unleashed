//! Real-hardware tests: every test executes on the ANE and is checked against an
//! exact CPU model. Run with `cargo test --release -- --test-threads=1`.
use ane_unleashed::query::{reference_query, AneTable, CmpOp, Filter};
use ane_unleashed::{cpu_baseline, dsp::AneFir, grep::AneGrep};

fn rng(seed: u64) -> impl FnMut() -> u64 {
    let mut s = seed;
    move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        s
    }
}
fn gauss(n: usize, seed: u64, scale: f32) -> Vec<f32> {
    let mut r = rng(seed);
    (0..n)
        .map(|_| {
            let a = (r() >> 11) as f64 / (1u64 << 53) as f64;
            let b = (r() >> 11) as f64 / (1u64 << 53) as f64;
            ((-2.0 * (a + 1e-300).ln()).sqrt() * (6.283185307 * b).cos()) as f32 * scale
        })
        .collect()
}

#[test]
fn query_matches_reference_exactly() {
    let rows = 20_000_123; // 2 chunks, ragged tail
    let a = gauss(rows, 1, 100.0);
    let b = gauss(rows, 2, 100.0);
    let c = gauss(rows, 3, 10.0);
    let cols = [a.as_slice(), b.as_slice(), c.as_slice()];
    let t = AneTable::from_f32(&cols).unwrap();
    let cases: Vec<Vec<Filter>> = vec![
        vec![Filter { col: 0, op: CmpOp::Gt, value: 50.0 }, Filter { col: 1, op: CmpOp::Lt, value: -20.0 }],
        vec![Filter { col: 0, op: CmpOp::Ge, value: 0.1 }], // non-fp16 constant
        vec![Filter { col: 1, op: CmpOp::Le, value: 0.0 }, Filter { col: 2, op: CmpOp::Ne, value: 0.0 }],
        vec![Filter { col: 2, op: CmpOp::Lt, value: 1e9 }], // matches everything incl. padding rows
    ];
    for f in &cases {
        let r = t.query(f, Some(2), true).unwrap();
        let (cnt, sum, mask) = reference_query(&cols, f, Some(2));
        assert_eq!(r.count, cnt, "count mismatch for {f:?}");
        assert!(r.mask.as_ref().unwrap() == &mask, "mask mismatch for {f:?}");
        let rel = (r.sum.unwrap() - sum).abs() / sum.abs().max(1.0);
        eprintln!("{f:?}: count={cnt} sum_ane={:.3} sum_ref={sum:.3} rel_err={rel:.2e}", r.sum.unwrap());
        assert!(rel < 2e-3, "sum error too large: {rel}");
        // CPU fp32 baseline agrees except within fp16 rounding of the constants/values
        let (ccpu, _) = cpu_baseline::query(&cols, f, Some(2));
        eprintln!("  cpu fp32 count={ccpu} (diff {})", ccpu as i64 - cnt as i64);
    }
}

#[test]
fn grep_finds_exactly_what_aho_corasick_finds() {
    let mut r = rng(7);
    let n = 40 << 20; // 40 MiB synthetic log
    let words: [&[u8]; 12] = [b"GET ", b"POST ", b"/api/v1/users ", b"200 ", b"404 ", b"timeout ", b"INFO ", b"WARN ", b"user=", b"id=", b"\n", b"latency_ms=12 "];
    let mut text = Vec::with_capacity(n + 64);
    while text.len() < n {
        let w = words[(r() % words.len() as u64) as usize];
        text.extend_from_slice(w);
        if r() % 7 == 0 {
            text.extend_from_slice(format!("{} ", r() % 100000).as_bytes());
        }
    }
    text.truncate(n);
    let pats: Vec<&[u8]> = vec![b"ERROR", b"timeout", b"404 ", b"user=4242", b"latency_ms=12 POST", b"segfault at 0x", b"id=99999", b"\nWARN timeout"];
    for proj in [1usize, 2] {
        let g = AneGrep::new(&pats, proj, 256).unwrap();
        let (m, st) = g.scan(&text).unwrap();
        let reference = cpu_baseline::grep(&text, &pats);
        eprintln!(
            "proj={proj}: {} matches, flagged {}/{} blocks ({:.3}%), ane {:.1} ms copy {:.1} ms confirm {:.1} ms",
            m.len(),
            st.flagged_blocks,
            st.total_blocks,
            100.0 * st.flagged_blocks as f64 / st.total_blocks as f64,
            st.ane_s * 1e3,
            st.copy_s * 1e3,
            st.confirm_s * 1e3
        );
        assert_eq!(m.len(), reference.len());
        assert!(m == reference, "match sets differ");
    }
}

#[test]
fn grep_row_boundaries() {
    // place a pattern across every row boundary of a 16 KiB-row layout
    let mut text = vec![b'.'; 16384 * 8 + 100];
    for row in 1..8 {
        let at = row * 16384 - 5;
        text[at..at + 11].copy_from_slice(b"needle_here");
    }
    let pats: Vec<&[u8]> = vec![b"needle_here", b"le_h"];
    let g = AneGrep::new(&pats, 2, 16).unwrap();
    let (m, _) = g.scan(&text).unwrap();
    assert_eq!(m, cpu_baseline::grep(&text, &pats));
    assert_eq!(m.len(), 14);
}

#[test]
fn fir_matches_fft_reference() {
    let ch = 4;
    let taps = 4096;
    let n = 48_000 * 3 + 777;
    let irs: Vec<Vec<f32>> = (0..ch)
        .map(|c| gauss(taps, 10 + c as u64, 0.05).iter().enumerate().map(|(i, v)| v * (-(i as f32) / 600.0).exp()).collect())
        .collect();
    let x: Vec<Vec<f32>> = (0..ch).map(|c| gauss(n, 100 + c as u64, 0.3)).collect();
    let ir: Vec<&[f32]> = irs.iter().map(|v| v.as_slice()).collect();
    let xs: Vec<&[f32]> = x.iter().map(|v| v.as_slice()).collect();
    let f = AneFir::new(&ir, None, 64).unwrap(); // 56 fresh frames per dispatch -> 6 overlap-save dispatches
    let (y, st) = f.process(&xs).unwrap();
    let r = cpu_baseline::fir_fft(&xs, &ir);
    let rd = cpu_baseline::fir_direct(&xs, &ir);
    for c in 0..ch {
        let (mut e, mut s, mut ed) = (0f64, 0f64, 0f64);
        for i in 0..n {
            e += ((y[c][i] - r[c][i]) as f64).powi(2);
            ed += ((rd[c][i] - r[c][i]) as f64).powi(2);
            s += (r[c][i] as f64).powi(2);
        }
        let snr = 10.0 * (s / e).log10();
        eprintln!("ch{c}: ANE SNR {snr:.1} dB vs FFT; vDSP direct vs FFT {:.1} dB; dispatches {}", 10.0 * (s / ed).log10(), st.dispatches);
        assert!(snr > 50.0, "SNR too low: {snr}");
    }
}
