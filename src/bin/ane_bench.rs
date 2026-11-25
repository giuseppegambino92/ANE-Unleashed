//! ane-bench — CPU (multi-core) vs ANE-Unleashed, with correctness cross-checks
//! and every power signal available at the current privilege level.
//!
//!   ane-bench [--quick] [--only query|grep|dsp] [--secs N] [--out DIR]
//!
//! Run with root (or a NOPASSWD sudoers rule for /usr/bin/powermetrics) to get
//! CPU / GPU / ANE rail watts; otherwise only per-process CPU energy, ANE
//! power-gate residency and IOReport GPU energy are recorded.
use ane_unleashed::cpu_baseline;
use ane_unleashed::dsp::AneFir;
use ane_unleashed::grep::AneGrep;
use ane_unleashed::query::{reference_query, AneTable, CmpOp, Filter};
use ane_unleashed::telemetry::{publish_progress, Measurement, Meter, PmSample, PowerMetrics};
use std::fmt::Write as _;
use std::time::{Duration, Instant};

struct Opts {
    rounds: usize,
    quick: bool,
    only: Option<String>,
    secs: f64,
    out: String,
}

#[derive(Clone)]
struct Row {
    workload: String,
    engine: String,
    bytes_per_iter: f64,
    iters: usize,
    m: Measurement,
    note: String,
}

impl Row {
    fn lat_ms(&self) -> f64 {
        self.m.wall_s / self.iters as f64 * 1e3
    }
    fn gbps(&self) -> f64 {
        self.bytes_per_iter * self.iters as f64 / self.m.wall_s / 1e9
    }
    /// Power attributable to the workload (W): rails above idle when available.
    fn power_w(&self, idle: &Option<PmSample>) -> (Option<f64>, &'static str) {
        match (&self.m.rails, idle) {
            (Some(r), Some(i)) => {
                let cpu = (r.cpu_mw - i.cpu_mw).max(0.0) / 1e3;
                let ane = (r.ane_mw - i.ane_mw).max(0.0) / 1e3;
                let gpu = (r.gpu_mw - i.gpu_mw).max(0.0) / 1e3;
                (Some(cpu + ane + gpu), "rails above idle (CPU+GPU+ANE)")
            }
            _ if self.engine.starts_with("cpu") => (Some(self.m.proc_cpu_j / self.m.wall_s), "process CPU energy (rusage)"),
            _ => (None, "ANE watts need root"),
        }
    }
}

fn rng(seed: u64) -> impl FnMut() -> u64 {
    let mut s = seed | 1;
    move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        s
    }
}

fn gauss(n: usize, seed: u64, scale: f32) -> Vec<f32> {
    use rayon::prelude::*;
    (0..n)
        .into_par_iter()
        .chunks(1 << 16)
        .flat_map_iter(|ix| {
            let mut r = rng(seed ^ (ix[0] as u64).wrapping_mul(0x9E3779B97F4A7C15));
            ix.into_iter()
                .map(move |_| {
                    let a = (r() >> 11) as f64 / (1u64 << 53) as f64;
                    let b = (r() >> 11) as f64 / (1u64 << 53) as f64;
                    ((-2.0 * (a + 1e-300).ln()).sqrt() * (6.283185307 * b).cos()) as f32 * scale
                })
                .collect::<Vec<_>>()
        })
        .collect()
}

/// Run `f` repeatedly for at least `secs` under a Meter.
fn measure<F: FnMut() -> f64 + ?Sized>(pm: Option<&PowerMetrics>, secs: f64, label: &str, engine: &str, f: &mut F) -> (Measurement, usize) {
    std::thread::sleep(Duration::from_millis(600)); // let rails settle between phases
    let meter = Meter::start(pm);
    let t0 = Instant::now();
    let mut iters = 0;
    let mut bytes = 0.0;
    while iters == 0 || t0.elapsed().as_secs_f64() < secs {
        bytes += f();
        iters += 1;
        publish_progress(label, engine, bytes);
    }
    (meter.stop(), iters)
}

struct Entry<'a> {
    engine: String,
    note: String,
    f: Box<dyn FnMut() -> f64 + 'a>,
}

/// Run every entry `o.rounds` times, interleaved (A B C, A B C, …) so slow drift in
/// background load or thermals hits all engines alike; report each entry's median
/// round (by throughput) together with the min–max spread.
fn run_set(o: &Opts, pm: Option<&PowerMetrics>, workload: &str, tag: &str, bytes: f64, mut entries: Vec<Entry>, rows_out: &mut Vec<Row>) {
    let mut res: Vec<Vec<(Measurement, usize)>> = entries.iter().map(|_| Vec::new()).collect();
    for round in 0..o.rounds {
        for (k, e) in entries.iter_mut().enumerate() {
            let eng = if e.engine.starts_with("ane") { "ane" } else { "cpu" };
            let (m, it) = measure(pm, o.secs, tag, eng, &mut e.f);
            eprintln!("[{tag}] round {} {:<32} {:.2} GB/s", round + 1, e.engine, bytes * it as f64 / m.wall_s / 1e9);
            res[k].push((m, it));
        }
    }
    for (e, mut r) in entries.into_iter().zip(res) {
        r.sort_by(|a, b| (a.1 as f64 / a.0.wall_s).partial_cmp(&(b.1 as f64 / b.0.wall_s)).unwrap());
        let g = |x: &(Measurement, usize)| bytes * x.1 as f64 / x.0.wall_s / 1e9;
        let spread = format!("{} rounds: {:.2}–{:.2} GB/s", r.len(), g(&r[0]), g(&r[r.len() - 1]));
        let (m, it) = r[r.len() / 2].clone();
        let note = if e.note.is_empty() { String::new() } else { format!("{}; {spread}", e.note) };
        rows_out.push(Row { workload: workload.to_string(), engine: format!("{} [{spread}]", e.engine), bytes_per_iter: bytes, iters: it, m, note });
    }
}

fn synth_log(n: usize, seed: u64) -> Vec<u8> {
    use rayon::prelude::*;
    let words: [&[u8]; 16] = [
        b"2026-11-18T23:41:07.123Z ", b"INFO ", b"WARN ", b"DEBUG ", b"GET ", b"POST ", b"/api/v1/users/", b"/api/v2/orders ",
        b"status=200 ", b"status=404 ", b"latency_ms=", b"user_id=", b"request_id=", b"cache=hit ", b"cache=miss ", b"\n",
    ];
    let chunk = 1 << 20;
    let mut v: Vec<u8> = (0..n.div_ceil(chunk))
        .into_par_iter()
        .flat_map_iter(|k| {
            let mut r = rng(seed + k as u64 * 7919);
            let mut out = Vec::with_capacity(chunk + 64);
            while out.len() < chunk {
                out.extend_from_slice(words[(r() % 16) as usize]);
                if r() % 3 == 0 {
                    out.extend_from_slice(format!("{} ", r() % 1_000_000).as_bytes());
                }
                if r() % 50_000 == 0 {
                    out.extend_from_slice(b"ERROR connection reset by peer ");
                }
                if r() % 200_000 == 0 {
                    out.extend_from_slice(b"FATAL segfault at 0xdeadbeef ");
                }
            }
            out.truncate(chunk);
            out
        })
        .collect();
    v.truncate(n);
    v
}

fn bench_query(o: &Opts, pm: Option<&PowerMetrics>, rows_out: &mut Vec<Row>) {
    let rows = if o.quick { 10_000_000 } else { 50_000_000 };
    eprintln!("[query] generating {rows} rows x 3 fp32 columns");
    let a = gauss(rows, 1, 100.0);
    let b = gauss(rows, 2, 100.0);
    let c = gauss(rows, 3, 10.0);
    let cols = [a.as_slice(), b.as_slice(), c.as_slice()];
    let filters = [Filter { col: 0, op: CmpOp::Gt, value: 25.0 }, Filter { col: 1, op: CmpOp::Lt, value: -10.0 }];
    let bytes = (rows * 3 * 4) as f64;

    let t = AneTable::from_f32(&cols).expect("ingest");
    eprintln!("[query] ingest into IOSurfaces: {:.1} ms (one-time, not in query timings)", t.ingest_s * 1e3);
    let r0 = t.query(&filters, Some(2), false).expect("ane query");
    eprintln!("[query] plan compile: {:.0} ms (cached afterwards)", r0.compile_ms);
    let (rc, rs, _) = reference_query(&cols, &filters, Some(2));
    let (cc, cs) = cpu_baseline::query(&cols, &filters, Some(2));
    let exact = r0.count == rc;
    let note = format!(
        "ANE count {} == fp16-semantics reference {}: {}; CPU fp32 count {} (diff {}); SUM rel.err vs reference {:.1e}",
        r0.count,
        rc,
        exact,
        cc,
        cc as i64 - rc as i64,
        (r0.sum.unwrap() - rs).abs() / rs.abs().max(1.0)
    );
    eprintln!("[query] {note}");
    assert!(exact, "ANE query disagrees with reference");
    let _ = cs;

    let entries = vec![
        Entry { engine: "cpu-rayon".into(), note: String::new(), f: Box::new(|| { std::hint::black_box(cpu_baseline::query(&cols, &filters, Some(2))); bytes }) },
        Entry { engine: "ane".into(), note, f: Box::new(|| { std::hint::black_box(t.query(&filters, Some(2), false).unwrap()); bytes }) },
    ];
    run_set(o, pm, &format!("query {}M rows", rows / 1_000_000), "query", bytes, entries, rows_out);
}

fn bench_grep(o: &Opts, pm: Option<&PowerMetrics>, rows_out: &mut Vec<Row>, scenario: &str, pats: &[&[u8]]) {
    let n = if o.quick { 128 << 20 } else { 1 << 30 };
    eprintln!("[grep] generating {} MiB synthetic log", n >> 20);
    let text = synth_log(n, 42);
    let pats = pats.to_vec();
    let bytes = n as f64;
    let label = format!("grep {} MiB, {} patterns, {scenario}", n >> 20, pats.len());
    let reference = cpu_baseline::grep(&text, &pats);
    let mut greps = Vec::new();
    for (proj, segs) in [(1usize, 1024usize), (2, 256)] {
        let g = AneGrep::new(&pats, proj, segs).expect("grep compile");
        eprintln!("[grep] proj={proj}: compile {:.0} ms ({} patterns, prefix {}, {} x 16 KiB segments/dispatch)", g.compile_ms, pats.len(), g.prefix_len(), g.segments_per_dispatch());
        let (m_ane, st) = g.scan(&text).unwrap();
        let exact = m_ane == reference;
        let note = format!(
            "projections={proj}: {} matches, identical to Aho-Corasick: {}; flagged {:.3}% of 64 B blocks; stage times in one scan: copy {:.0} ms, ANE {:.0} ms, parallel CPU confirm {:.0} ms (copy+ANE overlap the confirm)",
            m_ane.len(), exact,
            100.0 * st.flagged_blocks as f64 / st.total_blocks as f64,
            st.copy_s * 1e3, st.ane_s * 1e3, st.confirm_s * 1e3
        );
        eprintln!("[grep] {note}");
        assert!(exact, "ANE grep disagrees with Aho-Corasick");
        greps.push((proj, g, note));
    }
    let mut entries = vec![
        Entry { engine: "cpu-aho-corasick (match list)".into(), note: String::new(), f: Box::new(|| { std::hint::black_box(cpu_baseline::grep(&text, &pats)); bytes }) },
        Entry { engine: "cpu-aho-corasick (count only)".into(), note: String::new(), f: Box::new(|| { std::hint::black_box(cpu_baseline::grep_count(&text, &pats)); bytes }) },
    ];
    for (proj, g, note) in &greps {
        let text = &text;
        entries.push(Entry { engine: format!("ane (p={proj})"), note: note.clone(), f: Box::new(move || { std::hint::black_box(g.scan(text).unwrap()); bytes }) });
    }
    run_set(o, pm, &label, "grep", bytes, entries, rows_out);
}

fn bench_dsp(o: &Opts, pm: Option<&PowerMetrics>, rows_out: &mut Vec<Row>) {
    let ch = 32;
    let taps = 4096;
    let n = if o.quick { 48_000 * 5 } else { 48_000 * 30 };
    eprintln!("[dsp] {ch} ch x {taps} taps, {} s of 48 kHz audio per channel", n / 48_000);
    let irs: Vec<Vec<f32>> = (0..ch)
        .map(|c| gauss(taps, 10 + c as u64, 0.05).iter().enumerate().map(|(i, v)| v * (-(i as f32) / 900.0).exp()).collect())
        .collect();
    let x: Vec<Vec<f32>> = (0..ch).map(|c| gauss(n, 100 + c as u64, 0.3)).collect();
    let ir: Vec<&[f32]> = irs.iter().map(|v| v.as_slice()).collect();
    let xs: Vec<&[f32]> = x.iter().map(|v| v.as_slice()).collect();
    let f = AneFir::new(&ir, None, 1024).expect("fir compile");
    eprintln!("[dsp] compile: {:.0} ms, block {}, weights {:.0} MB", f.compile_ms, f.block, f.weight_bytes as f64 / 1e6);
    let (y, _) = f.process(&xs).unwrap();
    let r = cpu_baseline::fir_fft(&xs, &ir);
    let (mut e, mut s) = (0f64, 0f64);
    for c in 0..ch {
        for i in 0..n {
            e += ((y[c][i] - r[c][i]) as f64).powi(2);
            s += (r[c][i] as f64).powi(2);
        }
    }
    let snr = 10.0 * (s / e).log10();
    let note = format!("SNR vs fp32 FFT reference {snr:.1} dB; realtime factor computed from latency");
    eprintln!("[dsp] {note}");
    assert!(snr > 50.0);
    let bytes = (ch * n * 4) as f64;
    let label = format!("FIR {ch}ch x {taps} taps, {} s audio", n / 48_000);
    let entries = vec![
        Entry { engine: "cpu-vDSP-direct".into(), note: String::new(), f: Box::new(|| { std::hint::black_box(cpu_baseline::fir_direct(&xs, &ir)); bytes }) },
        Entry { engine: "cpu-FFT".into(), note: String::new(), f: Box::new(|| { std::hint::black_box(cpu_baseline::fir_fft(&xs, &ir)); bytes }) },
        Entry { engine: "ane".into(), note, f: Box::new(|| { std::hint::black_box(f.process(&xs).unwrap()); bytes }) },
    ];
    run_set(o, pm, &label, "dsp", bytes, entries, rows_out);
}

fn fmt_opt(v: Option<f64>, prec: usize) -> String {
    v.map(|x| format!("{x:.prec$}")).unwrap_or_else(|| "n/a".into())
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut o = Opts { rounds: 3, quick: false, only: None, secs: 4.0, out: "bench_results".into() };
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--quick" => o.quick = true,
            "--rounds" => {
                o.rounds = args.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(3).max(1);
                i += 1;
            }
            "--only" => {
                o.only = args.get(i + 1).cloned();
                i += 1;
            }
            "--secs" => {
                o.secs = args.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(4.0);
                i += 1;
            }
            "--out" => {
                o.out = args.get(i + 1).cloned().unwrap_or(o.out);
                i += 1;
            }
            _ => {}
        }
        i += 1;
    }
    ane_unleashed::available().expect("AppleNeuralEngine.framework SPI unavailable");
    let pm = PowerMetrics::start(250);
    eprintln!("power rails: {}", if pm.is_some() { "powermetrics (root)" } else { "UNAVAILABLE (not root) — recording unprivileged metrics only" });
    if pm.is_some() {
        std::thread::sleep(Duration::from_millis(800));
    }
    let idle = {
        let m = Meter::start(pm.as_ref());
        std::thread::sleep(Duration::from_secs_f64(o.secs.min(5.0)));
        m.stop()
    };
    let idle_rails = idle.rails;
    let load1: f64 = std::process::Command::new("sysctl").args(["-n", "vm.loadavg"]).output().ok()
        .and_then(|o| String::from_utf8_lossy(&o.stdout).split_whitespace().nth(1).and_then(|v| v.replace(',', ".").parse().ok()))
        .unwrap_or(f64::NAN);
    let force = std::env::args().any(|a| a == "--force");
    if let Some(r) = &idle_rails {
        eprintln!("idle baseline: CPU {:.2} W, GPU {:.2} W, ANE {:.3} W; load average (1 min) {load1:.1}", r.cpu_mw / 1e3, r.gpu_mw / 1e3, r.ane_mw / 1e3);
        // Rails "above idle" are only meaningful on a quiet machine.
        if (r.cpu_mw > 6000.0 || r.gpu_mw > 3000.0 || load1 > 6.0) && !force {
            eprintln!("ABORT: machine is not quiet (idle CPU rail > 6 W, idle GPU rail > 3 W, or load > 6). Close other workloads or pass --force.");
            std::process::exit(2);
        }
    }
    let mut rows = Vec::new();
    let want = |w: &str| o.only.as_deref().map(|x| x == w).unwrap_or(true);
    if want("query") {
        bench_query(&o, pm.as_ref(), &mut rows);
    }
    if want("grep") {
        // sparse: rare incident strings (the usual "search the logs for X" case)
        let sparse: Vec<&[u8]> = vec![
            b"FATAL", b"segfault", b"connection reset", b"0xdeadbeef", b"panic:", b"OOM killer", b"permission denied", b"status=500",
            b"status=503", b"user_id=4242 ", b"request_id=1337 ", b"latency_ms=99999", b"Traceback", b"NullPointerException", b"disk full", b"certificate expired",
        ];
        // dense: includes tokens that occur every few hundred bytes
        let dense: Vec<&[u8]> = vec![
            b"ERROR", b"FATAL", b"segfault", b"connection reset", b"status=500", b"user_id=4242 ", b"/api/v2/orders", b"cache=miss",
            b"timeout", b"0xdeadbeef", b"request_id=1337 ", b"WARN /api/v1", b"panic", b"OOM", b"denied", b"latency_ms=999 ",
        ];
        bench_grep(&o, pm.as_ref(), &mut rows, "sparse matches", &sparse);
        bench_grep(&o, pm.as_ref(), &mut rows, "dense matches", &dense);
    }
    if want("dsp") {
        bench_dsp(&o, pm.as_ref(), &mut rows);
    }

    // ---------------------------------------------------------------- report
    let mut md = String::new();
    let _ = writeln!(md, "# ANE-Unleashed benchmark results\n");
    let _ = writeln!(md, "Machine: Apple M3 Max (16-core ANE, 36 GB), macOS {}. Power rails: {}.\n",
        std::process::Command::new("sw_vers").arg("-productVersion").output().map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string()).unwrap_or_default(),
        if pm.is_some() { "`powermetrics` cpu_power/gpu_power/ane_power, 250 ms samples" } else { "not available (run as root)" });
    if let Some(r) = &idle_rails {
        let _ = writeln!(md, "Idle baseline (measured before the workloads): CPU {:.2} W, GPU {:.2} W, ANE {:.3} W; 1-min load average {load1:.1}.\n", r.cpu_mw / 1e3, r.gpu_mw / 1e3, r.ane_mw / 1e3);
    }
    let _ = writeln!(md, "| workload | engine | latency/iter | throughput | process CPU time / wall | ANE domain powered | CPU W | GPU W | ANE W | workload W (above idle) | J per GB | GB/s per W |");
    let _ = writeln!(md, "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|");
    let mut json = String::from("[\n");
    for (k, r) in rows.iter().enumerate() {
        let (pw, basis) = r.power_w(&idle_rails);
        let rails = r.m.rails;
        let _ = writeln!(
            md,
            "| {} | {} | {:.1} ms | {:.2} GB/s | {:.1} cores | {:.0}% | {} | {} | {} | {} | {} | {} |",
            r.workload,
            r.engine,
            r.lat_ms(),
            r.gbps(),
            r.m.proc_cpu_s / r.m.wall_s,
            r.m.ane_on_frac * 100.0,
            fmt_opt(rails.map(|x| x.cpu_mw / 1e3), 2),
            fmt_opt(rails.map(|x| x.gpu_mw / 1e3).or(r.m.gpu_w_ioreport), 2),
            fmt_opt(rails.map(|x| x.ane_mw / 1e3), 3),
            fmt_opt(pw, 2),
            fmt_opt(pw.map(|w| w / r.gbps()), 2),
            fmt_opt(pw.map(|w| r.gbps() / w.max(1e-3)), 3),
        );
        let _ = write!(
            json,
            "  {{\"workload\": {:?}, \"engine\": {:?}, \"iters\": {}, \"wall_s\": {:.6}, \"latency_ms\": {:.4}, \"gb_per_s\": {:.4}, \"proc_cpu_j\": {:.4}, \"proc_cpu_s\": {:.4}, \"ane_on_frac\": {:.4}, \"gpu_w_ioreport\": {}, \"rail_cpu_w\": {}, \"rail_gpu_w\": {}, \"rail_ane_w\": {}, \"workload_w\": {}, \"power_basis\": {:?}, \"note\": {:?}}}{}\n",
            r.workload, r.engine, r.iters, r.m.wall_s, r.lat_ms(), r.gbps(), r.m.proc_cpu_j, r.m.proc_cpu_s, r.m.ane_on_frac,
            fmt_opt(r.m.gpu_w_ioreport, 4).replace("n/a", "null"),
            fmt_opt(rails.map(|x| x.cpu_mw / 1e3), 4).replace("n/a", "null"),
            fmt_opt(rails.map(|x| x.gpu_mw / 1e3), 4).replace("n/a", "null"),
            fmt_opt(rails.map(|x| x.ane_mw / 1e3), 4).replace("n/a", "null"),
            fmt_opt(pw, 4).replace("n/a", "null"),
            basis, r.note,
            if k + 1 < rows.len() { "," } else { "" }
        );
    }
    json.push_str("]\n");
    let _ = writeln!(md, "\nNotes:\n");
    for r in rows.iter().filter(|r| !r.note.is_empty()) {
        let _ = writeln!(md, "- **{}**: {}", r.workload, r.note);
    }
    let _ = writeln!(md, "- \"process CPU time / wall\" = cores kept busy by this process; \"ANE domain powered\" = share of 2 ms polls with the H11ANE power domain on; it stays on while any ANE program is loaded, so it is a liveness signal, not utilisation — use the ANE rail watts.");
    let _ = std::fs::create_dir_all(&o.out);
    let _ = std::fs::write(format!("{}/results.md", o.out), &md);
    let _ = std::fs::write(format!("{}/results.json", o.out), &json);
    println!("{md}");
}
