//! ane-top — live CPU / GPU / ANE power and ANE-Unleashed throughput.
//!
//! Without root: ANE power-gate state (sampled every 5 ms), GPU watts (IOReport),
//! per-process CPU watts for `--pid` (rusage energy). With root (or a NOPASSWD
//! sudoers rule for /usr/bin/powermetrics): CPU / GPU / ANE rail watts.
//! Workloads built on ane_unleashed publish bytes processed to
//! `$TMPDIR/ane-unleashed.progress`, which ane-top turns into GB/s and GB/s/W.
use ane_unleashed::telemetry::{ane_power_state, proc_energy, read_energy_counters, PowerMetrics};
use std::io::Write;
use std::time::{Duration, Instant};

fn progress_path() -> std::path::PathBuf {
    std::env::temp_dir().join("ane-unleashed.progress")
}

/// (label, engine, bytes processed) written by ane_unleashed::telemetry users
fn read_progress() -> Option<(String, String, f64)> {
    let s = std::fs::read_to_string(progress_path()).ok()?;
    let mut it = s.split('\t');
    Some((it.next()?.to_string(), it.next()?.to_string(), it.next()?.trim().parse().ok()?))
}

fn bar(frac: f64, width: usize) -> String {
    let n = ((frac.clamp(0.0, 1.0)) * width as f64).round() as usize;
    format!("{}{}", "█".repeat(n), "·".repeat(width - n))
}

fn main() {
    let mut pid: Option<i32> = None;
    let mut interval_ms = 1000u64;
    let mut once = false;
    let args: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--pid" => {
                pid = args.get(i + 1).and_then(|s| s.parse().ok());
                i += 1;
            }
            "-i" | "--interval" => {
                interval_ms = args.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(1000);
                i += 1;
            }
            "--once" => once = true,
            "-h" | "--help" => {
                println!("usage: ane-top [--pid PID] [-i MS] [--once]\n  rail watts need root or a NOPASSWD sudoers rule for /usr/bin/powermetrics");
                return;
            }
            _ => {}
        }
        i += 1;
    }
    let pm = PowerMetrics::start(interval_ms.max(200));
    let mut last_io = read_energy_counters();
    let mut last_proc = pid.and_then(proc_energy);
    let mut last_prog = read_progress();
    let mut t_last = Instant::now();
    let mut peak = 0.0f64;
    loop {
        // sample ANE power state at 5 ms for one interval
        let t0 = Instant::now();
        let (mut on, mut tot) = (0u32, 0u32);
        while t0.elapsed() < Duration::from_millis(interval_ms) {
            on += (ane_power_state() == 1) as u32;
            tot += 1;
            std::thread::sleep(Duration::from_millis(5));
        }
        let dt = t_last.elapsed().as_secs_f64();
        t_last = Instant::now();
        let io = read_energy_counters();
        let gpu_w = match (last_io, io) {
            (Some(a), Some(b)) => (b.gpu_j - a.gpu_j) / dt,
            _ => f64::NAN,
        };
        last_io = io;
        let proc_w = match (last_proc, pid.and_then(proc_energy)) {
            (Some(a), Some(b)) => {
                last_proc = Some(b);
                Some(((b.0 - a.0) / dt, (b.1 - a.1) / dt))
            }
            _ => None,
        };
        let prog = read_progress();
        let gbps = match (&last_prog, &prog) {
            (Some(a), Some(b)) if a.0 == b.0 && b.2 >= a.2 => Some((b.2 - a.2) / dt / 1e9),
            _ => None,
        };
        last_prog = prog.clone();
        let rails = pm.as_ref().and_then(|p| p.last());
        let ane_frac = on as f64 / tot.max(1) as f64;

        print!("\x1b[2J\x1b[H");
        println!("ane-top — Apple Silicon power ({} ms){}", interval_ms, if pm.is_none() { "   [no root: rail watts unavailable]" } else { "" });
        println!("──────────────────────────────────────────────────────────────");
        println!("ANE power-gate on   {:>5.1}%  {}", ane_frac * 100.0, bar(ane_frac, 30));
        if let Some(r) = rails {
            peak = peak.max(r.ane_mw);
            println!("CPU rail            {:>7.2} W  {}", r.cpu_mw / 1e3, bar(r.cpu_mw / 40e3, 30));
            println!("GPU rail            {:>7.2} W  {}", r.gpu_mw / 1e3, bar(r.gpu_mw / 40e3, 30));
            println!("ANE rail            {:>7.2} W  {}   (peak {:.2} W)", r.ane_mw / 1e3, bar(r.ane_mw / 8e3, 30), peak / 1e3);
            println!("package (combined)  {:>7.2} W", r.combined_mw / 1e3);
        } else {
            println!("GPU (IOReport)      {:>7.2} W", gpu_w);
            println!("CPU / ANE rails     n/a  (run with sudo, or add a NOPASSWD rule for /usr/bin/powermetrics)");
        }
        if let Some((w, util)) = proc_w {
            println!("pid {:<6} CPU      {:>7.2} W   {:>5.0}% of one core", pid.unwrap(), w, util * 100.0);
        }
        if let (Some(g), Some(p)) = (gbps, &prog) {
            println!("──────────────────────────────────────────────────────────────");
            println!("workload  {:<10} {:<6}  {:>7.2} GB/s", p.0, p.1, g);
            if let Some(r) = rails {
                let w = if p.1 == "ane" { r.ane_mw + r.cpu_mw } else { r.cpu_mw } / 1e3;
                println!("          {:>7.3} GB/s per W ({} rails)", g / w.max(1e-3), if p.1 == "ane" { "ANE+CPU" } else { "CPU" });
            }
        }
        let _ = std::io::stdout().flush();
        if once {
            break;
        }
    }
}
