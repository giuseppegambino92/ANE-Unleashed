//! Power / activity telemetry.
//!
//! What can be measured on macOS 27.2 (M3 Max), by privilege level:
//!
//! | signal                                   | source                          | root? |
//! |------------------------------------------|---------------------------------|-------|
//! | ANE power-gate state (0 off / 1 on)      | IORegistry `H11ANEIn` IOPower   | no    |
//! | per-process CPU energy (J)               | `proc_pid_rusage` v6 `ri_energy_nj` | no (own pids) |
//! | GPU energy (J)                           | IOReport "Energy Model"         | no    |
//! | CPU / ANE / DRAM energy                  | IOReport "Energy Model" (PMGR)  | counters frozen when unprivileged |
//! | CPU / GPU / ANE rail power (mW)          | `powermetrics --samplers cpu_power,gpu_power,ane_power` | yes |
//!
//! `Meter` combines everything available and records which sources were live.
use crate::ffi;
use std::io::{BufRead, BufReader, Read};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

#[derive(Clone, Copy, Debug, Default)]
pub struct EnergyCounters {
    pub cpu_j: f64,
    pub gpu_j: f64,
    pub ane_j: f64,
    pub dram_j: f64,
}

pub fn read_energy_counters() -> Option<EnergyCounters> {
    let mut s = ffi::ane_energy_sample::default();
    if unsafe { ffi::ane_energy_read(&mut s) } != 0 {
        return None;
    }
    Some(EnergyCounters { cpu_j: s.cpu_j, gpu_j: s.gpu_j, ane_j: s.ane_j, dram_j: s.dram_j })
}

/// (CPU energy joules, CPU seconds) for a process (0 = self).
pub fn proc_energy(pid: i32) -> Option<(f64, f64)> {
    let (mut e, mut c) = (0.0, 0.0);
    if unsafe { ffi::ane_proc_energy(pid, &mut e, &mut c) } == 0 {
        Some((e, c))
    } else {
        None
    }
}

pub fn ane_power_state() -> i32 {
    unsafe { ffi::ane_power_state() }
}

/// One powermetrics sample (rail power in milliwatts).
#[derive(Clone, Copy, Debug, Default)]
pub struct PmSample {
    pub at: Option<Instant>,
    pub cpu_mw: f64,
    pub gpu_mw: f64,
    pub ane_mw: f64,
    pub combined_mw: f64,
}

/// Background `powermetrics` reader. Requires root: either run the binary with
/// sudo, or have a passwordless sudo rule for /usr/bin/powermetrics.
pub struct PowerMetrics {
    child: Child,
    pub samples: Arc<Mutex<Vec<PmSample>>>,
    pub interval_ms: u64,
}

fn plist_real(doc: &str, key: &str) -> Option<f64> {
    let k = format!("<key>{key}</key>");
    let i = doc.find(&k)? + k.len();
    let rest = &doc[i..];
    let open = rest.find('>')? + 1;
    let close = rest[open..].find('<')? + open;
    rest[open..close].trim().parse().ok()
}

impl PowerMetrics {
    pub fn can_run() -> bool {
        if unsafe { libc::geteuid() } == 0 {
            return true;
        }
        Command::new("sudo")
            .args(["-n", "/usr/bin/powermetrics", "-h"])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .map(|s| s.success())
            .unwrap_or(false)
    }

    pub fn start(interval_ms: u64) -> Option<PowerMetrics> {
        if !Self::can_run() {
            return None;
        }
        let args = [
            "-i".to_string(),
            interval_ms.to_string(),
            "--samplers".into(),
            "cpu_power,gpu_power,ane_power".into(),
            "-f".into(),
            "plist".into(),
        ];
        let mut cmd = if unsafe { libc::geteuid() } == 0 {
            let mut c = Command::new("/usr/bin/powermetrics");
            c.args(&args);
            c
        } else {
            let mut c = Command::new("sudo");
            c.arg("-n").arg("/usr/bin/powermetrics").args(&args);
            c
        };
        let mut child = cmd.stdout(Stdio::piped()).stderr(Stdio::null()).spawn().ok()?;
        let stdout = child.stdout.take()?;
        let samples = Arc::new(Mutex::new(Vec::new()));
        let s2 = samples.clone();
        std::thread::spawn(move || {
            // powermetrics -f plist emits NUL-separated plist documents
            let mut r = BufReader::new(stdout);
            let mut buf = Vec::new();
            while let Ok(n) = r.read_until(0, &mut buf) {
                if n == 0 {
                    break;
                }
                let doc = String::from_utf8_lossy(&buf);
                // power values live in the "processor" dict, in mW
                let s = PmSample {
                    at: Some(Instant::now()),
                    cpu_mw: plist_real(&doc, "cpu_power").unwrap_or(f64::NAN),
                    gpu_mw: plist_real(&doc, "gpu_power").unwrap_or(f64::NAN),
                    ane_mw: plist_real(&doc, "ane_power").unwrap_or(f64::NAN),
                    combined_mw: plist_real(&doc, "combined_power").unwrap_or(f64::NAN),
                };
                s2.lock().unwrap().push(s);
                buf.clear();
            }
        });
        Some(PowerMetrics { child, samples, interval_ms })
    }

    /// Mean rail power over samples that completed in (from + interval, to].
    pub fn mean_between(&self, from: Instant, to: Instant) -> Option<PmSample> {
        let s = self.samples.lock().unwrap();
        let lo = from + Duration::from_millis(self.interval_ms);
        let v: Vec<&PmSample> = s.iter().filter(|x| x.at.map(|t| t > lo && t <= to).unwrap_or(false)).collect();
        if v.is_empty() {
            return None;
        }
        let n = v.len() as f64;
        Some(PmSample {
            at: None,
            cpu_mw: v.iter().map(|x| x.cpu_mw).sum::<f64>() / n,
            gpu_mw: v.iter().map(|x| x.gpu_mw).sum::<f64>() / n,
            ane_mw: v.iter().map(|x| x.ane_mw).sum::<f64>() / n,
            combined_mw: v.iter().map(|x| x.combined_mw).sum::<f64>() / n,
        })
    }

    pub fn last(&self) -> Option<PmSample> {
        self.samples.lock().unwrap().last().copied()
    }
}

impl Drop for PowerMetrics {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let mut sink = Vec::new();
        if let Some(mut o) = self.child.stdout.take() {
            let _ = o.read_to_end(&mut sink);
        }
        let _ = self.child.wait();
    }
}

/// Polls the H11ANE power state on a background thread.
pub struct AneStatePoller {
    stop: Arc<AtomicBool>,
    on: Arc<AtomicU64>,
    total: Arc<AtomicU64>,
    h: Option<std::thread::JoinHandle<()>>,
}

impl AneStatePoller {
    pub fn start(period: Duration) -> Self {
        let stop = Arc::new(AtomicBool::new(false));
        let on = Arc::new(AtomicU64::new(0));
        let total = Arc::new(AtomicU64::new(0));
        let (s, o, t) = (stop.clone(), on.clone(), total.clone());
        let h = std::thread::spawn(move || {
            while !s.load(Ordering::Relaxed) {
                if ane_power_state() == 1 {
                    o.fetch_add(1, Ordering::Relaxed);
                }
                t.fetch_add(1, Ordering::Relaxed);
                std::thread::sleep(period);
            }
        });
        Self { stop, on, total, h: Some(h) }
    }
    /// Fraction of polls that saw the ANE powered on.
    pub fn finish(mut self) -> f64 {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(h) = self.h.take() {
            let _ = h.join();
        }
        let t = self.total.load(Ordering::Relaxed).max(1);
        self.on.load(Ordering::Relaxed) as f64 / t as f64
    }
}

/// Result of a measured region.
#[derive(Clone, Debug, Default)]
pub struct Measurement {
    pub wall_s: f64,
    /// CPU energy charged to this process (rusage), joules
    pub proc_cpu_j: f64,
    /// CPU time consumed by this process (user+sys), seconds
    pub proc_cpu_s: f64,
    /// fraction of 2 ms polls with the ANE powered on
    pub ane_on_frac: f64,
    /// IOReport GPU energy -> average watts (system-wide)
    pub gpu_w_ioreport: Option<f64>,
    /// powermetrics rail averages (root only)
    pub rails: Option<PmSample>,
}

pub struct Meter<'a> {
    t0: Instant,
    e0: (f64, f64),
    io0: Option<EnergyCounters>,
    poller: AneStatePoller,
    pm: Option<&'a PowerMetrics>,
}

impl<'a> Meter<'a> {
    pub fn start(pm: Option<&'a PowerMetrics>) -> Self {
        let poller = AneStatePoller::start(Duration::from_millis(2));
        Self { t0: Instant::now(), e0: proc_energy(0).unwrap_or_default(), io0: read_energy_counters(), poller, pm }
    }
    pub fn stop(self) -> Measurement {
        let t1 = Instant::now();
        let e1 = proc_energy(0).unwrap_or_default();
        let io1 = read_energy_counters();
        let wall = (t1 - self.t0).as_secs_f64();
        let gpu = match (self.io0, io1) {
            (Some(a), Some(b)) if b.gpu_j > a.gpu_j => Some((b.gpu_j - a.gpu_j) / wall),
            _ => None,
        };
        Measurement {
            wall_s: wall,
            proc_cpu_j: e1.0 - self.e0.0,
            proc_cpu_s: e1.1 - self.e0.1,
            ane_on_frac: self.poller.finish(),
            gpu_w_ioreport: gpu,
            rails: self.pm.and_then(|p| p.mean_between(self.t0, t1)),
        }
    }
}

/// Publish cumulative bytes processed so `ane-top` can show GB/s and GB/s/W.
/// `engine` is "ane" or "cpu". Cheap enough to call once per dispatch.
pub fn publish_progress(label: &str, engine: &str, bytes: f64) {
    let p = std::env::temp_dir().join("ane-unleashed.progress");
    let tmp = std::env::temp_dir().join(format!("ane-unleashed.progress.{}", std::process::id()));
    if std::fs::write(&tmp, format!("{label}\t{engine}\t{bytes}\n")).is_ok() {
        let _ = std::fs::rename(tmp, p);
    }
}
