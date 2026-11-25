# ANE-Unleashed

**English** · [Italiano](README.it.md)

**Run SQL-style column filters, multi-pattern grep and long FIR filters on the
Apple Neural Engine.** It's a Rust + Objective-C runtime that writes MIL programs
in memory and runs them on the ANE through Apple's private `_ANEInMemoryModel`
path. CoreML and `.mlmodelc` bundles are never involved.

**Measured result (M3 Max, `powermetrics` rails, median of 3 interleaved rounds):**
a two-predicate filter + COUNT/SUM over **50 M rows runs at 31.7 GB/s on 1.8 W of
ANE power**, with the CPU idle (0.1 cores). The 12-core CPU baseline reaches
97 GB/s but needs 24 W, so the ANE is **4.4× more energy-efficient** (0.06 vs
0.25 J/GB) at **3.1× lower speed**. Grep and FIR are mixed: see [Results](#results).
In wall-clock speed the CPU won every workload except direct-form FIR.

> ⚠️ **Private, undocumented Apple SPI.** Developed and measured on one machine
> (MacBook Pro M3 Max, 16-core ANE v176, macOS 27.2.1). It can break with any OS
> update and is not App Store-safe. Read [HARDWARE_NOTES.md](HARDWARE_NOTES.md)
> before changing graph shapes: one bad shape can wedge the system-wide ANE
> compiler for hours.

## How it works

```
 your Rust / Python code
        │  columns, text, audio (f32 / u8)
        ▼
 ┌──────────────────────────┐   MIL text + weight.bin, generated in RAM
 │ mil_codegen.rs           │   (program(1.3) / func main<ios18>)
 │ query.rs grep.rs dsp.rs  │
 └────────────┬─────────────┘
              │ hash-keyed compile cache (compile once, then ~0.15 ms per dispatch)
              ▼
 ┌──────────────────────────┐   _ANEInMemoryModelDescriptor
 │ native/ane_bridge.m      │   _ANEInMemoryModel compile/load/evaluate
 │ (C ABI, ObjC, ARC)       │   _ANERequest + _ANEIOSurfaceObject
 └────────────┬─────────────┘
              │ XPC: aned / ANECompilerService (compile, once)
              │ IOSurfaces (64-byte aligned rows), zero-copy to the ANE
              ▼
 ┌──────────────────────────┐
 │ H11ANE, 16-core ANE      │   fp16 datapath, exact wide accumulators
 └──────────────────────────┘
```

The program never reaches CoreML, but it isn't quite zero-disk. The compiler
service insists on reading a staged copy of the program from
`$TMPDIR/<hash>/`. The bridge writes it, compiles, and deletes it immediately.

## The three engines

### ANE-Query: `SELECT COUNT(*), SUM(c) WHERE a > X AND b < Y`

```
 col a ─cast fp16─▶ a > X ─┐
                           ├─ AND ─▶ mask ─┬─ reduce_sum/2048 rows ─▶ COUNT partials (exact)
 col b ─cast fp16─▶ b < Y ─┘               └─ × c ─ reduce_sum ────▶ SUM partials (fp16)
 tensors [1, 1024, 8, 2048] = 16.7 M rows per dispatch, ingested once into IOSurfaces
```

* **COUNT and the selection mask are exact.** They're verified bit-for-bit against a CPU
  model of the ANE semantics. Predicates act on values rounded to fp16, and the ANE
  rounds **ties away from zero**, not to even.
* **SUM is approximate**, with relative error ≈ 1e-4–1e-3 because the partials are fp16.
* fp32 constants are rewritten to equivalent fp16 constants, including ±∞ edge cases.

### ANE-Grep: exact multi-pattern scan, with the ANE as prefilter

```
 text (u8) as [1, C segments, 1, 16384]
   └─ conv: per pattern j, h_j(i) = Σ_k w_jk·(x[i+k] − p_jk)   (integer weights, fused bias)
        └─ h == 0 ─▶ grouped-conv count ─▶ ≥ m ─▶ max-pool 64 ─▶ 1 flag per 64-byte block
 CPU (rayon, pipelined with the next ANE dispatch): confirm full patterns inside flagged blocks only
```

The conv accumulates exactly and rounds once. Every true occurrence therefore
gives `h == 0` exactly, so there are **no false negatives**, and the CPU's
confirmation step throws out every false positive. The output equals Aho-Corasick's
match set, and the tests and benchmark check this on every run. The ANE looks at the
first 15 bytes of each pattern (the conv kernel can't be wider); the CPU checks
whatever is longer.

### ANE-DSP: multi-channel long FIR / impulse-response convolution

A 4096-tap filter won't fit in an ANE conv kernel (15 wide at most). It becomes a
block-Toeplitz convolution over 512-sample frames instead:

```
 x [1, C, M, 512] ─▶ transpose ─▶ [1, C·512, 1, M] ─▶ grouped conv (512→512 per channel,
 9 frame taps, causal) ─▶ transpose back ─▶ y        (overlap-save across dispatches)
```

32 channels × 4096 taps comes to 151 MB of fp16 weights and compiles once in about 5 s.
Output SNR against an fp32 FFT reference is 56–59 dB.

## Results

Raw output: [`bench_results/results.md`](bench_results/results.md),
[`results.json`](bench_results/results.json), [`run.log`](bench_results/run.log).
M3 Max on AC power, idle baseline CPU 1.48 W / GPU 0.28 W / ANE 0 W. Each engine
ran for 5 s per round over 3 interleaved rounds; the median round is shown.
"Workload W" is the sum of the CPU, GPU and ANE rails above idle. Every ANE result was
checked against its reference in the same run.

| workload | engine | throughput | cores busy | ANE rail | workload W | J per GB | vs CPU best (J/GB) |
|---|---|---:|---:|---:|---:|---:|---:|
| **query** 50 M rows × 3 f32 cols | CPU, rayon, 12 cores | 97.3 GB/s | 11.0 | 0 W | 24.3 W | 0.25 | |
| | **ANE** | 31.7 GB/s | 0.1 | 1.80 W | **1.8 W** | **0.06** | **4.4× less energy** |
| **grep** 1 GiB log, 16 patterns, *sparse* matches | CPU, Aho-Corasick | 8.3 GB/s | 12.8 | 0 W | 32.8 W | 3.9 | |
| | ANE, 2 projections | 0.69 GB/s | 1.1 | 2.12 W | 2.1 W | 3.1 | 1.3× less energy, 12× slower |
| | ANE, 1 projection | 1.48 GB/s | 4.7 | 2.45 W | 9.4 W | 6.4 | 1.6× more energy |
| **grep** 1 GiB log, 16 patterns, *dense* matches (11.6 M hits) | CPU, Aho-Corasick count | 6.3 GB/s | 11.5 | 0 W | 29.8 W | 4.8 | |
| | ANE, 1 projection | 1.37 GB/s | 5.1 | 2.16 W | 21.7 W | 15.8 | 3.3× more energy |
| **FIR** 32 ch × 4096 taps, 30 s @ 48 kHz | CPU, `vDSP_conv` direct | 0.29 GB/s | 9.9 | 0 W | 30.7 W | 104 | |
| | CPU, FFT overlap-save | 7.12 GB/s | 10.2 | 0 W | 32.5 W | 4.6 | |
| | ANE | 2.91 GB/s | 1.5 | 3.82 W | 12.0 W | 4.1 | 1.1× less energy than FFT; 25× less than direct |

What the measurements say:

* **The ANE wins on bulk predicate scans.** Throughput is bounded by memory
  bandwidth on both sides, and the ANE moves the same bytes at a fraction of the power.
  The CPU rail actually dropped *below* idle during the ANE query.
* **Grep is a poor fit.** ANE cost grows with patterns × projections, and the
  prefilter needs CPU confirmation. Two projections keep false positives low enough
  (8.7% of 64-byte blocks flagged on the sparse log) that the CPU stays near idle and
  energy per GB beats Aho-Corasick by 1.3×, at 1/12 of the speed. With dense matches,
  confirmation dominates and the CPU wins outright.
* **FIR beats direct convolution easily (10× faster, 25× less energy) but only
  narrowly beats an FFT.** An FFT does far less arithmetic. The ANE path runs at 474×
  real time and still spends about 1.5 cores on fp32 copies in and out.
* **Every speed record went to the CPU** except direct-form FIR. The ANE's case is energy,
  plus keeping the CPU free for other work.

## Quick start

```sh
cargo build --release
cargo test --release -- --test-threads=1      # real ANE runs, checked against exact CPU references

# live power monitor: CPU / GPU / ANE rails with root, ANE power-gate state without
./target/release/ane-top

# benchmarks. Rail watts need root or a NOPASSWD sudoers rule:
echo "$USER ALL=(root) NOPASSWD: /usr/bin/powermetrics" | sudo tee /etc/sudoers.d/powermetrics
sudo chmod 440 /etc/sudoers.d/powermetrics
scripts/run_bench.sh            # aborts if the machine isn't idle enough to measure
```

### Rust

```rust
use ane_unleashed::query::{AneTable, CmpOp, Filter};
let t = AneTable::from_f32(&[&a, &b, &c])?;                  // ingest once
let r = t.query(&[Filter { col: 0, op: CmpOp::Gt, value: 25.0 },
                  Filter { col: 1, op: CmpOp::Lt, value: -10.0 }], Some(2), false)?;
println!("{} rows, sum ≈ {}", r.count, r.sum.unwrap());

let g = ane_unleashed::grep::AneGrep::new(&[b"ERROR".as_slice(), b"timeout"], 2, 256)?;
let (matches, stats) = g.scan(&log_bytes)?;                  // exact, sorted (pos, pattern)

let fir = ane_unleashed::dsp::AneFir::new(&impulse_responses, None, 1024)?;
let (y, _) = fir.process(&channels)?;
```

### Python (NumPy)

```sh
scripts/build_python.sh && pip install ./python
```

```python
import numpy as np, ane_unleashed as ane
count, total = ane.Table([a, b, c]).query([(0, ">", 25.0), (1, "<", -10.0)], sum_col=2)
pos, pattern_id = ane.Grep([b"ERROR", b"timeout"]).scan(open("app.log", "rb").read())
y = ane.Fir(irs).process(x)        # irs: float32 [ch, taps], x: float32 [ch, n]
```

## Limits

* Every tensor dimension is ≤ 16384, and conv kernels are ≤ 15 wide. int32 inputs are unsupported.
* fp16 datapath: SUM is approximate and FIR SNR is about 56–59 dB. COUNT, masks and grep
  matches are exact.
* Compiling costs 0.05–6 s per distinct query, pattern set or filter bank (cached in-process).
* Grep throughput scales roughly as `8.6 GB/s ÷ (patterns × projections)` on the ANE,
  so large pattern sets suit the ANE poorly.
* `ane-top`/`ane-bench` show rail watts only with root. Without root, macOS 27.2 freezes
  the IOReport CPU/ANE energy counters.

## Repository layout

| path | contents |
|---|---|
| `native/ane_bridge.{h,m}` | C ABI over the private ANE classes, IOSurfaces, IOReport/rusage telemetry |
| `src/mil_codegen.rs` | MIL text + weight blob builder |
| `src/iosurface_tensor.rs` | IOSurface tensors in the ANE row layout |
| `src/runtime.rs` | compile cache, request binding |
| `src/{query,grep,dsp}.rs` | the three engines |
| `src/cpu_baseline.rs` | multi-core CPU baselines (rayon, Aho-Corasick, vDSP, FFT) |
| `src/bin/ane_top.rs`, `src/bin/ane_bench.rs` | power monitor, benchmark harness |
| `python/ane_unleashed` | NumPy bindings (ctypes over the cdylib) |
| `tests/hardware.rs`, `python/tests` | real-hardware correctness tests |
| `HARDWARE_NOTES.md` | everything measured about the ANE along the way |
| `bench_results/` | raw benchmark output (`results.md`, `results.json`, `run.log`) |

## License

Apache-2.0. See [LICENSE](LICENSE).
