# Running SQL Queries, Grep, and Audio DSP on Apple's Neural Engine at 1.8 Watts

*Draft. All numbers come from one MacBook Pro M3 Max (16-core ANE) on macOS 27.2.1,
measured with `powermetrics` rail telemetry. Raw data is in `bench_results/`.*

---

Every Apple Silicon Mac has a Neural Engine: a 16-core accelerator that sits dark
unless something runs a CoreML model. Meanwhile the CPU next to it burns 25–35 W
scanning columns, grepping logs and convolving audio. So I wanted to know whether
the ANE could do that non-AI work, and whether it would do it on less energy.

Short version: **yes for bulk column filters, sort of for FIR filtering, and mostly
no for grep.** A 50-million-row `WHERE a > X AND b < Y` with COUNT and SUM ran on
1.8 W of ANE power with the CPU idle, using 4.4× less energy than a 12-core CPU
implementation. It was also 3× slower. The details are below.

## Getting code onto the ANE without CoreML

Apple's documented route to the ANE is CoreML: build a model, compile it to
`.mlmodelc`, and hope the scheduler doesn't send your layers to the CPU. I wanted
programs generated on the fly, with no hope involved.

`AppleNeuralEngine.framework` (private) contains `_ANEInMemoryModelDescriptor`,
which accepts MIL text (CoreML's intermediate language) plus a weight blob
straight from memory. `_ANEInMemoryModel` compiles and loads it through the
`aned` daemon. `_ANERequest` binds IOSurfaces as inputs and outputs, and
`evaluateWithQoS:` runs it. This path has no CPU fallback: a program either
compiles for the ANE or fails. That's exactly what I wanted.

I learned the MIL dialect from models shipped with macOS (`program(1.3)`,
`func main<ios18>`). The weight-blob format is a 64-byte header followed by one
`0xDEADBEEF` chunk per tensor. The Rust side generates both in memory.

It isn't quite zero-disk, though. The compiler service refuses to compile until
the program also exists as files under `$TMPDIR/<hash>/`, so the bridge writes them,
compiles, and deletes them again.

## Things the ANE doesn't tell you

The SPI is undocumented, so most of the work was experiments. These are the
findings that shaped the design. Each was measured, not guessed:

1. **Every tensor dimension is capped at 16384**, and **conv kernels can be at most 15
   wide**. A 16-tap kernel fails to compile at any size.
2. **Tensor rows are 64-byte aligned.** A `[1,4,1,100]` fp16 tensor has a 256-byte row
   stride, and packing rows densely gives silently wrong results.
3. **fp32→fp16 conversion rounds ties away from zero**, not to nearest-even. I found
   this when a filter selected `0.50024414` for `x > 0.5`; that value is exactly halfway
   between two fp16 numbers. The CPU reference model now copies this rule, and query
   results match it bit-for-bit.
4. **The conv accumulator is exact for integer data and fuses the bias.** It matched
   "exact integer sum, then round once" on 100% of half a million outputs. Grep is
   built on this.
5. **Outputs bind to IOSurfaces in alphabetical order of their MIL names**, not in
   return order. A query returning `(counts, mask)` failed with "IOSurface smaller than
   the model expects" until the outputs were renamed `o000`, `o001`.
6. **One unused integer constant makes the program invalid**
   (`InvalidMILProgram`). That one cost an afternoon.
7. **The compiler is a single, root-owned, serial service.** My first grep graph
   (text rows on the H axis, 2048 rows) kept it at 100% CPU for more than 10 hours.
   Killing my process didn't cancel the job, and *every* ANE compile on the machine,
   macOS's own included, queued behind it until it was killed with `sudo`. Moving the
   text into channels brought compile time down to 0.7 s.
8. **Without root, the power counters lie.** On macOS 27.2 the IOReport CPU and ANE
   energy counters stay frozen for unprivileged processes; only GPU energy moves. The
   one honest unprivileged signal is the ANE power domain switching from 0 to 1. All
   watt figures here come from `sudo powermetrics`.

## Engine 1: SQL-style filters

A query such as `SELECT COUNT(*), SUM(c) WHERE a > 25 AND b < -10` maps cleanly:
cast each column to fp16, compare, AND, multiply by `c`, and reduce along rows.
Columns are ingested once into IOSurfaces shaped `[1, 1024, 8, 2048]`, which is
16.7 M rows per dispatch, and every row is dense and 64-byte aligned. Partial counts
are taken over 2048 rows so they stay exactly representable in fp16. **COUNT and the
row mask are exact; SUM carries fp16 error (about 3e-4 relative here).**

| 50 M rows, 3 fp32 columns | throughput | power above idle | energy |
|---|---:|---:|---:|
| CPU (12 cores, vectorised, rayon) | 97.3 GB/s | 24.3 W | 0.25 J/GB |
| ANE | 31.7 GB/s | **1.8 W** (ANE rail; CPU at idle) | **0.06 J/GB** |

The CPU is faster because both sides are bandwidth-bound and the CPU has more
bandwidth to spend. The ANE moves those bytes at about 1/13 of the power. During the
ANE run the CPU rail actually fell *below* its idle level, and the process used
0.1 cores.

## Engine 2: grep

Matching bytes means comparisons and branches, which is not what a convolution
engine is built for. The working design treats the ANE as an **exact prefilter**.
For each pattern *j*, one conv computes `h_j(i) = Σ_k w_jk·(x[i+k] − p_jk)` using
small random integer weights, with the pattern folded into the bias. Because the
accumulator is exact, a true match always gives exactly 0, so false negatives can't
happen. Zeros are counted with a second grouped conv, and a 64-byte max-pool reduces
everything to one flag per block. The CPU then confirms matches only in flagged blocks,
pipelined with the next ANE dispatch. On every run, the final match set is identical
to Aho-Corasick's.

Getting there took two redesigns. The first layout took 40 s to compile at 16 rows
and wedged the compiler at 2048 rows. Broadcasting compares against per-pattern
targets ran 7–15× slower than doing the same thing as a conv with a fused bias. In the
end, ANE cost is roughly **8.6 GB/s ÷ (patterns × projections)**.

| 1 GiB log, 16 patterns | throughput | power above idle | energy |
|---|---:|---:|---:|
| CPU Aho-Corasick, sparse matches | 8.3 GB/s | 32.8 W | 3.9 J/GB |
| ANE, 2 projections, sparse matches | 0.69 GB/s | 2.1 W | **3.1 J/GB** |
| CPU Aho-Corasick, dense matches (11.6 M hits) | 6.3 GB/s | 29.8 W | 4.8 J/GB |
| ANE, 1 projection, dense matches | 1.37 GB/s | 21.7 W | 15.8 J/GB |

For needle-in-a-haystack searches, the ANE saves about 20% of the energy, but at a
twelfth of the speed. When matches are dense, the CPU confirmation step dominates and
the ANE loses outright. I don't think the ANE is the right tool for grep.

## Engine 3: long FIR filters

A 4096-tap impulse response won't fit in a 15-wide kernel. Splitting the signal into
512-sample frames turns the FIR into a block-Toeplitz convolution: a 512→512 grouped
conv across 9 frames, wrapped in transposes. All of that runs on the ANE, with
overlap-save between dispatches. For 32 channels that's 151 MB of fp16 weights, which
compile once in about 5 s. SNR against an fp32 FFT reference is 59 dB.

| 32 ch × 4096 taps, 30 s @ 48 kHz | throughput | power above idle | energy |
|---|---:|---:|---:|
| CPU `vDSP_conv` (direct form) | 0.29 GB/s | 30.7 W | 104 J/GB |
| CPU FFT overlap-save | 7.12 GB/s | 32.5 W | 4.6 J/GB |
| ANE | 2.91 GB/s (474× real time) | 12.0 W (3.8 W ANE) | **4.1 J/GB** |

Against direct convolution, the operation the ANE actually computes, it's 10× faster
and uses 25× less energy. Against an FFT, which does far less arithmetic, it only
breaks even on energy, about 10% better. The fp32 copies in and out of the ANE's
memory layout take roughly 1.5 CPU cores and 8 W, so a pipeline that stayed in fp16
end to end would come out ahead.

## So, should you?

* **Columnar scans:** yes, if you care about energy or need the CPU for something else.
  4.4× less energy is real, and so is the 3× speed penalty.
* **Audio/sensor filter banks:** it holds its own against an FFT on energy, crushes
  direct-form filters, and leaves the CPU nearly free. Real-time use is fine at 474×
  headroom.
* **Grep:** not really. It's exact and it works, but the hardware isn't built for it.

The bigger caveat is that all of this rests on private SPI that broke in small ways
during development and will break again. Treat it as a research tool, not a
dependency.

Code, tests, raw benchmark data and the full hardware notes are in the repository:
`HARDWARE_NOTES.md` is the part most likely to save someone else a week.
