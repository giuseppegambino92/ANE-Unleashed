# Hardware notes: ANE behaviour measured on M3 Max / macOS 27.2.1

Everything here was observed on one machine: MacBook Pro M3 Max (16-core ANE,
`ANEDevicePropertyANEVersion = 176`), macOS 27.2.1 (26B131), Xcode 27.2. The
interfaces are private Apple SPI, so expect any of this to change between OS builds.

## The in-memory MIL path

`AppleNeuralEngine.framework` exposes these classes:

| class | used for |
|---|---|
| `_ANEInMemoryModelDescriptor +modelWithMILText:weights:optionsPlist:` | MIL text (NSData) + `{"@model_path/weights/weight.bin": {"offset": 0, "data": NSData}}` |
| `_ANEInMemoryModel +inMemoryModelWithDescriptor:` → `compileWithQoS:options:error:` → `loadWithQoS:options:error:` | compile + load via `aned` / `ANECompilerService` |
| `_ANEIOSurfaceObject +objectWithIOSurface:` | wraps IOSurfaces as inputs and outputs |
| `_ANERequest +requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:` | binds surfaces |
| `_ANEInMemoryModel evaluateWithQoS:options:request:error:` | synchronous execution |

**Caveat: it isn't truly zero-disk.** `compileWithQoS:` fails with
`verifyBundleAtPath: invalid model` unless `model.mil` and `weights/weight.bin` exist
under `$TMPDIR/<hexStringIdentifier>/`. The bridge writes them there, compiles,
loads, and deletes the directory straight away. Nothing goes through CoreML or
`.mlmodelc`, but the compiler service does read a short-lived staging copy.

There's no CPU fallback on this path: a program either compiles for the ANE or fails.

**IOSurfaces bind to inputs and outputs in alphabetical order of their MIL names, not
in argument/return order.** For outputs named `(zz_small, aa_big)`, the surface for
`aa_big` has to come first; binding in return order fails with `IOSurface smaller
than the model expects`. `MilBuilder` names inputs `i000_*` and outputs `o000…` so
the two orders line up.

**A non-weight `const` that nothing references makes the whole program
`InvalidMILProgram`** (an unused `tensor<int32,[4]>` pad constant is enough).
Unreferenced *blob* chunks are fine. `MilBuilder` only creates constants where
they're used.

## MIL dialect accepted

`program(1.3)`, `func main<ios18>(...)`, the same text syntax CoreML 9 emits (see
`/System/Library/Frameworks/MusicUnderstanding.framework/.../KeyModel.mlmodelc/model.mil`).
Weight blob: a 64-byte file header (`u32 count, u32 version=2`), then per tensor a
64-byte chunk header (`u32 0xDEADBEEF, u32 dtype (1=fp16, 2=fp32, 3=u8, 4=i8), u64 size,
u64 data_offset`) followed by 64-byte-aligned data. `BLOBFILE(offset=…)` points at the
**chunk header**.

## IOSurface tensor layout

* A `[N,C,H,W]` tensor is `N·C·H` rows of `W` elements, and **each row starts on a
  64-byte boundary**. For example, `[1,4,1,100]` fp16 has a 256-byte row stride; the
  dense 200-byte layout gives wrong results. Verified for W = 7, 33, 100, 1000.
* **Every dimension is capped at 16384.** `[1,1,1,65536]` fails to compile;
  `[1,1024,1,16384]` and `[1,256,64,1024]` work.

## Op support (compile = yes, results verified)

| op | notes |
|---|---|
| `cast` uint8/int8/fp32 → fp16, fp16 → uint8 | uint8 → fp16 exact. **int32 inputs fail to compile.** fp16 → uint8 saturates and rounds ties away from zero |
| `greater`, `less`, `*_equal`, `equal`, `not_equal`, `logical_and`, `select` | exact |
| `reduce_sum`, `reduce_min`, `abs`, `sub`, `mul`, `minimum`, `max_pool` | compile |
| `transpose`, `reshape` | used inside the FIR graph, verified end to end |
| `conv` | **kernel width ≤ 15** (16 fails, at any W). Kernel height 16 compiles |

## Numerics

* **fp32 → fp16 conversion rounds ties away from zero, not IEEE round-to-nearest-even.**
  On 16384 exact fp16 midpoints the ANE agreed with round-half-away 100% of the time
  and with RNE 50% of the time. A predicate like `x > 0.5` on fp32 input therefore
  selects `0.50024414` (= 0.5 + 2⁻¹², the midpoint), where numpy's RNE cast would
  not. `query::reference_query` models this exactly.
* **The conv accumulator is exact for integer data and the bias is fused**: a 15-tap
  conv over uint8 bytes with small-integer weights and an integer bias (|result| up
  to ~9.6k) matched `round_half_away_fp16(exact integer sum + bias)` on 100% of
  524,000 outputs. ANE-Grep's no-false-negative guarantee rests on this.

## Throughput figures seen during bring-up

| kernel | shape | time | rate |
|---|---|---:|---:|
| dispatch overhead (64×64 1×1 conv, 256 wide) | | ~0.13–0.18 ms | |
| 2-column predicate filter, fp16 in | 16.7 M rows | 1.7 ms | 9.8 G rows/s (39 GB/s in) |
| 2-column predicate filter, fp32 in | 16.7 M rows | 2.7 ms | 6.3 G rows/s (50 GB/s in) |
| 32 ch × 4096-tap FIR (block-Toeplitz) | 32 × 524,288 samples | 13.6 ms | 1.24 G samples/s, ~10 effective TFLOP/s, SNR 58.4 dB |

## Telemetry without root (macOS 27.2)

* `IOReport` "Energy Model": `IOReportCreateSamplesDelta` and
  `IOReportSimpleGetIntegerValue` return 0 on this build. The raw counter sits at byte
  32 of `RawElements`. Only **GPU Energy** advances for an unprivileged process; the
  PMGR `CPU Energy`, `ANE0` and `DRAM0` counters stay frozen.
* The `PMP0` ANE state-residency channels also don't move unprivileged.
* `IORegistry H11ANEIn → IOPowerManagement.CurrentPowerState` changes **0 → 1 while
  ANE work runs**. This is the unprivileged proof-of-activity signal ane-top and
  ane-bench use.
* `proc_pid_rusage(RUSAGE_INFO_V6).ri_energy_nj` gives per-process CPU energy.
* `_ANEPerformanceStats` hardware execution time could not be obtained:
  `perfStats` needs `_ANEPerformanceStatsIOSurface` objects, and the attempts made
  evaluation fail.
* Rail watts (`cpu_power`, `gpu_power`, `ane_power`) need `sudo powermetrics`.

## Layout lesson from ANE-Grep

Grep was first built as `[1, 1, R, 16384]` (text rows on the H axis) with
`[1, P·m, R, 16384]` intermediates. Compile time rose super-linearly with R: 0.8 s,
3.9 s and 40.7 s for R = 1, 4, 16, then more than 10 hours at R = 2048. Throughput
was 0.13 GB/s. Moving the text segments into **channels** (`[1, C, 1, 16384]`,
grouped convs with `groups = C`) brought every compile down to about 0.7 s.
Replacing broadcast compares against per-pattern targets (`[1,C,1,W]` vs
`[1,1,P,1]`, 7–15× slower than the hash itself) with a grouped conv carrying a fused
per-pattern bias gave another 8×. ANE-only throughput is roughly **8.6 GB/s divided
by the number of pattern×projection channels** (16 → 1.7 GB/s, 4 → 4.1 GB/s,
1 → 8.6 GB/s).

## Hazard: runaway compiles

`ANECompilerService` (root) compiles **one program at a time**. A grep graph with
`[1, 16, 2048, 16384]` intermediates (16 hash channels × 32 M positions) kept it at
100% CPU / 1.6 GB RSS for more than 10 hours. Killing the client doesn't cancel the
job, and every other ANE compile on the system, macOS's own included, waits behind
it. Recovering needs `sudo kill` on the service. The rows-on-H layout behind it is
gone (see above). With any new graph shape, start small and watch compile time grow.
