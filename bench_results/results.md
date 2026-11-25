# ANE-Unleashed benchmark results

Machine: Apple M3 Max (16-core ANE, 36 GB), macOS 27.2.1. Power rails: `powermetrics` cpu_power/gpu_power/ane_power, 250 ms samples.

Idle baseline (measured before the workloads): CPU 1.48 W, GPU 0.28 W, ANE 0.000 W; 1-min load average 2.2.

| workload | engine | latency/iter | throughput | process CPU time / wall | ANE domain powered | CPU W | GPU W | ANE W | workload W (above idle) | J per GB | GB/s per W |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| query 50M rows | cpu-rayon [3 rounds: 96.56–104.63 GB/s] | 6.2 ms | 97.27 GB/s | 11.0 cores | 99% | 25.81 | 0.20 | 0.000 | 24.33 | 0.25 | 3.998 |
| query 50M rows | ane [3 rounds: 31.56–32.23 GB/s] | 18.9 ms | 31.71 GB/s | 0.1 cores | 99% | 1.09 | 0.25 | 1.804 | 1.80 | 0.06 | 17.579 |
| grep 1024 MiB, 16 patterns, sparse matches | cpu-aho-corasick (match list) [3 rounds: 7.72–8.33 GB/s] | 129.0 ms | 8.32 GB/s | 12.8 cores | 100% | 34.30 | 0.06 | 0.000 | 32.82 | 3.94 | 0.254 |
| grep 1024 MiB, 16 patterns, sparse matches | cpu-aho-corasick (count only) [3 rounds: 8.02–8.33 GB/s] | 129.2 ms | 8.31 GB/s | 12.8 cores | 59% | 34.15 | 0.05 | 0.000 | 32.67 | 3.93 | 0.254 |
| grep 1024 MiB, 16 patterns, sparse matches | ane (p=1) [3 rounds: 1.47–1.49 GB/s] | 723.6 ms | 1.48 GB/s | 4.7 cores | 100% | 8.46 | 0.06 | 2.445 | 9.42 | 6.35 | 0.157 |
| grep 1024 MiB, 16 patterns, sparse matches | ane (p=2) [3 rounds: 0.67–0.69 GB/s] | 1559.5 ms | 0.69 GB/s | 1.1 cores | 100% | 1.15 | 0.06 | 2.122 | 2.12 | 3.08 | 0.324 |
| grep 1024 MiB, 16 patterns, dense matches | cpu-aho-corasick (match list) [3 rounds: 5.48–6.84 GB/s] | 181.2 ms | 5.93 GB/s | 11.1 cores | 100% | 33.47 | 0.09 | 0.000 | 31.99 | 5.40 | 0.185 |
| grep 1024 MiB, 16 patterns, dense matches | cpu-aho-corasick (count only) [3 rounds: 5.96–7.10 GB/s] | 171.6 ms | 6.26 GB/s | 11.5 cores | 70% | 31.25 | 0.16 | 0.000 | 29.77 | 4.76 | 0.210 |
| grep 1024 MiB, 16 patterns, dense matches | ane (p=1) [3 rounds: 1.33–1.40 GB/s] | 783.0 ms | 1.37 GB/s | 5.1 cores | 100% | 20.98 | 0.08 | 2.160 | 21.66 | 15.79 | 0.063 |
| grep 1024 MiB, 16 patterns, dense matches | ane (p=2) [3 rounds: 0.70–0.75 GB/s] | 1479.5 ms | 0.73 GB/s | 2.2 cores | 100% | 13.91 | 0.43 | 2.280 | 14.86 | 20.47 | 0.049 |
| FIR 32ch x 4096 taps, 30 s audio | cpu-vDSP-direct [3 rounds: 0.29–0.30 GB/s] | 625.7 ms | 0.29 GB/s | 9.9 cores | 99% | 32.02 | 0.44 | 0.000 | 30.70 | 104.22 | 0.010 |
| FIR 32ch x 4096 taps, 30 s audio | cpu-FFT [3 rounds: 7.10–7.29 GB/s] | 25.9 ms | 7.12 GB/s | 10.2 cores | 20% | 33.79 | 0.43 | 0.000 | 32.46 | 4.56 | 0.219 |
| FIR 32ch x 4096 taps, 30 s audio | ane [3 rounds: 2.84–3.07 GB/s] | 63.3 ms | 2.91 GB/s | 1.5 cores | 100% | 9.67 | 0.19 | 3.820 | 12.00 | 4.12 | 0.243 |

Notes:

- **query 50M rows**: ANE count 9261497 == fp16-semantics reference 9261497: true; CPU fp32 count 9262482 (diff 985); SUM rel.err vs reference 3.1e-4; 3 rounds: 31.56–32.23 GB/s
- **grep 1024 MiB, 16 patterns, sparse matches**: projections=1: 3243 matches, identical to Aho-Corasick: true; flagged 63.664% of 64 B blocks; stage times in one scan: copy 31 ms, ANE 716 ms, parallel CPU confirm 228 ms (copy+ANE overlap the confirm); 3 rounds: 1.47–1.49 GB/s
- **grep 1024 MiB, 16 patterns, sparse matches**: projections=2: 3243 matches, identical to Aho-Corasick: true; flagged 8.682% of 64 B blocks; stage times in one scan: copy 26 ms, ANE 1567 ms, parallel CPU confirm 106 ms (copy+ANE overlap the confirm); 3 rounds: 0.67–0.69 GB/s
- **grep 1024 MiB, 16 patterns, dense matches**: projections=1: 11560270 matches, identical to Aho-Corasick: true; flagged 91.359% of 64 B blocks; stage times in one scan: copy 22 ms, ANE 680 ms, parallel CPU confirm 304 ms (copy+ANE overlap the confirm); 3 rounds: 1.33–1.40 GB/s
- **grep 1024 MiB, 16 patterns, dense matches**: projections=2: 11560270 matches, identical to Aho-Corasick: true; flagged 56.047% of 64 B blocks; stage times in one scan: copy 48 ms, ANE 1447 ms, parallel CPU confirm 496 ms (copy+ANE overlap the confirm); 3 rounds: 0.70–0.75 GB/s
- **FIR 32ch x 4096 taps, 30 s audio**: SNR vs fp32 FFT reference 59.0 dB; realtime factor computed from latency; 3 rounds: 2.84–3.07 GB/s
- "process CPU time / wall" = cores kept busy by this process; "ANE domain powered" = share of 2 ms polls with the H11ANE power domain on; it stays on while any ANE program is loaded, so it is a liveness signal, not utilisation — use the ANE rail watts.
