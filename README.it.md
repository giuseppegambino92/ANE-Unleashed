# ANE-Unleashed

[English](README.md) · **Italiano**

**Filtri SQL su colonne, grep multi-pattern e filtri FIR lunghi eseguiti sull'Apple
Neural Engine.** È un runtime in Rust + Objective-C che genera programmi MIL in
memoria e li esegue sull'ANE tramite il percorso privato `_ANEInMemoryModel` di
Apple. CoreML e i bundle `.mlmodelc` non vengono mai usati.

**Risultato misurato (M3 Max, potenza dei rail da `powermetrics`, mediana di 3 round
alternati):** un filtro a due predicati con COUNT/SUM su **50 milioni di righe gira a
31,7 GB/s consumando 1,8 W sull'ANE**, con la CPU ferma (0,1 core). La versione CPU
a 12 core arriva a 97 GB/s ma assorbe 24 W: l'ANE è quindi **4,4× più efficiente in
energia** (0,06 contro 0,25 J/GB), ma **3,1× più lento**. Per grep e FIR il quadro è
misto: vedi [Risultati](#risultati). In velocità pura la CPU vince in tutti i
carichi tranne il FIR in forma diretta.

> ⚠️ **SPI privata e non documentata di Apple.** Sviluppato e misurato su una sola
> macchina (MacBook Pro M3 Max, ANE a 16 core v176, macOS 27.2.1). Può smettere di
> funzionare con qualsiasi aggiornamento di macOS e non è utilizzabile su App Store.
> Leggi [HARDWARE_NOTES.md](HARDWARE_NOTES.md) prima di cambiare le forme dei grafi:
> una forma sbagliata può bloccare per ore il compilatore ANE di tutto il sistema.

## A cosa serve

Ogni Mac con Apple Silicon ha un Neural Engine: un acceleratore da 16 core che resta
spento finché qualcosa non esegue un modello CoreML. Nel frattempo la CPU accanto
consuma 25–35 W per scansionare colonne di dati, cercare nei log o filtrare audio.

Questo progetto verifica se l'ANE può fare quel lavoro *non-AI* e se lo fa
consumando meno energia, riformulando ciascun algoritmo come operazioni che l'ANE
sa eseguire (convoluzioni, confronti elemento per elemento, riduzioni):

* **Query** su colonne: `WHERE a > X AND b < Y` con COUNT e SUM.
* **Grep**: ricerca esatta di molti pattern in gigabyte di log.
* **DSP**: filtri FIR lunghi (risposte all'impulso) su molti canali audio.

La risposta, misurata: **sì per i filtri su colonne, più o meno per il FIR, quasi
sempre no per grep.**

## Come funziona

```
 il tuo codice Rust / Python
        │  colonne, testo, audio (f32 / u8)
        ▼
 ┌──────────────────────────┐   testo MIL + weight.bin, generati in RAM
 │ mil_codegen.rs           │   (program(1.3) / func main<ios18>)
 │ query.rs grep.rs dsp.rs  │
 └────────────┬─────────────┘
              │ cache delle compilazioni (si compila una volta, poi ~0,15 ms per esecuzione)
              ▼
 ┌──────────────────────────┐   _ANEInMemoryModelDescriptor
 │ native/ane_bridge.m      │   _ANEInMemoryModel compile/load/evaluate
 │ (C ABI, ObjC, ARC)       │   _ANERequest + _ANEIOSurfaceObject
 └────────────┬─────────────┘
              │ XPC: aned / ANECompilerService (compilazione, una volta)
              │ IOSurface (righe allineate a 64 byte), zero-copy verso l'ANE
              ▼
 ┌──────────────────────────┐
 │ H11ANE, ANE a 16 core    │   datapath fp16, accumulatori larghi ed esatti
 └──────────────────────────┘
```

Il programma non passa mai da CoreML, ma non è del tutto "zero-disk": il servizio di
compilazione vuole leggerne una copia temporanea da `$TMPDIR/<hash>/`. Il bridge la
scrive, compila e la cancella subito.

## I tre motori

### ANE-Query: `SELECT COUNT(*), SUM(c) WHERE a > X AND b < Y`

```
 col a ─cast fp16─▶ a > X ─┐
                           ├─ AND ─▶ maschera ─┬─ reduce_sum/2048 righe ─▶ parziali COUNT (esatti)
 col b ─cast fp16─▶ b < Y ─┘                   └─ × c ─ reduce_sum ─────▶ parziali SUM (fp16)
 tensori [1, 1024, 8, 2048] = 16,7 M righe per esecuzione, caricati una volta nelle IOSurface
```

* **COUNT e la maschera delle righe sono esatti**, verificati bit per bit contro un
  modello CPU della semantica dell'ANE. I predicati lavorano sui valori arrotondati a
  fp16, e l'ANE arrotonda i casi a metà **lontano dallo zero**, non al pari più vicino.
* **SUM è approssimata**: errore relativo ≈ 1e-4–1e-3, perché i parziali sono fp16.
* Le costanti fp32 vengono riscritte come costanti fp16 equivalenti, compresi i casi
  limite ±∞.

### ANE-Grep: ricerca esatta multi-pattern, con l'ANE come prefiltro

```
 testo (u8) come [1, C segmenti, 1, 16384]
   └─ conv: per ogni pattern j, h_j(i) = Σ_k w_jk·(x[i+k] − p_jk)   (pesi interi, bias integrato)
        └─ h == 0 ─▶ conteggio con conv a gruppi ─▶ ≥ m ─▶ max-pool 64 ─▶ 1 flag per blocco da 64 byte
 CPU (rayon, in pipeline con l'esecuzione ANE successiva): conferma i pattern solo nei blocchi segnalati
```

La convoluzione accumula in modo esatto e arrotonda una sola volta, quindi ogni
occorrenza vera dà esattamente `h == 0`: **nessun falso negativo**. La conferma sulla
CPU scarta tutti i falsi positivi, e il risultato coincide con quello di
Aho-Corasick (lo verificano i test e ogni esecuzione del benchmark). L'ANE guarda i
primi 15 byte di ogni pattern (il kernel di convoluzione non può essere più largo);
il resto lo controlla la CPU.

### ANE-DSP: filtri FIR / risposte all'impulso lunghe su più canali

Un filtro da 4096 coefficienti non entra in un kernel ANE (al massimo 15 di
larghezza). Diventa quindi una convoluzione a blocchi di Toeplitz su frame da 512
campioni:

```
 x [1, C, M, 512] ─▶ transpose ─▶ [1, C·512, 1, M] ─▶ conv a gruppi (512→512 per canale,
 9 frame, causale) ─▶ transpose inversa ─▶ y        (overlap-save tra un'esecuzione e l'altra)
```

32 canali × 4096 coefficienti fanno 151 MB di pesi fp16, compilati una volta in circa
5 s. L'SNR rispetto a una FFT fp32 di riferimento è 56–59 dB.

## Risultati

Output completo: [`bench_results/results.md`](bench_results/results.md),
[`results.json`](bench_results/results.json), [`run.log`](bench_results/run.log).
M3 Max collegato all'alimentazione, consumo a riposo CPU 1,48 W / GPU 0,28 W /
ANE 0 W. Ogni motore gira 5 s per round, su 3 round alternati; la tabella riporta il
round mediano. "W del carico" è la somma dei rail CPU, GPU e ANE al di sopra del
riposo. Ogni risultato ANE è stato confrontato con il riferimento nella stessa
esecuzione.

| carico | motore | throughput | core occupati | rail ANE | W del carico | J per GB | rispetto alla CPU migliore (J/GB) |
|---|---|---:|---:|---:|---:|---:|---:|
| **query** 50 M righe × 3 colonne f32 | CPU, rayon, 12 core | 97,3 GB/s | 11,0 | 0 W | 24,3 W | 0,25 | |
| | **ANE** | 31,7 GB/s | 0,1 | 1,80 W | **1,8 W** | **0,06** | **4,4× meno energia** |
| **grep** log da 1 GiB, 16 pattern, occorrenze *rare* | CPU, Aho-Corasick | 8,3 GB/s | 12,8 | 0 W | 32,8 W | 3,9 | |
| | ANE, 2 proiezioni | 0,69 GB/s | 1,1 | 2,12 W | 2,1 W | 3,1 | 1,3× meno energia, 12× più lento |
| | ANE, 1 proiezione | 1,48 GB/s | 4,7 | 2,45 W | 9,4 W | 6,4 | 1,6× più energia |
| **grep** log da 1 GiB, 16 pattern, occorrenze *frequenti* (11,6 M) | CPU, Aho-Corasick (solo conteggio) | 6,3 GB/s | 11,5 | 0 W | 29,8 W | 4,8 | |
| | ANE, 1 proiezione | 1,37 GB/s | 5,1 | 2,16 W | 21,7 W | 15,8 | 3,3× più energia |
| **FIR** 32 canali × 4096 coeff., 30 s a 48 kHz | CPU, `vDSP_conv` diretto | 0,29 GB/s | 9,9 | 0 W | 30,7 W | 104 | |
| | CPU, FFT overlap-save | 7,12 GB/s | 10,2 | 0 W | 32,5 W | 4,6 | |
| | ANE | 2,91 GB/s | 1,5 | 3,82 W | 12,0 W | 4,1 | 1,1× meno energia della FFT; 25× meno del diretto |

Cosa dicono le misure:

* **L'ANE vince nelle scansioni massive con predicati.** Entrambi sono limitati dalla
  banda di memoria, e l'ANE sposta gli stessi byte con una frazione della potenza.
  Durante la query su ANE il rail della CPU è sceso perfino *sotto* il riposo.
* **Grep non è adatto.** Il costo sull'ANE cresce con pattern × proiezioni, e il
  prefiltro ha bisogno della conferma CPU. Con due proiezioni i falsi positivi sono
  pochi (8,7% dei blocchi da 64 byte sul log con occorrenze rare): la CPU resta quasi
  ferma e l'energia per GB batte Aho-Corasick di 1,3×, ma a 1/12 della velocità. Con
  molte occorrenze domina la conferma e la CPU vince nettamente.
* **Il FIR batte nettamente la convoluzione diretta (10× più veloce, 25× meno
  energia) ma supera di poco la FFT**, che fa molti meno calcoli. Il percorso ANE gira
  a 474× il tempo reale e spende ancora circa 1,5 core per copiare i dati fp32 in
  entrata e in uscita.
* **In velocità ha sempre vinto la CPU**, tranne contro il FIR diretto. Il vantaggio
  dell'ANE è l'energia, e lasciare la CPU libera per altro.

## Avvio rapido

```sh
cargo build --release
cargo test --release -- --test-threads=1      # esecuzioni reali su ANE, confrontate con riferimenti CPU esatti

# monitor della potenza in tempo reale: rail CPU / GPU / ANE con root, stato di accensione dell'ANE senza
./target/release/ane-top

# benchmark. I watt dei rail richiedono root o una regola sudoers NOPASSWD:
echo "$USER ALL=(root) NOPASSWD: /usr/bin/powermetrics" | sudo tee /etc/sudoers.d/powermetrics
sudo chmod 440 /etc/sudoers.d/powermetrics
scripts/run_bench.sh            # si interrompe se la macchina non è abbastanza a riposo per misurare
```

### Rust

```rust
use ane_unleashed::query::{AneTable, CmpOp, Filter};
let t = AneTable::from_f32(&[&a, &b, &c])?;                  // caricamento una tantum
let r = t.query(&[Filter { col: 0, op: CmpOp::Gt, value: 25.0 },
                  Filter { col: 1, op: CmpOp::Lt, value: -10.0 }], Some(2), false)?;
println!("{} righe, somma ≈ {}", r.count, r.sum.unwrap());

let g = ane_unleashed::grep::AneGrep::new(&[b"ERROR".as_slice(), b"timeout"], 2, 256)?;
let (matches, stats) = g.scan(&log_bytes)?;                  // esatto, ordinato per (posizione, pattern)

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
y = ane.Fir(irs).process(x)        # irs: float32 [canali, coeff.], x: float32 [canali, n]
```

## Limiti

* Ogni dimensione dei tensori è ≤ 16384 e i kernel di convoluzione sono larghi ≤ 15.
  Gli input int32 non sono supportati.
* Datapath fp16: SUM è approssimata e l'SNR del FIR è circa 56–59 dB. COUNT, maschere e
  risultati di grep sono esatti.
* La compilazione costa 0,05–6 s per ogni query, insieme di pattern o banco di filtri
  diverso (con cache nel processo).
* Sull'ANE grep scala circa come `8,6 GB/s ÷ (pattern × proiezioni)`, quindi molti
  pattern sono poco adatti all'ANE.
* `ane-top`/`ane-bench` mostrano i watt dei rail solo con root: senza root, macOS 27.2
  congela i contatori di energia IOReport di CPU e ANE.

## Struttura del repository

| percorso | contenuto |
|---|---|
| `native/ane_bridge.{h,m}` | C ABI sulle classi ANE private, IOSurface, telemetria IOReport/rusage |
| `src/mil_codegen.rs` | generatore di testo MIL e blob dei pesi |
| `src/iosurface_tensor.rs` | tensori IOSurface nel layout di righe dell'ANE |
| `src/runtime.rs` | cache delle compilazioni, binding delle richieste |
| `src/{query,grep,dsp}.rs` | i tre motori |
| `src/cpu_baseline.rs` | versioni CPU multi-core di confronto (rayon, Aho-Corasick, vDSP, FFT) |
| `src/bin/ane_top.rs`, `src/bin/ane_bench.rs` | monitor della potenza, harness dei benchmark |
| `python/ane_unleashed` | binding NumPy (ctypes sulla cdylib) |
| `tests/hardware.rs`, `python/tests` | test di correttezza su hardware reale |
| `HARDWARE_NOTES.md` | tutto ciò che è stato misurato sull'ANE lungo il percorso (in inglese) |
| `ARTICLE_DRAFT.md` | bozza dell'articolo sul progetto (in inglese) |
| `bench_results/` | output grezzo dei benchmark (`results.md`, `results.json`, `run.log`) |

## Licenza

Apache-2.0. Vedi [LICENSE](LICENSE).
