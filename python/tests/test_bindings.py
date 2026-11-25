"""End-to-end checks of the Python bindings on real ANE hardware."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
import ane_unleashed as ane


def test_query():
    rng = np.random.default_rng(0)
    a, b, c = (rng.standard_normal(3_000_000).astype(np.float32) * s for s in (100, 100, 10))
    a[:1000] = 64.0  # fp16-exact values on the threshold
    t = ane.Table([a, b, c])
    cnt, s, mask = t.query([(0, ">", 25.0), (1, "<", -10.0)], sum_col=2, mask=True)
    a16, b16 = a.astype(np.float16), b.astype(np.float16)  # RNE; ties are measure-zero for random data
    ref = (a16 > 25.0) & (b16 < -10.0)
    assert cnt == int(ref.sum()) and np.array_equal(mask.astype(bool), ref)
    assert abs(s - float(c[ref].astype(np.float64).sum())) / max(1.0, abs(s)) < 5e-3


def test_grep():
    data = (b"INFO ok\n" * 50_000) + b"ERROR disk full\n" + (b"WARN slow\n" * 50_000) + b"ERROR again\n"
    pos, pat = ane.Grep([b"ERROR", b"WARN slow"]).scan(data)
    ref = sorted([(i, 0) for i in range(len(data)) if data.startswith(b"ERROR", i)] +
                 [(i, 1) for i in range(len(data)) if data.startswith(b"WARN slow", i)])
    assert list(zip(pos.tolist(), pat.tolist())) == ref


def test_fir():
    rng = np.random.default_rng(1)
    h = (rng.standard_normal((2, 1024)) * np.exp(-np.arange(1024) / 200) * 0.1).astype(np.float32)
    x = (rng.standard_normal((2, 100_000)) * 0.3).astype(np.float32)
    y = ane.Fir(h, frames=256).process(x)
    ref = np.stack([np.convolve(x[c].astype(np.float64), h[c].astype(np.float64))[: x.shape[1]] for c in range(2)])
    snr = 10 * np.log10((ref ** 2).sum() / ((y - ref) ** 2).sum())
    assert snr > 50, snr


if __name__ == "__main__":
    for f in (test_query, test_grep, test_fir):
        f(); print(f.__name__, "ok")
