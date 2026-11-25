"""ane_unleashed — run columnar filters, multi-pattern byte scans and long FIR
filters on the Apple Neural Engine (via the Rust crate's C ABI).

    import numpy as np, ane_unleashed as ane
    t = ane.Table([a, b, c])                       # float32 columns, ingested once
    count, total = t.query([(0, ">", 25.0), (1, "<", -10.0)], sum_col=2)
    hits = ane.Grep([b"ERROR", b"timeout"]).scan(open("app.log", "rb").read())
    y = ane.Fir(irs).process(x)                    # irs: [ch, taps], x: [ch, n]

Uses undocumented Apple SPI (AppleNeuralEngine.framework); macOS on Apple Silicon only.
"""
from __future__ import annotations

import ctypes as C
import os
import pathlib

import numpy as np

__all__ = ["Table", "Grep", "Fir", "available", "ane_power_state", "AneError"]


class AneError(RuntimeError):
    pass


def _load():
    cands = []
    if os.environ.get("ANE_UNLEASHED_LIB"):
        cands.append(os.environ["ANE_UNLEASHED_LIB"])
    here = pathlib.Path(__file__).resolve().parent
    cands += [here / "libane_unleashed.dylib", here.parents[1] / "target" / "release" / "libane_unleashed.dylib"]
    for p in cands:
        if os.path.exists(p):
            return C.CDLL(str(p))
    raise OSError("libane_unleashed.dylib not found; run `cargo build --release` or set ANE_UNLEASHED_LIB")


_L = _load()
_vp, _sz, _i32, _u64 = C.c_void_p, C.c_size_t, C.c_int32, C.c_uint64
_L.aneu_last_error.restype = C.c_char_p
_L.aneu_ane_power_state.restype = _i32


class _CFilter(C.Structure):
    _fields_ = [("col", C.c_uint32), ("op", C.c_uint32), ("value", C.c_float)]


_OPS = {">": 0, ">=": 1, "<": 2, "<=": 3, "==": 4, "!=": 5}


def _check(rc):
    if rc != 0:
        raise AneError(_L.aneu_last_error().decode(errors="replace"))


def available() -> bool:
    return _L.aneu_available() == 0


def ane_power_state() -> int:
    """H11ANE power-gate state: 0 = off, 1 = on, -1 = unknown."""
    return _L.aneu_ane_power_state()


def _ptr_array(arrs, ctype):
    return (C.POINTER(ctype) * len(arrs))(*[a.ctypes.data_as(C.POINTER(ctype)) for a in arrs])


class Table:
    """Columnar table resident in ANE IOSurfaces. Columns: equal-length float32 arrays."""

    def __init__(self, columns):
        self._cols = [np.ascontiguousarray(c, dtype=np.float32) for c in columns]
        n = len(self._cols[0])
        if any(len(c) != n for c in self._cols):
            raise ValueError("columns must have equal length")
        self.rows = n
        self._h = _vp()
        _check(_L.aneu_table_new_f32(_ptr_array(self._cols, C.c_float), _sz(len(self._cols)), _sz(n), C.byref(self._h)))

    def query(self, filters, sum_col=None, mask=False):
        """filters: [(col, op, value)], op in > >= < <= == !=, ANDed together.
        Returns (count, sum) or (count, sum, mask: uint8[rows])."""
        fs = (_CFilter * len(filters))(*[_CFilter(c, _OPS[o], v) for c, o, v in filters])
        cnt, s, ane_s = _u64(), C.c_double(), C.c_double()
        m = np.zeros(self.rows, np.uint8) if mask else None
        _check(_L.aneu_table_query(self._h, fs, _sz(len(filters)), C.c_int64(-1 if sum_col is None else sum_col),
                                   C.byref(cnt), C.byref(s), m.ctypes.data_as(_vp) if mask else None, C.byref(ane_s)))
        out = (int(cnt.value), float(s.value) if sum_col is not None else None)
        return out + (m,) if mask else out

    def __del__(self):
        if getattr(self, "_h", None):
            _L.aneu_table_free(self._h)
            self._h = None


class Grep:
    """Exact multi-pattern byte scanner (ANE prefilter + CPU confirmation)."""

    def __init__(self, patterns, projections=2, rows_per_dispatch=256):
        self._pats = [bytes(p) for p in patterns]
        bufs = [C.create_string_buffer(p, len(p)) for p in self._pats]
        ptrs = (C.c_void_p * len(bufs))(*[C.cast(b, C.c_void_p) for b in bufs])
        lens = (C.c_size_t * len(bufs))(*[len(p) for p in self._pats])
        self._h = _vp()
        _check(_L.aneu_grep_new(ptrs, lens, _sz(len(bufs)), _sz(projections), _sz(rows_per_dispatch), C.byref(self._h)))

    def scan(self, data):
        """Returns (positions uint64[n], pattern_ids uint32[n]) sorted by position."""
        buf = np.frombuffer(data, dtype=np.uint8) if not isinstance(data, np.ndarray) else np.ascontiguousarray(data, np.uint8)
        lst, n = _vp(), _sz()
        _check(_L.aneu_grep_scan(self._h, buf.ctypes.data_as(_vp), _sz(buf.size), C.byref(lst), C.byref(n)))
        pos, pat = np.empty(n.value, np.uint64), np.empty(n.value, np.uint32)
        _L.aneu_matches_take(lst, pos.ctypes.data_as(_vp), pat.ctypes.data_as(_vp))
        return pos, pat

    def __del__(self):
        if getattr(self, "_h", None):
            _L.aneu_grep_free(self._h)
            self._h = None


class Fir:
    """Multi-channel FIR bank. irs: float32 [channels, taps]."""

    def __init__(self, irs, block=0, frames=1024):
        self._irs = np.ascontiguousarray(irs, dtype=np.float32)
        ch, taps = self._irs.shape
        rows = [self._irs[i] for i in range(ch)]
        self.channels = ch
        self._h = _vp()
        _check(_L.aneu_fir_new(_ptr_array(rows, C.c_float), _sz(ch), _sz(taps), _sz(block), _sz(frames), C.byref(self._h)))

    def process(self, x):
        """x: float32 [channels, n] -> y float32 [channels, n] (causal, zero initial state)."""
        x = np.ascontiguousarray(x, dtype=np.float32)
        if x.shape[0] != self.channels:
            raise ValueError(f"expected {self.channels} channels")
        y = np.empty_like(x)
        xs, ys = [x[i] for i in range(self.channels)], [y[i] for i in range(self.channels)]
        _check(_L.aneu_fir_process(self._h, _ptr_array(xs, C.c_float), _ptr_array(ys, C.c_float), _sz(x.shape[1])))
        return y

    def __del__(self):
        if getattr(self, "_h", None):
            _L.aneu_fir_free(self._h)
            self._h = None
