//! In-memory MIL (CoreML Model Intermediate Language) program generator.
//!
//! Emits the textual `program(1.3)` / `func main<ios18>` dialect that the ANE
//! compiler on macOS 27 accepts, plus a `weight.bin` blob in the CoreML blob
//! format (64-byte file header, then per-tensor 64-byte `0xDEADBEEF` chunk
//! headers each followed by 64-byte aligned data). MIL `BLOBFILE` offsets point
//! at the chunk header, not at the data.
use crate::iosurface_tensor::DType;
use half::f16;
use std::fmt::Write as _;

/// A value in the program being built.
#[derive(Clone, Debug)]
pub struct Var {
    pub name: String,
    pub dtype: VDType,
    pub shape: Vec<usize>,
}

/// MIL value types (adds `bool` to the IO dtypes).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VDType {
    Io(DType),
    Bool,
}
impl VDType {
    fn mil(self) -> &'static str {
        match self {
            VDType::Io(d) => d.mil(),
            VDType::Bool => "bool",
        }
    }
}
impl From<DType> for VDType {
    fn from(d: DType) -> Self {
        VDType::Io(d)
    }
}

fn tensor_ty(dt: VDType, shape: &[usize]) -> String {
    let dims: Vec<String> = shape.iter().map(|d| d.to_string()).collect();
    format!("tensor<{}, [{}]>", dt.mil(), dims.join(", "))
}

/// Exact hexadecimal float literal (e.g. `0x1.8p-1`) so no decimal rounding can occur.
pub fn hex_float(v: f64) -> String {
    if v == 0.0 {
        return if v.is_sign_negative() { "-0x0p+0".into() } else { "0x0p+0".into() };
    }
    assert!(v.is_finite(), "non-finite constant {v}");
    let bits = v.to_bits();
    let sign = if v < 0.0 { "-" } else { "" };
    let exp_bits = ((bits >> 52) & 0x7ff) as i64;
    let mut mant = bits & ((1u64 << 52) - 1);
    let (lead, exp) = if exp_bits == 0 {
        // subnormal f64: normalise
        let shift = mant.leading_zeros() as i64 - 11;
        mant = (mant << shift) & ((1u64 << 52) - 1);
        (1, -1022 - shift)
    } else {
        (1, exp_bits - 1023)
    };
    let mut hex = format!("{mant:013x}");
    while hex.ends_with('0') {
        hex.pop();
    }
    if hex.is_empty() {
        format!("{sign}0x{lead}p{exp:+}")
    } else {
        format!("{sign}0x{lead}.{hex}p{exp:+}")
    }
}

/// CoreML weight blob builder.
#[derive(Default)]
pub struct WeightBlob {
    chunks: Vec<u8>,
    count: u32,
}

impl WeightBlob {
    /// Append raw little-endian element bytes; returns the MIL offset (chunk header position).
    pub fn add_raw(&mut self, dtype: DType, data: &[u8]) -> u64 {
        let off = 64 + self.chunks.len() as u64;
        let mut hdr = [0u8; 64];
        hdr[0..4].copy_from_slice(&0xDEADBEEFu32.to_le_bytes());
        hdr[4..8].copy_from_slice(&dtype.blob_code().to_le_bytes());
        hdr[8..16].copy_from_slice(&(data.len() as u64).to_le_bytes());
        hdr[16..24].copy_from_slice(&(off + 64).to_le_bytes());
        self.chunks.extend_from_slice(&hdr);
        self.chunks.extend_from_slice(data);
        let pad = (64 - data.len() % 64) % 64;
        self.chunks.extend(std::iter::repeat(0u8).take(pad));
        self.count += 1;
        off
    }
    pub fn add_f16(&mut self, v: &[f16]) -> u64 {
        let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_bits().to_le_bytes()).collect();
        self.add_raw(DType::F16, &bytes)
    }
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }
    pub fn finish(&self) -> Vec<u8> {
        if self.count == 0 {
            return Vec::new();
        }
        let mut out = vec![0u8; 64];
        out[0..4].copy_from_slice(&self.count.to_le_bytes());
        out[4..8].copy_from_slice(&2u32.to_le_bytes());
        out.extend_from_slice(&self.chunks);
        out
    }
}

/// Conv parameters (MIL `conv`, NCHW cross-correlation).
#[derive(Clone, Debug)]
pub struct ConvSpec {
    /// weight shape [Cout, Cin/groups, KH, KW]
    pub kernel: [usize; 4],
    pub groups: usize,
    pub strides: [usize; 2],
    /// [top, bottom, left, right]
    pub pad: [usize; 4],
    pub dilations: [usize; 2],
}

/// Builder for a single-function MIL program.
pub struct MilBuilder {
    inputs: Vec<Var>,
    body: String,
    pub blob: WeightBlob,
    counter: usize,
}

impl Default for MilBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl MilBuilder {
    pub fn new() -> Self {
        Self { inputs: Vec::new(), body: String::new(), blob: WeightBlob::default(), counter: 0 }
    }

    fn fresh(&mut self, stem: &str) -> String {
        self.counter += 1;
        format!("{stem}_{}", self.counter)
    }

    fn line(&mut self, s: &str) {
        self.body.push_str("        ");
        self.body.push_str(s);
        self.body.push('\n');
    }

    /// Declare an input. The ANE runtime binds IOSurfaces to inputs (and outputs)
    /// in **alphabetical order of their MIL names**, not declaration order, so
    /// names are prefixed `i000_`, `i001_`, … to make the two orders identical.
    pub fn input(&mut self, name: &str, dtype: DType, shape: &[usize]) -> Var {
        let name = format!("i{:03}_{name}", self.inputs.len());
        let v = Var { name, dtype: dtype.into(), shape: shape.to_vec() };
        self.inputs.push(v.clone());
        v
    }

    fn const_line(&mut self, ty: &str, val: &str, stem: &str) -> String {
        let n = self.fresh(stem);
        self.line(&format!("{ty} {n} = const()[name = string(\"{n}\"), val = {val}];"));
        n
    }
    fn c_string(&mut self, s: &str) -> String {
        self.const_line("string", &format!("string(\"{s}\")"), "s")
    }
    fn c_bool(&mut self, b: bool) -> String {
        self.const_line("bool", &format!("bool({b})"), "b")
    }
    fn c_int(&mut self, i: i64) -> String {
        self.const_line("int32", &format!("int32({i})"), "i")
    }
    fn c_ints(&mut self, v: &[i64]) -> String {
        let ty = format!("tensor<int32, [{}]>", v.len());
        let vals: Vec<String> = v.iter().map(|x| x.to_string()).collect();
        let val = format!("{ty}([{}])", vals.join(", "));
        self.const_line(&ty, &val, "iv")
    }
    /// fp16 scalar constant; `v` must be exactly representable in fp16.
    pub fn scalar_f16(&mut self, v: f32) -> Var {
        let h = f16::from_f32(v);
        assert_eq!(h.to_f32(), v, "constant {v} is not exactly representable in fp16");
        let n = self.const_line("fp16", &format!("fp16({})", hex_float(v as f64)), "k");
        Var { name: n, dtype: DType::F16.into(), shape: vec![] }
    }
    /// Constant fp16 tensor stored in the weight blob.
    pub fn tensor_f16(&mut self, shape: &[usize], data: &[f16]) -> Var {
        assert_eq!(shape.iter().product::<usize>(), data.len());
        let off = self.blob.add_f16(data);
        let ty = tensor_ty(DType::F16.into(), shape);
        let n = self.fresh("w");
        self.line(&format!(
            "{ty} {n} = const()[name = string(\"{n}\"), val = {ty}(BLOBFILE(path = string(\"@model_path/weights/weight.bin\"), offset = uint64({off})))];"
        ));
        Var { name: n, dtype: DType::F16.into(), shape: shape.to_vec() }
    }

    fn emit(&mut self, dtype: VDType, shape: Vec<usize>, call: String, stem: &str) -> Var {
        let n = self.fresh(stem);
        let ty = tensor_ty(dtype, &shape);
        self.line(&format!("{ty} {n} = {call}[name = string(\"{n}\")];"));
        Var { name: n, dtype, shape }
    }

    pub fn cast(&mut self, x: &Var, to: DType) -> Var {
        let d = self.c_string(to.mil());
        self.emit(to.into(), x.shape.clone(), format!("cast(dtype = {d}, x = {})", x.name), "cast")
    }

    fn broadcast(a: &[usize], b: &[usize]) -> Vec<usize> {
        if a.is_empty() {
            return b.to_vec();
        }
        if b.is_empty() {
            return a.to_vec();
        }
        assert_eq!(a.len(), b.len(), "rank mismatch {a:?} vs {b:?}");
        a.iter()
            .zip(b)
            .map(|(&x, &y)| {
                assert!(x == y || x == 1 || y == 1, "cannot broadcast {a:?} {b:?}");
                x.max(y)
            })
            .collect()
    }

    /// Elementwise binary op: add, sub, mul, maximum, minimum (numeric result),
    /// greater, greater_equal, less, less_equal, equal, not_equal (bool result),
    /// logical_and, logical_or (bool inputs).
    pub fn binary(&mut self, op: &str, x: &Var, y: &Var) -> Var {
        let shape = Self::broadcast(&x.shape, &y.shape);
        let dt = match op {
            "greater" | "greater_equal" | "less" | "less_equal" | "equal" | "not_equal" | "logical_and"
            | "logical_or" => VDType::Bool,
            _ => x.dtype,
        };
        self.emit(dt, shape, format!("{op}(x = {}, y = {})", x.name, y.name), op)
    }

    pub fn unary(&mut self, op: &str, x: &Var) -> Var {
        self.emit(x.dtype, x.shape.clone(), format!("{op}(x = {})", x.name), op)
    }

    pub fn select(&mut self, cond: &Var, a: &Var, b: &Var) -> Var {
        let shape = Self::broadcast(&Self::broadcast(&cond.shape, &a.shape), &b.shape);
        self.emit(a.dtype, shape, format!("select(a = {}, b = {}, cond = {})", a.name, b.name, cond.name), "sel")
    }

    pub fn conv(&mut self, x: &Var, weight: &Var, bias: Option<&Var>, spec: &ConvSpec) -> Var {
        assert_eq!(x.shape.len(), 4);
        let [co, _, kh, kw] = spec.kernel;
        let h = x.shape[2] + spec.pad[0] + spec.pad[1];
        let w = x.shape[3] + spec.pad[2] + spec.pad[3];
        let eh = (kh - 1) * spec.dilations[0] + 1;
        let ew = (kw - 1) * spec.dilations[1] + 1;
        let oh = (h - eh) / spec.strides[0] + 1;
        let ow = (w - ew) / spec.strides[1] + 1;
        let pt = self.c_string("custom");
        let st = self.c_ints(&spec.strides.map(|v| v as i64));
        let pd = self.c_ints(&spec.pad.map(|v| v as i64));
        let dl = self.c_ints(&spec.dilations.map(|v| v as i64));
        let gr = self.c_int(spec.groups as i64);
        let b = bias.map(|b| format!("bias = {}, ", b.name)).unwrap_or_default();
        self.emit(
            x.dtype,
            vec![x.shape[0], co, oh, ow],
            format!(
                "conv(dilations = {dl}, groups = {gr}, pad = {pd}, pad_type = {pt}, strides = {st}, weight = {}, {b}x = {})",
                weight.name, x.name
            ),
            "conv",
        )
    }

    pub fn transpose(&mut self, x: &Var, perm: [usize; 4]) -> Var {
        let p = self.c_ints(&perm.map(|v| v as i64));
        let shape = perm.iter().map(|&i| x.shape[i]).collect();
        self.emit(x.dtype, shape, format!("transpose(perm = {p}, x = {})", x.name), "tr")
    }

    pub fn reshape(&mut self, x: &Var, shape: &[usize]) -> Var {
        assert_eq!(x.shape.iter().product::<usize>(), shape.iter().product::<usize>());
        let s = self.c_ints(&shape.iter().map(|&v| v as i64).collect::<Vec<_>>());
        self.emit(x.dtype, shape.to_vec(), format!("reshape(shape = {s}, x = {})", x.name), "rs")
    }

    /// 1 x k max pool with stride k along W.
    pub fn max_pool_w(&mut self, x: &Var, k: usize) -> Var {
        let ks = self.c_ints(&[1, k as i64]);
        let st = self.c_ints(&[1, k as i64]);
        let pt = self.c_string("valid");
        let pd = self.c_ints(&[0, 0, 0, 0]);
        let cm = self.c_bool(false);
        let mut shape = x.shape.clone();
        shape[3] /= k;
        self.emit(
            x.dtype,
            shape,
            format!("max_pool(ceil_mode = {cm}, kernel_sizes = {ks}, pad = {pd}, pad_type = {pt}, strides = {st}, x = {})", x.name),
            "mp",
        )
    }

    /// reduce_{sum,max,min} over `axes`, keep_dims = true.
    pub fn reduce(&mut self, op: &str, x: &Var, axes: &[usize]) -> Var {
        let ax = self.c_ints(&axes.iter().map(|&a| a as i64).collect::<Vec<_>>());
        let kd = self.c_bool(true);
        let mut shape = x.shape.clone();
        for &a in axes {
            shape[a] = 1;
        }
        self.emit(x.dtype, shape, format!("reduce_{op}(axes = {ax}, keep_dims = {kd}, x = {})", x.name), "red")
    }

    /// Finish: returns (MIL text, weight blob bytes). Outputs are renamed
    /// `o000`, `o001`, … so the runtime's alphabetical binding order matches the
    /// order given here.
    pub fn finish(mut self, outputs: &[&Var]) -> (String, Vec<u8>) {
        let mut out_names = Vec::new();
        for (k, v) in outputs.iter().enumerate() {
            let new = format!("o{k:03}");
            self.body = rename_token(&self.body, &v.name, &new);
            out_names.push(new);
        }
        let mut s = String::new();
        s.push_str("program(1.3)\n");
        s.push_str("[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3600.16.1\"}, {\"coremlc-version\", \"3600.22.1\"}, {\"ane-unleashed\", \"0.1.0\"}})]\n{\n");
        let args: Vec<String> = self.inputs.iter().map(|v| format!("{} {}", tensor_ty(v.dtype, &v.shape), v.name)).collect();
        let _ = writeln!(s, "    func main<ios18>({}) {{", args.join(", "));
        s.push_str(&self.body);
        let outs: Vec<&str> = out_names.iter().map(|v| v.as_str()).collect();
        let _ = writeln!(s, "    }} -> ({});", outs.join(", "));
        s.push_str("}\n");
        (s, self.blob.finish())
    }
}

/// Replace whole-identifier occurrences of `from` with `to`.
fn rename_token(src: &str, from: &str, to: &str) -> String {
    let is_id = |c: char| c.is_ascii_alphanumeric() || c == '_';
    let mut out = String::with_capacity(src.len());
    let mut rest = src;
    while let Some(i) = rest.find(from) {
        let before_ok = rest[..i].chars().next_back().map(|c| !is_id(c)).unwrap_or(true);
        let after_ok = rest[i + from.len()..].chars().next().map(|c| !is_id(c)).unwrap_or(true);
        out.push_str(&rest[..i]);
        out.push_str(if before_ok && after_ok { to } else { from });
        rest = &rest[i + from.len()..];
    }
    out.push_str(rest);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rename_whole_tokens() {
        assert_eq!(rename_token("red_1 = f(red_12, red_1)[name = string(\"red_1\")]", "red_1", "o000"),
                   "o000 = f(red_12, o000)[name = string(\"o000\")]");
    }
    #[test]
    fn hex_floats() {
        assert_eq!(hex_float(1.0), "0x1p+0");
        assert_eq!(hex_float(0.5), "0x1p-1");
        assert_eq!(hex_float(-0.75), "-0x1.8p-1");
        assert_eq!(hex_float(3.0), "0x1.8p+1");
        assert_eq!(hex_float(0.0), "0x0p+0");
    }
    #[test]
    fn blob_layout() {
        let mut b = WeightBlob::default();
        let o1 = b.add_f16(&[f16::ONE; 3]);
        let o2 = b.add_f16(&[f16::ONE; 40]);
        assert_eq!(o1, 64);
        assert_eq!(o2, 64 + 64 + 64);
        let bytes = b.finish();
        assert_eq!(&bytes[64..68], &0xDEADBEEFu32.to_le_bytes());
        assert_eq!(u64::from_le_bytes(bytes[o2 as usize + 16..o2 as usize + 24].try_into().unwrap()), o2 + 64);
    }
}
