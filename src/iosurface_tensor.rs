//! IOSurface-backed ANE tensors.
//!
//! Layout rule (measured on M3 Max / ANE v176, see HARDWARE_NOTES.md): a MIL
//! tensor `[N, C, H, W]` is stored as `N*C*H` rows of `W` elements and **every row
//! starts on a 64-byte boundary**. A `[1,4,1,100]` fp16 tensor therefore has a row
//! stride of 256 bytes, not 200; the unpadded layout produces wrong results.
//! Every dimension is limited to 16384 elements.
use crate::ffi;
use std::ptr::NonNull;

pub const ANE_ROW_ALIGN: usize = 64;
pub const ANE_MAX_DIM: usize = 16384;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DType {
    F16,
    F32,
    U8,
    I8,
}

impl DType {
    pub fn size(self) -> usize {
        match self {
            DType::F16 => 2,
            DType::F32 => 4,
            DType::U8 | DType::I8 => 1,
        }
    }
    pub fn mil(self) -> &'static str {
        match self {
            DType::F16 => "fp16",
            DType::F32 => "fp32",
            DType::U8 => "uint8",
            DType::I8 => "int8",
        }
    }
    /// dtype code used in weight.bin chunk headers
    pub fn blob_code(self) -> u32 {
        match self {
            DType::F16 => 1,
            DType::F32 => 2,
            DType::U8 => 3,
            DType::I8 => 4,
        }
    }
}

pub fn row_stride_bytes(w: usize, dt: DType) -> usize {
    (w * dt.size()).div_ceil(ANE_ROW_ALIGN) * ANE_ROW_ALIGN
}

/// An IOSurface holding one ANE tensor in the padded row layout.
pub struct AneTensor {
    surf: NonNull<std::ffi::c_void>,
    pub shape: [usize; 4],
    pub dtype: DType,
    pub row_stride: usize,
}

unsafe impl Send for AneTensor {}
unsafe impl Sync for AneTensor {}

impl AneTensor {
    pub fn new(shape: [usize; 4], dtype: DType) -> crate::Result<Self> {
        for &d in &shape {
            if d == 0 || d > ANE_MAX_DIM {
                return Err(crate::Error::Shape(format!(
                    "dimension {d} outside 1..={ANE_MAX_DIM} in {shape:?}"
                )));
            }
        }
        let row_stride = row_stride_bytes(shape[3], dtype);
        let rows = shape[0] * shape[1] * shape[2];
        let bytes = rows * row_stride;
        let s = unsafe { ffi::ane_surface_create(bytes) };
        let surf = NonNull::new(s).ok_or_else(|| crate::Error::Ane("IOSurfaceCreate failed".into()))?;
        Ok(Self { surf, shape, dtype, row_stride })
    }

    pub fn raw(&self) -> ffi::IOSurfaceRef {
        self.surf.as_ptr()
    }
    pub fn rows(&self) -> usize {
        self.shape[0] * self.shape[1] * self.shape[2]
    }
    pub fn row_bytes(&self) -> usize {
        self.shape[3] * self.dtype.size()
    }
    pub fn byte_len(&self) -> usize {
        self.rows() * self.row_stride
    }
    /// True when rows are densely packed (no padding), i.e. the surface is a plain array.
    pub fn is_dense(&self) -> bool {
        self.row_stride == self.row_bytes()
    }

    /// Lock for CPU writes and expose the raw padded bytes.
    pub fn write_with<R>(&self, f: impl FnOnce(&mut [u8]) -> R) -> R {
        unsafe {
            let p = ffi::ane_surface_lock(self.raw(), 0) as *mut u8;
            let r = f(std::slice::from_raw_parts_mut(p, self.byte_len()));
            ffi::ane_surface_unlock(self.raw(), 0);
            r
        }
    }
    /// Lock for CPU reads and expose the raw padded bytes.
    pub fn read_with<R>(&self, f: impl FnOnce(&[u8]) -> R) -> R {
        unsafe {
            let p = ffi::ane_surface_lock(self.raw(), 1) as *const u8;
            let r = f(std::slice::from_raw_parts(p, self.byte_len()));
            ffi::ane_surface_unlock(self.raw(), 1);
            r
        }
    }

    /// Copy a dense row-major array of `rows*W` elements in, inserting row padding.
    pub fn write_dense<T: Copy>(&self, src: &[T]) {
        assert_eq!(std::mem::size_of::<T>(), self.dtype.size());
        let rb = self.row_bytes();
        let n = src.len() * std::mem::size_of::<T>();
        assert!(n <= self.rows() * rb, "source larger than tensor");
        let sb = unsafe { std::slice::from_raw_parts(src.as_ptr() as *const u8, n) };
        self.write_with(|dst| {
            if self.is_dense() {
                dst[..n].copy_from_slice(sb);
                dst[n..].fill(0);
            } else {
                dst.fill(0);
                for (r, chunk) in sb.chunks(rb).enumerate() {
                    dst[r * self.row_stride..r * self.row_stride + chunk.len()].copy_from_slice(chunk);
                }
            }
        })
    }

    /// Copy out to a dense row-major array, dropping row padding.
    pub fn read_dense<T: Copy + Default>(&self) -> Vec<T> {
        assert_eq!(std::mem::size_of::<T>(), self.dtype.size());
        let per_row = self.shape[3];
        let mut out = vec![T::default(); self.rows() * per_row];
        let rb = self.row_bytes();
        self.read_with(|src| {
            let ob = unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut u8, out.len() * self.dtype.size()) };
            if self.is_dense() {
                ob.copy_from_slice(&src[..ob.len()]);
            } else {
                for r in 0..self.rows() {
                    ob[r * rb..(r + 1) * rb].copy_from_slice(&src[r * self.row_stride..r * self.row_stride + rb]);
                }
            }
        });
        out
    }
}

impl Drop for AneTensor {
    fn drop(&mut self) {
        unsafe { ffi::ane_surface_release(self.raw()) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn stride_rule() {
        assert_eq!(row_stride_bytes(100, DType::F16), 256);
        assert_eq!(row_stride_bytes(32, DType::F16), 64);
        assert_eq!(row_stride_bytes(33, DType::F16), 128);
        assert_eq!(row_stride_bytes(16384, DType::U8), 16384);
    }
}
