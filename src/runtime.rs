//! Compile/load MIL programs on the ANE and run them on IOSurface tensors.
//!
//! Compiled programs are cached in-process keyed by a hash of (MIL text, weight
//! blob), so a repeated query/scan/filter signature costs one dispatch
//! (~0.13-0.2 ms fixed overhead) instead of a 15 ms-6 s compile.
use crate::ffi;
use crate::iosurface_tensor::AneTensor;
use crate::{Error, Result};
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, Mutex, OnceLock};

pub struct AneProgram {
    model: *mut ffi::ane_model,
    pub compile_ms: f64,
    pub load_ms: f64,
    pub mil: String,
    pub weight_bytes: usize,
}
unsafe impl Send for AneProgram {}
unsafe impl Sync for AneProgram {}

impl Drop for AneProgram {
    fn drop(&mut self) {
        unsafe { ffi::ane_model_free(self.model) }
    }
}

fn cache() -> &'static Mutex<HashMap<(u64, usize, usize), Arc<AneProgram>>> {
    static C: OnceLock<Mutex<HashMap<(u64, usize, usize), Arc<AneProgram>>>> = OnceLock::new();
    C.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Check that the private ANE framework resolves on this machine.
pub fn available() -> Result<()> {
    let mut err = [0u8; 512];
    let rc = unsafe { ffi::ane_available(err.as_mut_ptr() as *mut _, err.len()) };
    if rc == 0 {
        Ok(())
    } else {
        Err(Error::Ane(ffi::err_string(&err)))
    }
}

impl AneProgram {
    /// Compile (or fetch from cache) a MIL program + weight blob.
    pub fn compile(mil: &str, blob: &[u8]) -> Result<Arc<AneProgram>> {
        let mut h = std::collections::hash_map::DefaultHasher::new();
        mil.hash(&mut h);
        blob.hash(&mut h);
        let key = (h.finish(), mil.len(), blob.len());
        if let Some(p) = cache().lock().unwrap().get(&key) {
            return Ok(p.clone());
        }
        let p = Arc::new(Self::compile_uncached(mil, blob)?);
        cache().lock().unwrap().insert(key, p.clone());
        Ok(p)
    }

    pub fn compile_uncached(mil: &str, blob: &[u8]) -> Result<AneProgram> {
        available()?;
        let mut err = vec![0u8; 8192];
        let (mut c, mut l) = (0.0, 0.0);
        let m = unsafe {
            ffi::ane_model_compile(
                mil.as_ptr() as *const _,
                mil.len(),
                if blob.is_empty() { std::ptr::null() } else { blob.as_ptr() as *const _ },
                blob.len(),
                &mut c,
                &mut l,
                err.as_mut_ptr() as *mut _,
                err.len(),
            )
        };
        if m.is_null() {
            return Err(Error::Compile(ffi::err_string(&err)));
        }
        Ok(AneProgram { model: m, compile_ms: c, load_ms: l, mil: mil.to_string(), weight_bytes: blob.len() })
    }

    /// Bind input/output tensors (in MIL argument / return order).
    pub fn bind(self: &Arc<Self>, ins: &[&AneTensor], outs: &[&AneTensor]) -> Result<BoundRequest> {
        let i: Vec<ffi::IOSurfaceRef> = ins.iter().map(|t| t.raw()).collect();
        let o: Vec<ffi::IOSurfaceRef> = outs.iter().map(|t| t.raw()).collect();
        let r = unsafe { ffi::ane_request_new(self.model, i.as_ptr(), i.len() as i32, o.as_ptr(), o.len() as i32) };
        if r.is_null() {
            return Err(Error::Ane("_ANERequest creation failed".into()));
        }
        Ok(BoundRequest { req: r, _prog: self.clone() })
    }
}

/// A request with surfaces bound; `run` executes synchronously on the ANE.
pub struct BoundRequest {
    req: *mut ffi::ane_request,
    _prog: Arc<AneProgram>,
}
unsafe impl Send for BoundRequest {}

impl BoundRequest {
    pub fn run(&self) -> Result<()> {
        let mut err = [0u8; 2048];
        let rc = unsafe { ffi::ane_request_run(self.req, err.as_mut_ptr() as *mut _, err.len()) };
        if rc == 0 {
            Ok(())
        } else {
            Err(Error::Ane(ffi::err_string(&err)))
        }
    }
}

impl Drop for BoundRequest {
    fn drop(&mut self) {
        unsafe { ffi::ane_request_free(self.req) }
    }
}

/// H11ANE IOPowerManagement CurrentPowerState: 0 = power-gated, 1 = on. -1 if unknown.
pub fn ane_power_state() -> i32 {
    unsafe { ffi::ane_power_state() }
}
