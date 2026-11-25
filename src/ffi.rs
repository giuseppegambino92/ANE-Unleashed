//! Raw bindings to `native/ane_bridge.m` (private AppleNeuralEngine.framework SPI)
//! plus the handful of IOReport / libproc symbols used for telemetry.
#![allow(non_camel_case_types)]
use std::os::raw::{c_char, c_int, c_void};

pub type IOSurfaceRef = *mut c_void;
#[repr(C)]
pub struct ane_model {
    _p: [u8; 0],
}
#[repr(C)]
pub struct ane_request {
    _p: [u8; 0],
}

extern "C" {
    pub fn ane_available(err: *mut c_char, errlen: usize) -> c_int;
    pub fn ane_model_compile(
        mil: *const c_char,
        mil_len: usize,
        weights: *const c_void,
        weights_len: usize,
        compile_ms: *mut f64,
        load_ms: *mut f64,
        err: *mut c_char,
        errlen: usize,
    ) -> *mut ane_model;
    pub fn ane_model_free(m: *mut ane_model);
    pub fn ane_request_new(
        m: *mut ane_model,
        ins: *const IOSurfaceRef,
        nin: c_int,
        outs: *const IOSurfaceRef,
        nout: c_int,
    ) -> *mut ane_request;
    pub fn ane_request_free(r: *mut ane_request);
    pub fn ane_request_run(r: *mut ane_request, err: *mut c_char, errlen: usize) -> c_int;
    pub fn ane_surface_create(bytes: usize) -> IOSurfaceRef;
    pub fn ane_surface_release(s: IOSurfaceRef);
    pub fn ane_surface_lock(s: IOSurfaceRef, readonly: c_int) -> *mut c_void;
    pub fn ane_surface_unlock(s: IOSurfaceRef, readonly: c_int);
    pub fn ane_surface_size(s: IOSurfaceRef) -> usize;
    pub fn ane_power_state() -> c_int;
}

pub(crate) fn err_string(buf: &[u8]) -> String {
    let end = buf.iter().position(|&b| b == 0).unwrap_or(buf.len());
    String::from_utf8_lossy(&buf[..end]).into_owned()
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct ane_energy_sample {
    pub cpu_j: f64,
    pub gpu_j: f64,
    pub ane_j: f64,
    pub dram_j: f64,
    pub t_s: f64,
}

extern "C" {
    pub fn ane_energy_read(out: *mut ane_energy_sample) -> c_int;
    pub fn ane_proc_energy(pid: c_int, energy_j: *mut f64, cpu_s: *mut f64) -> c_int;
}
