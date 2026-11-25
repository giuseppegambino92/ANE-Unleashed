// ane_bridge.h — minimal C ABI over Apple's private AppleNeuralEngine.framework
// in-memory MIL path (_ANEInMemoryModelDescriptor / _ANEInMemoryModel / _ANERequest).
//
// Everything here is undocumented SPI. It works on the macOS build it was tested
// on (macOS 27.2.1, M3 Max, ANE v176) and may break on any OS update.
#pragma once
#include <stddef.h>
#include <stdint.h>
#include <IOSurface/IOSurfaceRef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ane_model ane_model;
typedef struct ane_request ane_request;

// Returns 0 if the private framework and all required classes resolve.
int ane_available(char *err, size_t errlen);

// Compile + load a MIL program. `weights` is a complete weight.bin blob
// (64-byte file header + 0xDEADBEEF chunks) referenced from the MIL text as
// "@model_path/weights/weight.bin"; may be NULL/0 when the program has no blob.
// On failure returns NULL and writes a message into err.
ane_model *ane_model_compile(const char *mil, size_t mil_len,
                             const void *weights, size_t weights_len,
                             double *compile_ms, double *load_ms,
                             char *err, size_t errlen);
void ane_model_free(ane_model *m);

// Bind IOSurfaces to a model's inputs/outputs (in MIL argument / return order).
ane_request *ane_request_new(ane_model *m, IOSurfaceRef *ins, int nin,
                             IOSurfaceRef *outs, int nout);
void ane_request_free(ane_request *r);

// Synchronously execute one request on the ANE. Returns 0 on success.
int ane_request_run(ane_request *r, char *err, size_t errlen);

// IOSurface helpers (plain linear byte buffers; 16 KiB page aligned by IOSurface).
IOSurfaceRef ane_surface_create(size_t bytes);
void ane_surface_release(IOSurfaceRef s);
void *ane_surface_lock(IOSurfaceRef s, int readonly);
void ane_surface_unlock(IOSurfaceRef s, int readonly);
size_t ane_surface_size(IOSurfaceRef s);

// H11ANE IOPowerManagement CurrentPowerState (0 = power-gated, 1 = powered),
// read from the IORegistry without privileges. Returns -1 if not found.
int ane_power_state(void);

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
extern "C" {
#endif
// ---- telemetry -----------------------------------------------------------
// Cumulative energy counters in joules, from IOReport "Energy Model" (no root).
// On macOS 27.2 for unprivileged processes only the GPU counter advances; the
// CPU/ANE/DRAM PMGR counters stay frozen; callers must compare successive
// samples and treat counters that do not advance as unavailable.
typedef struct {
    double cpu_j, gpu_j, ane_j, dram_j;
    double t_s;          // CFAbsoluteTime of the sample
} ane_energy_sample;
int ane_energy_read(ane_energy_sample *out);

// Per-process energy estimate (rusage_info_v6.ri_energy_nj, CPU only), in joules,
// plus user+system CPU time in seconds. pid 0 = self. Returns 0 on success.
int ane_proc_energy(int pid, double *energy_j, double *cpu_s);
#ifdef __cplusplus
}
#endif
