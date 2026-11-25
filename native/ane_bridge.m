// ane_bridge.m — see ane_bridge.h. Compiled with -fobjc-arc.
#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#import <IOKit/IOKitLib.h>
#import <objc/message.h>
#include <dlfcn.h>
#include <stdio.h>
#include "ane_bridge.h"

static const unsigned kQoS = 21; // NSQualityOfServiceUserInteractive-ish; what CoreML uses

struct ane_model { void *model; NSString *stageDir; };
struct ane_request { void *req; void *model; };

static Class cDesc, cModel, cReq, cIO;

static void seterr(char *err, size_t n, NSString *msg) {
    if (err && n) snprintf(err, n, "%s", msg.UTF8String ?: "unknown error");
}

static int resolve(char *err, size_t n) {
    static int state = 0; // 0 unknown, 1 ok, -1 bad
    if (state) return state == 1 ? 0 : -1;
    if (!dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW)) {
        seterr(err, n, @"dlopen AppleNeuralEngine.framework failed"); state = -1; return -1;
    }
    cDesc = NSClassFromString(@"_ANEInMemoryModelDescriptor");
    cModel = NSClassFromString(@"_ANEInMemoryModel");
    cReq = NSClassFromString(@"_ANERequest");
    cIO = NSClassFromString(@"_ANEIOSurfaceObject");
    if (!cDesc || !cModel || !cReq || !cIO) { seterr(err, n, @"private ANE classes missing"); state = -1; return -1; }
    state = 1; return 0;
}

int ane_available(char *err, size_t errlen) { return resolve(err, errlen); }

ane_model *ane_model_compile(const char *mil, size_t mil_len, const void *weights, size_t wlen,
                             double *compile_ms, double *load_ms, char *err, size_t errlen) {
    if (resolve(err, errlen)) return NULL;
    @autoreleasepool {
        NSData *milData = [NSData dataWithBytes:mil length:mil_len];
        NSData *blob = weights && wlen ? [NSData dataWithBytes:weights length:wlen] : [NSData data];
        NSDictionary *wdict = wlen ? @{@"@model_path/weights/weight.bin": @{@"offset": @0, @"data": blob}} : @{};
        id desc = ((id(*)(id, SEL, id, id, id))objc_msgSend)(cDesc, @selector(modelWithMILText:weights:optionsPlist:),
                                                             milData, wdict, nil);
        if (!desc) { seterr(err, errlen, @"descriptor creation failed"); return NULL; }
        id m = ((id(*)(id, SEL, id))objc_msgSend)(cModel, @selector(inMemoryModelWithDescriptor:), desc);
        if (!m) { seterr(err, errlen, @"inMemoryModelWithDescriptor failed"); return NULL; }

        // The compiler service (aned) verifies a bundle at localModelPath on disk,
        // so the in-memory program must be staged there. We delete it after load.
        NSString *lp = [m valueForKey:@"localModelPath"];
        NSFileManager *fm = NSFileManager.defaultManager;
        [fm createDirectoryAtPath:[lp stringByAppendingPathComponent:@"weights"]
      withIntermediateDirectories:YES attributes:nil error:nil];
        [milData writeToFile:[lp stringByAppendingPathComponent:@"model.mil"] atomically:NO];
        if (wlen) [blob writeToFile:[lp stringByAppendingPathComponent:@"weights/weight.bin"] atomically:NO];

        NSError *e = nil;
        CFAbsoluteTime t0 = CFAbsoluteTimeGetCurrent();
        BOOL ok = ((BOOL(*)(id, SEL, unsigned, id, NSError **))objc_msgSend)(m, @selector(compileWithQoS:options:error:), kQoS, @{}, &e);
        CFAbsoluteTime t1 = CFAbsoluteTimeGetCurrent();
        if (!ok) { seterr(err, errlen, [NSString stringWithFormat:@"compile: %@", e]); [fm removeItemAtPath:lp error:nil]; return NULL; }
        ok = ((BOOL(*)(id, SEL, unsigned, id, NSError **))objc_msgSend)(m, @selector(loadWithQoS:options:error:), kQoS, @{}, &e);
        CFAbsoluteTime t2 = CFAbsoluteTimeGetCurrent();
        if (!ok) { seterr(err, errlen, [NSString stringWithFormat:@"load: %@", e]); [fm removeItemAtPath:lp error:nil]; return NULL; }
        if (compile_ms) *compile_ms = (t1 - t0) * 1e3;
        if (load_ms) *load_ms = (t2 - t1) * 1e3;
        ane_model *am = calloc(1, sizeof *am);
        am->model = (__bridge_retained void *)m;
        am->stageDir = lp;
        // compiled program now lives in aned; drop the staged source bundle
        [fm removeItemAtPath:lp error:nil];
        return am;
    }
}

void ane_model_free(ane_model *am) {
    if (!am) return;
    @autoreleasepool {
        id m = CFBridgingRelease(am->model);
        NSError *e = nil;
        ((BOOL(*)(id, SEL, unsigned, NSError **))objc_msgSend)(m, @selector(unloadWithQoS:error:), kQoS, &e);
        am->stageDir = nil;
    }
    free(am);
}

ane_request *ane_request_new(ane_model *am, IOSurfaceRef *ins, int nin, IOSurfaceRef *outs, int nout) {
    if (!am) return NULL;
    @autoreleasepool {
        NSMutableArray *ia = [NSMutableArray array], *ii = [NSMutableArray array];
        NSMutableArray *oa = [NSMutableArray array], *oi = [NSMutableArray array];
        for (int i = 0; i < nin; i++) {
            [ia addObject:((id(*)(id, SEL, IOSurfaceRef))objc_msgSend)(cIO, @selector(objectWithIOSurface:), ins[i])];
            [ii addObject:@(i)];
        }
        for (int i = 0; i < nout; i++) {
            [oa addObject:((id(*)(id, SEL, IOSurfaceRef))objc_msgSend)(cIO, @selector(objectWithIOSurface:), outs[i])];
            [oi addObject:@(i)];
        }
        id req = ((id(*)(id, SEL, id, id, id, id, id, id, id))objc_msgSend)(
            cReq, @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
            ia, ii, oa, oi, nil, nil, @0);
        if (!req) return NULL;
        ane_request *r = calloc(1, sizeof *r);
        r->req = (__bridge_retained void *)req;
        r->model = am->model;
        return r;
    }
}

void ane_request_free(ane_request *r) {
    if (!r) return;
    CFBridgingRelease(r->req);
    free(r);
}

int ane_request_run(ane_request *r, char *err, size_t errlen) {
    @autoreleasepool {
        id m = (__bridge id)r->model, req = (__bridge id)r->req;
        NSError *e = nil;
        BOOL ok = ((BOOL(*)(id, SEL, unsigned, id, id, NSError **))objc_msgSend)(
            m, @selector(evaluateWithQoS:options:request:error:), kQoS, @{}, req, &e);
        if (!ok) { seterr(err, errlen, [NSString stringWithFormat:@"evaluate: %@", e]); return -1; }
        return 0;
    }
}

IOSurfaceRef ane_surface_create(size_t bytes) {
    if (bytes == 0) bytes = 64;
    NSDictionary *p = @{(id)kIOSurfaceWidth: @(bytes), (id)kIOSurfaceHeight: @1,
                        (id)kIOSurfaceBytesPerElement: @1, (id)kIOSurfaceBytesPerRow: @(bytes),
                        (id)kIOSurfaceAllocSize: @(bytes), (id)kIOSurfacePixelFormat: @0};
    return IOSurfaceCreate((__bridge CFDictionaryRef)p);
}
void ane_surface_release(IOSurfaceRef s) { if (s) CFRelease(s); }
void *ane_surface_lock(IOSurfaceRef s, int ro) {
    IOSurfaceLock(s, ro ? kIOSurfaceLockReadOnly : 0, NULL);
    return IOSurfaceGetBaseAddress(s);
}
void ane_surface_unlock(IOSurfaceRef s, int ro) { IOSurfaceUnlock(s, ro ? kIOSurfaceLockReadOnly : 0, NULL); }
size_t ane_surface_size(IOSurfaceRef s) { return IOSurfaceGetAllocSize(s); }

int ane_power_state(void) {
    io_service_t svc = IOServiceGetMatchingService(kIOMainPortDefault, IOServiceMatching("H11ANEIn"));
    if (!svc) return -1;
    int st = -1;
    CFTypeRef pm = IORegistryEntryCreateCFProperty(svc, CFSTR("IOPowerManagement"), kCFAllocatorDefault, 0);
    if (pm && CFGetTypeID(pm) == CFDictionaryGetTypeID()) {
        CFNumberRef n = CFDictionaryGetValue(pm, CFSTR("CurrentPowerState"));
        if (n) CFNumberGetValue(n, kCFNumberIntType, &st);
    }
    if (pm) CFRelease(pm);
    IOObjectRelease(svc);
    return st;
}

// ---------------------------------------------------------------- telemetry
#include <libproc.h>
#include <sys/resource.h>
#include <unistd.h>

typedef CFDictionaryRef (*IORCopyGroup)(CFStringRef, CFStringRef, uint64_t, uint64_t, uint64_t);
typedef void *(*IORCreateSub)(void *, CFMutableDictionaryRef, CFMutableDictionaryRef *, uint64_t, CFTypeRef);
typedef CFDictionaryRef (*IORSamples)(void *, CFMutableDictionaryRef, CFTypeRef);
typedef CFStringRef (*IORUnit)(CFDictionaryRef);

int ane_energy_read(ane_energy_sample *out) {
    static void *sub = NULL;
    static CFMutableDictionaryRef subd = NULL;
    static IORSamples smp = NULL;
    static IORUnit unit = NULL;
    static int init = 0;
    if (!init) {
        init = 1;
        void *h = dlopen("/usr/lib/libIOReport.dylib", RTLD_NOW);
        if (!h) return -1;
        IORCopyGroup cg = (IORCopyGroup)dlsym(h, "IOReportCopyChannelsInGroup");
        IORCreateSub cs = (IORCreateSub)dlsym(h, "IOReportCreateSubscription");
        smp = (IORSamples)dlsym(h, "IOReportCreateSamples");
        unit = (IORUnit)dlsym(h, "IOReportChannelGetUnitLabel");
        if (!cg || !cs || !smp || !unit) return -1;
        CFDictionaryRef ch = cg(CFSTR("Energy Model"), NULL, 0, 0, 0);
        if (!ch) return -1;
        CFMutableDictionaryRef mch = CFDictionaryCreateMutableCopy(NULL, 0, ch);
        CFRelease(ch);
        sub = cs(NULL, mch, &subd, 0, NULL);
    }
    if (!sub) return -1;
    @autoreleasepool {
        NSDictionary *d = (__bridge_transfer NSDictionary *)smp(sub, subd, NULL);
        memset(out, 0, sizeof *out);
        out->t_s = CFAbsoluteTimeGetCurrent();
        for (NSDictionary *c in d[@"IOReportChannels"]) {
            NSString *n = [c[@"LegendChannel"] lastObject];
            NSData *raw = c[@"RawElements"];
            if (![n isKindOfClass:NSString.class] || raw.length < 40) continue;
            // IOReportSimpleGetIntegerValue/IOReportCreateSamplesDelta return 0 on
            // this OS build; the simple-format value lives at byte 32 of RawElements.
            int64_t v = *(const int64_t *)((const uint8_t *)raw.bytes + 32);
            NSString *u = (__bridge NSString *)unit((__bridge CFDictionaryRef)c);
            double sc = [u isEqual:@"mJ"] ? 1e-3 : [u isEqual:@"uJ"] ? 1e-6 : [u isEqual:@"nJ"] ? 1e-9 : 0;
            double j = v * sc;
            if ([n isEqual:@"CPU Energy"]) out->cpu_j = j;
            else if ([n isEqual:@"GPU Energy"]) out->gpu_j = j;
            else if ([n hasPrefix:@"ANE"]) out->ane_j += j;
            else if ([n hasPrefix:@"DRAM"]) out->dram_j += j;
        }
    }
    return 0;
}

int ane_proc_energy(int pid, double *energy_j, double *cpu_s) {
    struct rusage_info_v6 ri;
    if (proc_pid_rusage(pid ? pid : getpid(), RUSAGE_INFO_V6, (rusage_info_t *)&ri) != 0) return -1;
    if (energy_j) *energy_j = ri.ri_energy_nj * 1e-9;
    if (cpu_s) {
        mach_timebase_info_data_t tb; mach_timebase_info(&tb);
        *cpu_s = (double)(ri.ri_user_time + ri.ri_system_time) * tb.numer / tb.denom * 1e-9;
    }
    return 0;
}
