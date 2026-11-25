fn main() {
    println!("cargo:rerun-if-changed=native/ane_bridge.m");
    println!("cargo:rerun-if-changed=native/ane_bridge.h");
    cc::Build::new()
        .file("native/ane_bridge.m")
        .flag("-fobjc-arc")
        .flag("-fmodules")
        .opt_level(2)
        .compile("ane_bridge");
    for fw in ["Foundation", "IOSurface", "IOKit", "CoreFoundation", "Accelerate"] {
        println!("cargo:rustc-link-lib=framework={fw}");
    }
}
