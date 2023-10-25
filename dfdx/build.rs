fn main() {
    println!("cargo:rerun-if-changed=build.rs");

    // If on nightly, enable "nightly" feature
    maybe_enable_nightly();
}

fn maybe_enable_nightly() {
    let cmd = std::env::var_os("RUSTC").unwrap_or_else(|| std::ffi::OsString::from("rustc"));
    let out = std::process::Command::new(cmd).arg("-vV").output().unwrap();
    assert!(out.status.success());
    if std::str::from_utf8(&out.stdout)
        .unwrap()
        .contains("nightly")
    {
        println!("cargo:rustc-cfg=feature=\"nightly\"");
    }
}
