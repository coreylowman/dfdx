//! This script links to Intel MKL when one of the following features is selected:
//!
//! 1. `mkl-static-iomp`: staticly link, use threaded libraries
//! 2. `mkl-static-seq`: staticly link, use sequential libraries
//! 3. `mkl-dynamic-iomp`: dynamic link, use threaded libraries
//! 4. `mkl-dynamic-seq`: dynamic link, use sequential libraries
//!
//! As described [by Intel here](https://www.intel.com/content/www/us/en/developer/articles/technical/intel-math-kernel-library-intel-mkl-and-pkg-config-tool.html).
//!
//! **NOTE** lp64 vs ilp64 is chosen based on [target_pointer_width](https://doc.rust-lang.org/reference/conditional-compilation.html#target_pointer_width) cfg variable.
//!
//! # Adding a new system
//!
//! To add a new target system, the following blocks need to exist:
//!
//! - `SHARED_LIB_DIRS`: This should contain directories relative to `ONEAPI_ROOT` environment variable that contain the
//!     shared libraries. `main()` will check if any of these directories are not on the `PATH` environment variable and crash if not.
//! - `LINK_DIRS`: The directory where `.lib` files are. `main()` will output a `cargo:rustc-link-search` for each of these.
//!
//! # Supported systems
//!
//! - [x] Windows 32 bit
//! - [x] Windows 64 bit
//! - [x] Linux 32 bit
//! - [x] Linux 64 bit
//! - [ ] MacOS 32 bit
//! - [ ] MacOS 64 bit
//!
//! This script also creates a "nightly" feature if the crate is compiled on a nightly branch
use rustc_version::{version_meta, Channel};

pub const MKL_VERSION: &str = "2022.1.0";
pub const STATIC: bool = cfg!(feature = "mkl-static-seq") || cfg!(feature = "mkl-static-iomp");
pub const DYNAMIC: bool = cfg!(feature = "mkl-dynamic-seq") || cfg!(feature = "mkl-dynamic-iomp");
pub const SEQUENTIAL: bool = cfg!(feature = "mkl-static-seq") || cfg!(feature = "mkl-dynamic-seq");
pub const THREADED: bool = cfg!(feature = "mkl-static-iomp") || cfg!(feature = "mkl-dynamic-iomp");
pub const MKL: bool = (STATIC || DYNAMIC) && (SEQUENTIAL || THREADED);

pub const LINK_TYPE: &str = if STATIC { "static" } else { "dylib" };
pub const LIB_POSTFIX: &str = if cfg!(windows) && DYNAMIC { "_dll" } else { "" };
pub const LD_DIR: &str = if cfg!(windows) {
    "PATH"
} else {
    "LD_LIBRARY_PATH"
};

pub const DEFAULT_ONEAPI_ROOT: &str = if cfg!(windows) {
    "C:/Program Files (x86)/Intel/oneAPI/"
} else {
    "/opt/intel/oneapi/"
};

pub const MKL_CORE: &str = "mkl_core";
pub const MKL_THREAD: &str = if SEQUENTIAL {
    "mkl_sequential"
} else {
    "mkl_intel_thread"
};
pub const THREADING_LIB: &str = if cfg!(windows) { "libiomp5md" } else { "iomp5" };
pub const MKL_INTERFACE: &str = if cfg!(target_pointer_width = "32") {
    "mkl_intel_ilp64"
} else {
    "mkl_intel_lp64"
};

#[cfg(not(any(target_os = "linux", target_os = "windows", target_os = "macos")))]
pub const UNSUPPORTED_OS_ERROR: _ = "Target OS is not supported. Please contact me";

#[cfg(all(target_os = "windows", target_pointer_width = "32"))]
pub const LINK_DIRS: &[&str] = &[
    "compiler/latest/windows/compiler/lib/ia32_win",
    "mkl/latest/lib/ia32",
];

#[cfg(all(target_os = "windows", target_pointer_width = "64"))]
pub const LINK_DIRS: &[&str] = &[
    "compiler/latest/windows/compiler/lib/intel64_win",
    "mkl/latest/lib/intel64",
];

#[cfg(all(target_os = "windows", target_pointer_width = "32"))]
pub const SHARED_LIB_DIRS: &[&str] = &[
    "compiler/latest/windows/redist/ia32_win/compiler",
    "mkl/latest/redist/ia32",
];

#[cfg(all(target_os = "windows", target_pointer_width = "64"))]
pub const SHARED_LIB_DIRS: &[&str] = &[
    "compiler/latest/windows/redist/intel64_win/compiler",
    "mkl/latest/redist/intel64",
];

#[cfg(all(target_os = "linux", target_pointer_width = "32"))]
pub const LINK_DIRS: &[&str] = &[
    "compiler/latest/linux/compiler/lib/ia32_lin",
    "mkl/latest/lib/ia32",
];

#[cfg(all(target_os = "linux", target_pointer_width = "64"))]
pub const LINK_DIRS: &[&str] = &[
    "compiler/latest/linux/compiler/lib/intel64_lin",
    "mkl/latest/lib/intel64",
];

#[cfg(target_os = "linux")]
pub const SHARED_LIB_DIRS: &[&str] = LINK_DIRS;

#[derive(Debug)]
pub enum BuildError {
    OneAPINotFound(std::path::PathBuf),
    OneAPINotADir(std::path::PathBuf),
    PathNotFound(std::env::VarError),
    AddSharedLibDirToPath(String),
}

fn main() -> Result<(), BuildError> {
    println!("cargo:rerun-if-changed=build.rs");

    #[cfg(any(
        feature = "mkl-static-iomp",
        feature = "mkl-static-seq",
        feature = "mkl-dynamic-iomp",
        feature = "mkl-dynamic-seq"
    ))]
    {
        let root = std::env::var("ONEAPI_ROOT").unwrap_or_else(|_| DEFAULT_ONEAPI_ROOT.to_string());
        println!("Using '{root}' as ONEAPI_ROOT");

        let path = std::env::var(LD_DIR).map_err(BuildError::PathNotFound)?;

        if DYNAMIC {
            // check to make sure that things in `SHARED_LIB_DIRS` are in `$LD_DIR`.
            let path = path.replace('\\', "/");
            for shared_lib_dir in SHARED_LIB_DIRS {
                let versioned_dir = shared_lib_dir.replace("latest", MKL_VERSION);

                println!("Checking that '{shared_lib_dir}' or '{versioned_dir}' is in {LD_DIR}");
                if !path.contains(shared_lib_dir) && !path.contains(&versioned_dir) {
                    let suggested_cmd = if cfg!(windows) {
                        format!("{root}/setvars.bat")
                    } else {
                        format!("source {root}/setvars.sh")
                    };
                    println!("'{shared_lib_dir}' not found in library path. Run `{suggested_cmd}`");
                    return Err(BuildError::AddSharedLibDirToPath(
                        shared_lib_dir.to_string(),
                    ));
                }
            }
        }

        let root: std::path::PathBuf = root.into();

        if !root.exists() {
            return Err(BuildError::OneAPINotFound(root));
        }
        if !root.is_dir() {
            return Err(BuildError::OneAPINotADir(root));
        }

        for rel_lib_dir in LINK_DIRS {
            let lib_dir = root.join(rel_lib_dir);
            println!("cargo:rustc-link-search={}", lib_dir.display());
        }

        println!("cargo:rustc-link-lib={LINK_TYPE}={MKL_INTERFACE}{LIB_POSTFIX}");
        println!("cargo:rustc-link-lib={LINK_TYPE}={MKL_THREAD}{LIB_POSTFIX}");
        println!("cargo:rustc-link-lib={LINK_TYPE}={MKL_CORE}{LIB_POSTFIX}");
        if THREADED {
            println!("cargo:rustc-link-lib=dylib={THREADING_LIB}");
        }

        if !cfg!(windows) {
            println!("cargo:rustc-link-lib=pthread");
            println!("cargo:rustc-link-lib=m");
            println!("cargo:rustc-link-lib=dl");
        }
    }

    // If on nightly, enable "nightly" feature
    if version_meta().unwrap().channel == Channel::Nightly {
        println!("cargo:rustc-cfg=feature=\"nightly\"");
    }

    Ok(())
}

// This section creates a feature "nightly" enabled if built on a nightly branch
