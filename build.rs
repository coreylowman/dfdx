//! This script links to Intel MKL when one of the following features is selected:
//!
//! 1. `mkl-static-iomp`: staticly link, use threaded libraries
//! 2. `mkl-static-seq`: staticly link, use sequential libraries
//! 3. `mkl-dynamic-iomp`: staticly link, use threaded libraries
//! 4. `mkl-dynamic-seq`: staticly link, use sequential libraries
//!
//! As described [by Intel here](https://www.intel.com/content/www/us/en/developer/articles/technical/intel-math-kernel-library-intel-mkl-and-pkg-config-tool.html).
//!
//! **NOTE** lp64 vs ilp64 is chosen based on [target_pointer_width](https://doc.rust-lang.org/reference/conditional-compilation.html#target_pointer_width) cfg variable.
//!
//! # Adding a new system
//!
//! To add a new target system, the following blocks need to exist:
//!
//! - `REDIST_DIRS`: This should contain directories relative to `ONEAPI_ROOT` environment variable that contain the
//!     shared libraries. `main()` will check if any of these directories are not on the `PATH` environment variable and crash if not.
//! - `LINK_DIRS`: The directory where `.lib` files are. `main()` will output a `cargo:rustc-link-search` for each of these.
//! - `LINK_LIBS`: The names of the `.lib` files to link. `main()` will output a `cargo:rustc-link-lib` for each of these.
//!

//! # Supported systems
//!
//! - [x] Windows 32 bit
//! - [x] Windows 64 bit
//! - [ ] Linux 32 bit
//! - [ ] Linux 64 bit
//! - [ ] MacOS 32 bit
//! - [ ] MacOS 64 bit

#[cfg(not(any(target_os = "linux", target_os = "windows", target_os = "macos")))]
pub const UNSUPPORTED_OS_ERROR: _ =
    "Target OS is not supported. Please contact me so you can help me do this!";

#[cfg(target_os = "linux")]
pub const LINUX_NEEDS_TESTING_ERROR: _ =
    "Linux targets are not supported/tested yet. Please contact me so you can help me do this!";

#[cfg(target_os = "macos")]
pub const MACOS_NEEDS_TESTING_ERROR: _ =
    "MacOS targets are not supported/tested yet. Please contact me so you can help me do this!";

#[cfg(any(feature = "mkl-static-iomp", feature = "mkl-static-seq"))]
pub const LINK_TYPE: &str = "static";

#[cfg(any(feature = "mkl-dynamic-iomp", feature = "mkl-dynamic-seq"))]
pub const LINK_TYPE: &str = "dylib";

#[cfg(all(target_os = "windows", target_pointer_width = "32"))]
pub const REDIST_DIRS: &[&str] = &[
    "compiler/latest/windows/redist/ia32_win/compiler",
    "mkl/latest/redist/ia32",
];

#[cfg(all(target_os = "windows", target_pointer_width = "64"))]
pub const REDIST_DIRS: &[&str] = &[
    "compiler/latest/windows/redist/intel64_win/compiler",
    "mkl/latest/redist/intel64",
];

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

#[cfg(all(
    feature = "mkl-static-iomp",
    target_os = "windows",
    target_pointer_width = "64"
))]
pub const LINK_LIBS: &[&str] = &[
    "mkl_core",
    "mkl_intel_lp64",
    "mkl_intel_thread",
    "libiomp5md",
];

#[cfg(all(
    feature = "mkl-static-seq",
    target_os = "windows",
    target_pointer_width = "64",
))]
pub const LINK_LIBS: &[&str] = &["mkl_core", "mkl_intel_lp64", "mkl_sequential"];

#[cfg(all(
    feature = "mkl-static-iomp",
    target_os = "windows",
    target_pointer_width = "32",
))]
pub const LINK_LIBS: &[&str] = &[
    "mkl_core",
    "mkl_intel_ilp64",
    "mkl_intel_thread",
    "libiomp5md",
];

#[cfg(all(
    feature = "mkl-static-seq",
    target_os = "windows",
    target_pointer_width = "32",
))]
pub const LINK_LIBS: &[&str] = &["mkl_core", "mkl_intel_ilp64", "mkl_sequential"];

#[cfg(all(
    feature = "mkl-dynamic-iomp",
    target_os = "windows",
    target_pointer_width = "64",
))]
pub const LINK_LIBS: &[&str] = &[
    "mkl_core_dll",
    "mkl_intel_lp64_dll",
    "mkl_intel_thread_dll",
    "libiomp5md",
];

#[cfg(all(
    feature = "mkl-dynamic-seq",
    target_os = "windows",
    target_pointer_width = "64",
))]
pub const LINK_LIBS: &[&str] = &["mkl_core_dll", "mkl_intel_lp64_dll", "mkl_sequential_dll"];

#[cfg(all(
    feature = "mkl-dynamic-iomp",
    target_os = "windows",
    target_pointer_width = "32",
))]
pub const LINK_LIBS: &[&str] = &[
    "mkl_core_dll",
    "mkl_intel_ilp64_dll",
    "mkl_intel_thread_dll",
    "libiomp5md",
];

#[cfg(all(
    feature = "mkl-dynamic-seq",
    target_os = "windows",
    target_pointer_width = "32",
))]
const LINK_LIBS: &[&str] = &["mkl_core_dll", "mkl_intel_ilp64_dll", "mkl_sequential_dll"];

#[derive(Debug)]
pub enum BuildError {
    OneAPINotFound(std::env::VarError),
    PathNotFound(std::env::VarError),
    RedistDirNotFoundInPath(String),
}

fn main() -> Result<(), BuildError> {
    #[cfg(any(
        feature = "mkl-static-iomp",
        feature = "mkl-static-seq",
        feature = "mkl-dynamic-iomp",
        feature = "mkl-dynamic-seq"
    ))]
    {
        let root = std::env::var("ONEAPI_ROOT").map_err(BuildError::OneAPINotFound)?;
        let path = std::env::var("PATH").map_err(BuildError::PathNotFound)?;

        let path = path.replace('\\', "/");
        for redist_dir in REDIST_DIRS {
            if !path.contains(redist_dir) {
                return Err(BuildError::RedistDirNotFoundInPath(redist_dir.to_string()));
            }
        }

        let root: std::path::PathBuf = root.into();

        for lib_dir in LINK_DIRS {
            println!("cargo:rustc-link-search={}", root.join(lib_dir).display());
        }

        for lib_name in LINK_LIBS {
            println!("cargo:rustc-link-lib={}={}", LINK_TYPE, lib_name);
        }
    }

    Ok(())
}
