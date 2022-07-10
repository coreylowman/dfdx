//! This is only for the "cblas" feature.
//!
//! If you enable the cblas feature then you must choose one of the following
//! link formats:
//!
//! 1. mkl-static-iomp
//! 2. mkl-static-seq
//! 3. mkl-dynamic-iomp
//! 4. mkl-dynamic-seq
//!
//! **NOTE** lp64 vs ilp64 is chosen based on [target_pointer_width](https://doc.rust-lang.org/reference/conditional-compilation.html#target_pointer_width) cfg variable.
//!
//! As described [by Intel here](https://www.intel.com/content/www/us/en/developer/articles/technical/intel-math-kernel-library-intel-mkl-and-pkg-config-tool.html).
//!
//! To add a new target system, copy one of the link_info modules and wrap it with the corresponding cfg block.
//!
//! # `link_info` blocks
//!
//! Each of these contains various system dependent information on how to link to the MKL libraries:
//!
//! - `LINK_TYPE`: either `STATIC_LINK` or `DYNAMIC_LINK`. This is used for `cargo:rustc-link-lib` output in `main`.
//! - `REDIST_DIRS`: This should contain directories relative to `ONEAPI_ROOT` environment variable that contain the
//!     shared libraries. `main()` will crash if any of these directories are not on the `PATH` environment variable.
//! - `LINK_DIRS`: The directory where `.lib` files are. `main()` will output a `cargo:rustc-link-search` for each of these.
//! - `LINK_LIBS`: The names of the `.lib` files to link. `main()` will output a `cargo:rustc-link-lib` for each of these.

pub const STATIC_LINK: &str = "static";
pub const DYNAMIC_LINK: &str = "dylib";

#[cfg(all(
    feature = "cblas",
    not(any(
        feature = "mkl-static-iomp",
        feature = "mkl-static-seq",
        feature = "mkl-dynamic-iomp",
        feature = "mkl-dynamic-seq",
    ))
))]
mod link_info {
    pub const FEATURE_NOT_SPECIFIED_ERROR: _ = "You need to specify the type of mkl";
    use super::*;
    pub const LINK_TYPE: &str = STATIC_LINK;
    pub const REDIST_DIRS: &[&str] = &[];
    pub const LINK_DIRS: &[&str] = &[];
    pub const LINK_LIBS: &[&str] = &[];
}

#[cfg(all(feature = "cblas", target_os = "linux"))]
mod link_info {
    pub const LINUX_NEEDS_TESTING_ERROR: _ =
        "Linux targets are not supported/tested yet. Please contact me so you can help me do this!";
    use super::*;
    pub const LINK_TYPE: &str = STATIC_LINK;
    pub const REDIST_DIRS: &[&str] = &[];
    pub const LINK_DIRS: &[&str] = &[];
    pub const LINK_LIBS: &[&str] = &[];
}

#[cfg(all(feature = "cblas", target_os = "macos"))]
mod link_info {
    pub const MACOS_NEEDS_TESTING_ERROR: _ =
        "MacOS targets are not supported/tested yet. Please contact me so you can help me do this!";
    use super::*;
    pub const LINK_TYPE: &str = STATIC_LINK;
    pub const REDIST_DIRS: &[&str] = &[];
    pub const LINK_DIRS: &[&str] = &[];
    pub const LINK_LIBS: &[&str] = &[];
}

#[cfg(all(
    feature = "cblas",
    feature = "mkl-static-iomp",
    target_os = "windows",
    target_pointer_width = "64"
))]
mod link_info {
    use super::*;

    pub const LINK_TYPE: &str = STATIC_LINK;
    pub const REDIST_DIRS: &[&str] = &[
        "compiler/latest/windows/redist/intel64_win/compiler",
        "mkl/latest/redist/intel64",
    ];
    pub const LINK_DIRS: &[&str] = &[
        "compiler/latest/windows/compiler/lib/intel64_win",
        "mkl/latest/lib/intel64",
    ];
    pub const LINK_LIBS: &[&str] = &[
        "mkl_core",
        "mkl_intel_lp64",
        "mkl_intel_thread",
        "libiomp5md",
    ];
}

#[cfg(all(
    feature = "cblas",
    feature = "mkl-static-seq",
    target_os = "windows",
    target_pointer_width = "64",
))]
mod link_info {
    use super::*;
    pub const LINK_TYPE: &str = STATIC_LINK;
    pub const REDIST_DIRS: &[&str] = &[
        "compiler/latest/windows/redist/intel64_win/compiler",
        "mkl/latest/redist/intel64",
    ];
    pub const LINK_DIRS: &[&str] = &[
        "compiler/latest/windows/compiler/lib/intel64_win",
        "mkl/latest/lib/intel64",
    ];
    pub const LINK_LIBS: &[&str] = &["mkl_core", "mkl_intel_lp64", "mkl_sequential"];
}

#[cfg(all(
    feature = "cblas",
    feature = "mkl-static-iomp",
    target_os = "windows",
    target_pointer_width = "32",
))]
mod link_info {
    use super::*;

    pub const LINK_TYPE: &str = STATIC_LINK;
    pub const REDIST_DIRS: &[&str] = &[
        "compiler/latest/windows/redist/ia32_win/compiler",
        "mkl/latest/redist/ia32",
    ];
    pub const LINK_DIRS: &[&str] = &[
        "compiler/latest/windows/compiler/lib/ia32_win",
        "mkl/latest/lib/ia32",
    ];
    pub const LINK_LIBS: &[&str] = &[
        "mkl_core",
        "mkl_intel_iilp64",
        "mkl_intel_thread",
        "libiomp5md",
    ];
}

#[cfg(all(
    feature = "cblas",
    feature = "mkl-static-seq",
    target_os = "windows",
    target_pointer_width = "32",
))]
mod link_info {
    use super::*;
    pub const LINK_TYPE: &str = STATIC_LINK;
    pub const REDIST_DIRS: &[&str] = &[
        "compiler/latest/windows/redist/ia32_win/compiler",
        "mkl/latest/redist/ia32",
    ];
    pub const LINK_DIRS: &[&str] = &[
        "compiler/latest/windows/compiler/lib/ia32_win",
        "mkl/latest/lib/ia32",
    ];
    pub const LINK_LIBS: &[&str] = &["mkl_core", "mkl_intel_ilp64", "mkl_sequential"];
}

#[cfg(all(
    feature = "cblas",
    feature = "mkl-dynamic-iomp",
    target_os = "windows",
    target_pointer_width = "64",
))]
mod link_info {
    use super::*;
    pub const LINK_TYPE: &str = DYNAMIC_LINK;
    pub const REDIST_DIRS: &[&str] = &[
        "compiler/latest/windows/redist/intel64_win/compiler",
        "mkl/latest/redist/intel64",
    ];
    pub const LINK_DIRS: &[&str] = &[
        "compiler/latest/windows/compiler/lib/intel64_win",
        "mkl/latest/lib/intel64",
    ];
    pub const LINK_LIBS: &[&str] = &[
        "mkl_core_dll",
        "mkl_intel_lp64_dll",
        "mkl_intel_thread_dll",
        "libiomp5md",
    ];
}

#[cfg(all(
    feature = "cblas",
    feature = "mkl-dynamic-seq",
    target_os = "windows",
    target_pointer_width = "64",
))]
mod link_info {
    use super::*;
    pub const LINK_TYPE: &str = DYNAMIC_LINK;
    pub const REDIST_DIRS: &[&str] = &[
        "compiler/latest/windows/redist/intel64_win/compiler",
        "mkl/latest/redist/intel64",
    ];
    pub const LINK_DIRS: &[&str] = &[
        "compiler/latest/windows/compiler/lib/intel64_win",
        "mkl/latest/lib/intel64",
    ];
    pub const LINK_LIBS: &[&str] = &["mkl_core_dll", "mkl_intel_lp64_dll", "mkl_sequential_dll"];
}

#[cfg(all(
    feature = "cblas",
    feature = "mkl-dynamic-iomp",
    target_os = "windows",
    target_pointer_width = "32",
))]
mod link_info {
    use super::*;
    pub const LINK_TYPE: &str = DYNAMIC_LINK;
    pub const REDIST_DIRS: &[&str] = &[
        "compiler/latest/windows/redist/ia32_win/compiler",
        "mkl/latest/redist/ia32",
    ];
    pub const LINK_DIRS: &[&str] = &[
        "compiler/latest/windows/compiler/lib/ia32_win",
        "mkl/latest/lib/ia32",
    ];
    pub const LINK_LIBS: &[&str] = &[
        "mkl_core_dll",
        "mkl_intel_ilp64_dll",
        "mkl_intel_thread_dll",
        "libiomp5md",
    ];
}

#[cfg(all(
    feature = "cblas",
    feature = "mkl-dynamic-seq",
    target_os = "windows",
    target_pointer_width = "32",
))]
mod link_info {
    use super::*;
    pub const LINK_TYPE: &str = DYNAMIC_LINK;
    pub const REDIST_DIRS: &[&str] = &[
        "compiler/latest/windows/redist/ia32_win/compiler",
        "mkl/latest/redist/ia32",
    ];
    pub const LINK_DIRS: &[&str] = &[
        "compiler/latest/windows/compiler/lib/ia32_win",
        "mkl/latest/lib/ia32",
    ];
    const LINK_LIBS: &[&str] = &["mkl_core_dll", "mkl_intel_ilp64_dll", "mkl_sequential_dll"];
}

#[derive(Debug)]
pub enum BuildError {
    OneAPINotFound(std::env::VarError),
    PathNotFound(std::env::VarError),
    RedistDirNotFoundInPath(String),
}

fn main() -> Result<(), BuildError> {
    #[cfg(feature = "cblas")]
    {
        let root = std::env::var("ONEAPI_ROOT").map_err(BuildError::OneAPINotFound)?;
        let path = std::env::var("PATH").map_err(BuildError::PathNotFound)?;

        let path = path.replace('\\', "/");
        for redist_dir in link_info::REDIST_DIRS {
            if !path.contains(redist_dir) {
                return Err(BuildError::RedistDirNotFoundInPath(redist_dir.to_string()));
            }
        }

        let root: std::path::PathBuf = root.into();

        for lib_dir in link_info::LINK_DIRS {
            println!("cargo:rustc-link-search={}", root.join(lib_dir).display());
        }

        for lib_name in link_info::LINK_LIBS {
            println!("cargo:rustc-link-lib={}={}", link_info::LINK_TYPE, lib_name);
        }
    }

    Ok(())
}
