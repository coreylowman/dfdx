//! This is only for the "cblas" feature.
//!
//! If you enable the cblas feature then you must choose one of the following
//! link formats:
//!
//! 1. mkl-static-lp64-iomp
//! 2. mkl-static-lp64-seq
//! 3. mkl-static-ilp64-iomp
//! 4. mkl-static-ilp64-seq
//! 5. mkl-dynamic-lp64-iomp
//! 6. mkl-dynamic-lp64-seq
//! 7. mkl-dynamic-ilp64-iomp
//! 8. mkl-dynamic-ilp64-seq
//!
//! As described [by Intel here](https://www.intel.com/content/www/us/en/developer/articles/technical/intel-math-kernel-library-intel-mkl-and-pkg-config-tool.html).

pub const STATIC_LINK: &str = "static";
pub const DYNAMIC_LINK: &str = "dylib";

#[cfg(all(
    feature = "cblas",
    not(any(
        feature = "mkl-static-lp64-iomp",
        feature = "mkl-static-lp64-seq",
        feature = "mkl-static-ilp64-iomp",
        feature = "mkl-static-ilp64-seq",
        feature = "mkl-dynamic-lp64-iomp",
        feature = "mkl-dynamic-lp64-seq",
        feature = "mkl-dynamic-ilp64-iomp",
        feature = "mkl-dynamic-ilp64-seq",
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

#[cfg(all(
    feature = "cblas",
    feature = "mkl-static-lp64-iomp",
    target_os = "windows"
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
        "libiompstubs5md",
    ];
}

#[cfg(all(
    feature = "cblas",
    feature = "mkl-static-lp64-seq",
    target_os = "windows"
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
    feature = "mkl-static-ilp64-iomp",
    target_os = "windows"
))]
mod link_info {
    use super::*;

    pub const LINK_TYPE: &str = STATIC_LINK;
    pub const REDIST_DIRS: &[&str] = &[
        "compiler/latest/windows/redist/ia32_win/compiler",
        "mkl/latest/redist/intel64",
    ];
    pub const LINK_DIRS: &[&str] = &[
        "compiler/latest/windows/compiler/lib/ia32_win",
        "mkl/latest/lib/intel64",
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
    feature = "mkl-static-ilp64-seq",
    target_os = "windows"
))]
mod link_info {
    use super::*;
    pub const LINK_TYPE: &str = STATIC_LINK;
    pub const REDIST_DIRS: &[&str] = &[
        "compiler/latest/windows/redist/ia32_win/compiler",
        "mkl/latest/redist/intel64",
    ];
    pub const LINK_DIRS: &[&str] = &[
        "compiler/latest/windows/compiler/lib/ia32_win",
        "mkl/latest/lib/intel64",
    ];
    pub const LINK_LIBS: &[&str] = &["mkl_core", "mkl_intel_ilp64", "mkl_sequential"];
}

#[cfg(all(
    feature = "cblas",
    feature = "mkl-dynamic-lp64-iomp",
    target_os = "windows"
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
    feature = "mkl-dynamic-lp64-seq",
    target_os = "windows"
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
    feature = "mkl-dynamic-ilp64-iomp",
    target_os = "windows"
))]
mod link_info {
    use super::*;
    pub const LINK_TYPE: &str = DYNAMIC_LINK;
    pub const REDIST_DIRS: &[&str] = &[
        "compiler/latest/windows/redist/ia32_win/compiler",
        "mkl/latest/redist/intel64",
    ];
    pub const LINK_DIRS: &[&str] = &[
        "compiler/latest/windows/compiler/lib/ia32_win",
        "mkl/latest/lib/intel64",
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
    feature = "mkl-dynamic-ilp64-seq",
    target_os = "windows"
))]
mod link_info {
    use super::*;
    pub const LINK_TYPE: &str = DYNAMIC_LINK;
    pub const REDIST_DIRS: &[&str] = &[
        "compiler/latest/windows/redist/ia32_win/compiler",
        "mkl/latest/redist/intel64",
    ];
    pub const LINK_DIRS: &[&str] = &[
        "compiler/latest/windows/compiler/lib/ia32_win",
        "mkl/latest/lib/intel64",
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
