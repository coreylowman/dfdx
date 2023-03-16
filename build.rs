fn main() {
    println!("cargo:rerun-if-changed=build.rs");

    // If on nightly, enable "nightly" feature
    maybe_enable_nightly();

    #[cfg(feature = "cuda")]
    cuda::build_ptx();

    #[cfg(feature = "cpu-mkl-matmul")]
    intel_mkl::link().unwrap();
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

#[cfg(feature = "cuda")]
mod cuda {
    pub fn build_ptx() {
        let out_dir = std::env::var("OUT_DIR").unwrap();
        let kernel_paths: Vec<std::path::PathBuf> = glob::glob("src/**/*.cu")
            .unwrap()
            .map(|p| p.unwrap())
            .collect();
        let mut include_directories: Vec<std::path::PathBuf> = glob::glob("src/**/*.cuh")
            .unwrap()
            .map(|p| p.unwrap())
            .collect();

        for path in &mut include_directories {
            println!("cargo:rerun-if-changed={}", path.display());
            // remove the filename from the path so it's just the directory
            path.pop();
        }

        include_directories.sort();
        include_directories.dedup();

        #[allow(unused)]
        let include_options: Vec<String> = include_directories
            .into_iter()
            .map(|s| "-I".to_string() + &s.into_os_string().into_string().unwrap())
            .collect::<Vec<_>>();

        #[cfg(feature = "ci-check")]
        for mut kernel_path in kernel_paths.into_iter() {
            kernel_path.set_extension("ptx");

            let mut ptx_path: std::path::PathBuf = out_dir.clone().into();
            ptx_path.push(kernel_path.as_path().file_name().unwrap());
            std::fs::File::create(ptx_path).unwrap();
        }

        #[cfg(not(feature = "ci-check"))]
        {
            let start = std::time::Instant::now();

            let compute_cap = {
                let out = std::process::Command::new("nvidia-smi")
                    .arg("--query-gpu=compute_cap")
                    .arg("--format=csv")
                    .output()
                    .unwrap();
                let out = std::str::from_utf8(&out.stdout).unwrap();
                let mut lines = out.lines();
                assert_eq!(lines.next().unwrap(), "compute_cap");
                lines.next().unwrap().replace('.', "")
            };

            kernel_paths
                .iter()
                .for_each(|p| println!("cargo:rerun-if-changed={}", p.display()));

            let children = kernel_paths
                .iter()
                .map(|p| {
                    std::process::Command::new("nvcc")
                        .arg(format!("--gpu-architecture=sm_{compute_cap}"))
                        .arg("--ptx")
                        .args(["--default-stream", "per-thread"])
                        .args(["--output-directory", &out_dir])
                        .args(&include_options)
                        .arg(p)
                        .spawn()
                        .unwrap()
                })
                .collect::<Vec<_>>();

            for (kernel_path, child) in kernel_paths.iter().zip(children.into_iter()) {
                let output = child.wait_with_output().unwrap();
                assert!(
                    output.status.success(),
                    "nvcc error while compiling {kernel_path:?}: {output:?}",
                );
            }

            println!(
                "cargo:warning=Compiled {:?} cuda kernels in {:?}",
                kernel_paths.len(),
                start.elapsed()
            );
        }
    }
}

#[cfg(feature = "cpu-mkl-matmul")]
mod intel_mkl {
    //! This script links to Intel MKL when the `cpu-mkl-matmul` feature is enabled.
    //! The dynamically linked and threaded implementation is chosen as
    //! a good default, because:
    //! 1. Dynamic compiles faster and makes smaller binaries
    //! 2. Threaded implementation is faster for large workloads.
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
    //! - [x] MacOS 64 bit
    //!
    //! This script also creates a "nightly" feature if the crate is compiled on a nightly branch

    pub const MKL_VERSION: &str = "2022.1.0";
    pub const STATIC: bool = false;
    pub const DYNAMIC: bool = true;
    pub const SEQUENTIAL: bool = false;
    pub const THREADED: bool = true;

    pub const LINK_TYPE: &str = if STATIC { "static" } else { "dylib" };
    pub const LIB_POSTFIX: &str = if cfg!(windows) && DYNAMIC { "_dll" } else { "" };
    pub const LD_DIR: &str = if cfg!(windows) {
        "PATH"
    } else if cfg!(target_os = "linux") {
        "LD_LIBRARY_PATH"
    } else if cfg!(target_os = "macos") {
        "DYLD_LIBRARY_PATH"
    } else {
        ""
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

    #[cfg(target_os = "macos")]
    const MACOS_COMPILER_PATH: &str = "compiler/latest/mac/compiler/lib";

    #[cfg(target_os = "macos")]
    pub const LINK_DIRS: &[&str] = &[MACOS_COMPILER_PATH, "mkl/latest/lib"];

    #[cfg(target_os = "macos")]
    pub const SHARED_LIB_DIRS: &[&str] = &["mkl/latest/lib"];

    #[derive(Debug)]
    pub enum BuildError {
        OneAPINotFound(std::path::PathBuf),
        OneAPINotADir(std::path::PathBuf),
        PathNotFound(std::env::VarError),
        AddSharedLibDirToPath(String),
    }

    pub fn link() -> Result<(), BuildError> {
        let root = std::env::var("ONEAPI_ROOT").unwrap_or_else(|_| DEFAULT_ONEAPI_ROOT.to_string());
        println!("Using '{root}' as ONEAPI_ROOT");

        let path = match std::env::var(LD_DIR) {
            Ok(path) => path,
            Err(e) => {
                // On macOS it's unusual to set $DYLD_LIBRARY_PATH, so we want to provide a helpful message
                println!(
                    "Library path env variable '{LD_DIR}' was not found. Run `{}`",
                    suggest_setvars_cmd(&root)
                );
                return Err(BuildError::PathNotFound(e));
            }
        };

        if DYNAMIC {
            // check to make sure that things in `SHARED_LIB_DIRS` are in `$LD_DIR`.
            let path = path.replace('\\', "/");
            for shared_lib_dir in SHARED_LIB_DIRS {
                let versioned_dir = shared_lib_dir.replace("latest", MKL_VERSION);

                println!("Checking that '{shared_lib_dir}' or '{versioned_dir}' is in {LD_DIR}");
                if !path.contains(shared_lib_dir) && !path.contains(&versioned_dir) {
                    println!(
                        "'{shared_lib_dir}' not found in library path. Run `{}`",
                        suggest_setvars_cmd(&root)
                    );
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

        #[cfg(target_os = "macos")]
        {
            println!(
                "cargo:rustc-link-arg=-Wl,-rpath,{}/{MACOS_COMPILER_PATH}",
                root.display(),
            );
        }

        Ok(())
    }

    // This section creates a feature "nightly" enabled if built on a nightly branch
    fn suggest_setvars_cmd(root: &str) -> String {
        if cfg!(windows) {
            format!("{root}/setvars.bat")
        } else {
            format!("source {root}/setvars.sh")
        }
    }
}
