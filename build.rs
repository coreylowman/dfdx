fn main() {
    println!("cargo:rerun-if-changed=build.rs");

    // If on nightly, enable "nightly" feature
    maybe_enable_nightly();

    #[cfg(feature = "cuda")]
    cuda::set_include_dir();

    #[cfg(feature = "cuda")]
    cuda::build_ptx();
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
    pub fn set_include_dir() {
        // NOTE: copied from cudarc build.rs.
        // We can't actually set a env!() value from another crate,
        // so we have to do that here.

        use std::path::PathBuf;

        let env_vars = [
            "CUDA_PATH",
            "CUDA_ROOT",
            "CUDA_TOOLKIT_ROOT_DIR",
            "CUDNN_LIB",
        ];
        #[allow(unused)]
        let env_vars = env_vars
            .into_iter()
            .map(std::env::var)
            .filter_map(Result::ok)
            .map(Into::<PathBuf>::into);

        let roots = [
            "/usr",
            "/usr/local/cuda",
            "/opt/cuda",
            "/usr/lib/cuda",
            "C:/Program Files/NVIDIA GPU Computing Toolkit",
            "C:/CUDA",
        ];
        #[allow(unused)]
        let roots = roots.into_iter().map(Into::<PathBuf>::into);

        #[cfg(feature = "ci-check")]
        let root: PathBuf = "ci".into();

        #[cfg(not(feature = "ci-check"))]
        let root = env_vars
            .chain(roots)
            .find(|path| path.join("include").join("cuda.h").is_file())
            .unwrap();

        println!(
            "cargo:rustc-env=CUDA_INCLUDE_DIR={}",
            root.join("include").display()
        );
    }

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
            let destination =
                std::format!("{out_dir}/{}", path.file_name().unwrap().to_str().unwrap());
            std::fs::copy(path.clone(), destination).unwrap();
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
        {
            println!("cargo:rustc-env=CUDA_COMPUTE_CAP=ci");

            for mut kernel_path in kernel_paths.into_iter() {
                kernel_path.set_extension("ptx");

                let mut ptx_path: std::path::PathBuf = out_dir.clone().into();
                ptx_path.push(kernel_path.as_path().file_name().unwrap());
                std::fs::File::create(ptx_path).unwrap();
            }
        }

        #[cfg(not(feature = "ci-check"))]
        {
            let start = std::time::Instant::now();

            // Grab compute code from nvidia-smi
            let mut compute_cap = {
                let out = std::process::Command::new("nvidia-smi")
                    .arg("--query-gpu=compute_cap")
                    .arg("--format=csv")
                    .output()
                    .unwrap();
                let out = std::str::from_utf8(&out.stdout).unwrap();
                let mut lines = out.lines();
                assert_eq!(lines.next().unwrap(), "compute_cap");
                let cap = lines.next().unwrap().replace('.', "");
                cap.parse::<usize>().unwrap()
            };

            // Grab available GPU codes from nvcc and select the highest one
            let max_nvcc_code = {
                let out = std::process::Command::new("nvcc")
                    .arg("--list-gpu-code")
                    .output()
                    .unwrap();
                let out = std::str::from_utf8(&out.stdout).unwrap();

                let out = out.lines().collect::<Vec<&str>>();
                let mut codes = Vec::with_capacity(out.len());
                for code in out {
                    let code = code.split("_").collect::<Vec<&str>>();
                    if code.len() != 0 && code.contains(&"sm") {
                        if let Ok(num) = code[1].parse::<usize>() {
                            codes.push(num);
                        }
                    }
                }
                codes.sort();
                if !codes.contains(&compute_cap) {
                    panic!("nvcc cannot target gpu arch {compute_cap}. Available nvcc targets are {codes:?}.");
                }
                *codes.last().unwrap()
            };

            // If nvidia-smi compute_cap is higher than the highest gpu code from nvcc,
            // then choose the highest gpu code in nvcc
            if compute_cap > max_nvcc_code {
                println!("cargo:warning=Lowering gpu arch {compute_cap} to max nvcc target {max_nvcc_code}.");
                compute_cap = max_nvcc_code;
            }

            println!("cargo:rerun-if-env-changed=CUDA_COMPUTE_CAP");
            if let Ok(compute_cap_str) = std::env::var("CUDA_COMPUTE_CAP") {
                compute_cap = compute_cap_str.parse::<usize>().unwrap();
                println!("cargo:warning=Using gpu arch {compute_cap} from $CUDA_COMPUTE_CAP");
            }

            println!("cargo:rustc-env=CUDA_COMPUTE_CAP=sm_{compute_cap}");

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
                        .stdout(std::process::Stdio::piped())
                        .stderr(std::process::Stdio::piped())
                        .spawn()
                        .unwrap()
                })
                .collect::<Vec<_>>();

            for (kernel_path, child) in kernel_paths.iter().zip(children.into_iter()) {
                let output = child.wait_with_output().unwrap();
                assert!(
                    output.status.success(),
                    "nvcc error while compiling {kernel_path:?}:\n\n# stdout\n{:#}\n\n# stderr\n{:#}",
                    String::from_utf8_lossy(&output.stdout),
                    String::from_utf8_lossy(&output.stderr)
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
