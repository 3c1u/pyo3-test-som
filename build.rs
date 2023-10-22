use std::path::PathBuf;

fn main() {
    let conda_prefix = std::env::var("CONDA_PREFIX").ok();
    let conda_prefix = match conda_prefix {
        Some(prefix) => prefix,
        None => {
            println!("cargo:warning=CONDA_PREFIX is not set, using default");
            return;
        }
    };

    let mut lib_path = PathBuf::from(conda_prefix);
    lib_path.push("lib");

    println!("cargo:rustc-link-search=native={}", lib_path.display());
    println!("cargo:rustc-link-lib=dylib=openblas.0");
}
