use std::env;
use std::path::PathBuf;

fn main() {
    let torch_path = env::var("TORCH_PATH").expect("TORCH_PATH not defined");
    let cuda_path = env::var("CUDA_PATH").expect("CUDA_PATH not defined");
    // println!("cargo:rustc-link-search=native={}", torch_path);
    println!(
        "cargo:rustc-link-search=native={}/lib/../../../../",
        torch_path
    );
    println!("cargo:rustc-link-search=native={}/lib", torch_path);

    println!("cargo:rustc-link-lib=torch");
    // println!("cargo:rustc-link-lib=c10");
    // println!("cargo:rustc-link-lib=c10_cuda");
    println!("cargo:rustc-link-lib=caffe2");
    println!("cargo:rustc-link-lib=caffe2_gpu");

    // println!("cargo:rustc-link-lib=mkl_intel_lp64");
    // println!("cargo:rustc-link-lib=mkl_gnu_thread");
    // println!("cargo:rustc-link-lib=mkl_core");

    // println!("cargo:rustc-link-lib=nvToolsExt");
    // println!("cargo:rustc-link-lib=cudart");
    // println!("cargo:rustc-link-lib=cusparse");
    // println!("cargo:rustc-link-lib=curand");
    // println!("cargo:rustc-link-lib=cufft");
    // println!("cargo:rustc-link-lib=cublas");

    println!("cargo:rustc-link-lib=nvrtc");
    println!("cargo:rustc-link-lib=cuda");

    // println!("cargo:rustc-link-lib=mkldnn");
    // println!("cargo:rustc-link-lib=shm");

    // println!("cargo:rustc-link-lib=_C");

    // let torch_path = PathBuf::from(torch_path);

    let bindings = bindgen::Builder::default()
        .clang_arg(format!(
            "-I{}/lib/include/torch/csrc/api/include",
            torch_path
        ))
        .clang_arg(format!("-I{}/lib/include/", torch_path))
        .clang_arg(format!("-I{}/lib/include/TH", torch_path))
        .clang_arg(format!("-I{}/lib", torch_path))
        .clang_arg(format!("-I{}/include", cuda_path))
        // .clang_arg(format!("-L{}/lib/", torch_path))
        // .clang_arg(format!("-L{}/lib/../../../../", torch_path))
        // .clang_arg(format!("-L{}", torch_path))
        .clang_arg("-x")
        .clang_arg("c++")
        .clang_arg("-std=c++11")
        // .enable_cxx_namespaces()
        // .blacklist_type("max_align_t")
        // .opaque_type("intrusive_ptr.*")
        // .trust_clang_mangling(false)
        // .derive_copy(false)
        // .clang_arg("-stdlib=libc++")
        // .blacklist_type("std::.*")
        // .blacklist_type(".*counted.*")
        // .opaque_type("std::.*")
        // .whitelist_type("std::shared_ptr")
        // .whitelist_type("at::.*")
        // .whitelist_function("at::.*")
        // .whitelist_type("torch::.*")
        // .whitelist_function("torch::.*")
        // .whitelist_type("caffe2::.*")
        // .whitelist_type("IntList")
        // .trust_clang_mangling(fwdfqwdfalse)
        // .whitelist_type("c10::.*")
        // .whitelist_function("c10::.*")
        .layout_tests(false)
        // .whitelist_function("TORCH_API .*")
        // The input header we would like to generate
        // bindings for.
        // .header("wrapper.hpp")
        // https://github.com/servo/rust-bindgen/issues/687
        // .blacklist_type("FP_NAN")
        // .blacklist_type("FP_INFINITE")
        // .blacklist_type("FP_ZERO")
        // .blacklist_type("FP_SUBNORMAL")
        // .blacklist_type("FP_NORMAL")
        // https://github.com/servo/rust-bindgen/issues/550
        // .blacklist_type("max_align_t")
        // .blacklist_item("FP_.*")
        .whitelist_function("TH.*")
        .header("wrapper.h")
        // Finish the builder and generate the bindings.
        .generate()
        // Unwrap the Result and panic on failure.
        .expect("Unable to generate bindings");

    // Write the bindings to the $OUT_DIR/bindings.rs file.
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");
}
