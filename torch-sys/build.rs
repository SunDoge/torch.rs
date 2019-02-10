use std::env;
use std::path::PathBuf;

fn main() {
    let torch_path = env::var("TORCH_PATH").expect("TORCH_PATH not defined");
    println!("cargo:rustc-link-search=native={}/lib", torch_path);
    println!("cargo:rustc-link-lib=torch");
    println!("cargo:rustc-link-lib=c10");
    println!("cargo:rustc-link-lib=c10_cuda");
    println!("cargo:rustc-link-lib=caffe2");
    println!("cargo:rustc-link-lib=caffe2_gpu");
    println!("cargo:rustc-link-lib=nvrtc");
    println!("cargo:rustc-link-lib=cudart");
    println!("cargo:rustc-link-lib=nvToolsExt");
    println!("cargo:rustc-link-lib=nvrtc-builtins");
    println!("cargo:rustc-link-lib=cuda");
    println!("cargo:rustc-link-lib=shm");

    // let torch_path = PathBuf::from(torch_path);

    let bindings = bindgen::Builder::default()
        .clang_arg(format!(
            "-I{}/lib/include/torch/csrc/api/include",
            torch_path
        ))
        .clang_arg(format!("-I{}/lib/include/", torch_path))
        // .clang_arg(format!("-L{}/lib/", torch_path))
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
        .opaque_type("std::.*")
        // .whitelist_type("std::shared_ptr")
        .whitelist_type("at::.*")
        .whitelist_function("at::.*")
        .whitelist_type("torch::.*")
        .whitelist_function("torch::.*")
        .layout_tests(false)
        // .whitelist_function("TORCH_API .*")
        // The input header we would like to generate
        // bindings for.
        .header("wrapper.hpp")
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
