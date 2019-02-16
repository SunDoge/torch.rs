use std::env;

fn main() {
    let torch_path = env::var("TORCH_PATH").expect("TORCH_PATH not defined");
    println!(
        "cargo:rustc-link-search=native={}/lib/../../../../",
        torch_path
    );
    println!("cargo:rustc-link-search=native={}/lib", torch_path);
    println!(
        "cargo:rustc-env=LD_LIBRARY_PATH={}/lib:{}/lib/../../../../",
        torch_path, torch_path
    );
}
