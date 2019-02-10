use torch_sys::root::*;

pub struct Tensor {
    base: at::Tensor
}

impl Tensor {
    pub fn new() -> Tensor {
        Tensor {
            base: unsafe {at::Tensor::new()}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use torch_sys::root::*;

    #[test]
    fn new_empty_tensor() {
        // let _t = Tensor::new();
        let _t = unsafe {at::ones()}
    }
}