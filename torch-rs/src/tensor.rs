use torch_sys::*;
use std::marker::PhantomData;

pub struct Tensor<T> {
    ptr: *mut at_TensorImpl,
    phantom: PhantomData<T>
}

pub trait TensorNew<T> {
    fn new() -> Tensor<T>;
}

impl TensorNew<f32> for Tensor<f32> {
    fn new() -> Tensor<f32> {
        Tensor {
            ptr: unsafe {THFloatTensor_new()},
            phantom: PhantomData
        }
    }
}

impl TensorNew<f64> for Tensor<f64> {
    fn new() -> Tensor<f64> {
        Tensor {
            ptr: unsafe {THDoubleTensor_new()},
            phantom: PhantomData
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn new_float_tensor() {
        let _t = Tensor::<f32>::new();
    }

    #[test]
    fn new_double_tensor() {
        let _t = Tensor::<f64>::new();
    }
}
