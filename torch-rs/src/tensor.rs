pub mod op;

use std::marker::PhantomData;
use std::rc::Rc;
use torch_sys::*;


/// FloatTesnorImpl: TensorImpl
/// DoubleTensorImpl: TensorImpl
/// Tensor(impl TensorImpl)
pub trait TensorImpl {
    fn tensor_impl(&self) -> *mut at_TensorImpl;
}

pub struct Tensor<T> {
    tensor_impl: Rc<TensorImpl>,
    phantom: PhantomData<T>,
}

impl<T> TensorImpl for Tensor<T> {
    fn tensor_impl(&self) -> *mut at_TensorImpl {
        self.tensor_impl.tensor_impl()
    }
}

pub trait TensorGeneric<T> {
    fn new() -> Tensor<T>;
    fn is_contiguous(&self) -> bool;
}

macro_rules! impl_tensor_impl {
    ($name:ident, $prefix:ident, $type:ident) => {
        pub struct $name {
            tensor_impl: *mut at_TensorImpl,
        }

        impl TensorImpl for $name {
            fn tensor_impl(&self) -> *mut at_TensorImpl {
                self.tensor_impl
            }
        }

        impl Drop for $name {
            fn drop(&mut self) {
                unsafe {
                    concat_idents!($prefix, _free)(self.tensor_impl());
                }
            }
        }

        impl $name {
            pub fn new() -> Self {
                $name {
                    tensor_impl: unsafe { concat_idents!($prefix, _new)() },
                }
            }
        }

        impl TensorGeneric<$type> for Tensor<$type> {
            fn new() -> Tensor<$type> {
                Tensor {
                    tensor_impl: Rc::new($name::new()),
                    phantom: PhantomData,
                }
            }

            fn is_contiguous(&self) -> bool {
                let ret = unsafe { concat_idents!($prefix, _isContiguous)(self.tensor_impl()) };
                ret == 1
            }
        }
    };
}

impl_tensor_impl!(FloatTensorImpl, THFloatTensor, f32);
impl_tensor_impl!(DoubleTensorImpl, THDoubleTensor, f64);

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn new_float_tensor() {
        let _t = Tensor::<f32>::new();
        assert_eq!(_t.is_contiguous(), true);
        // let _t = Tensor::<f32>::new();
        // let _t = Tensor::<f32>::new();
    }

    #[test]
    fn new_double_tensor() {
        let _t = Tensor::<f64>::new();
        assert_eq!(_t.is_contiguous(), true);
    }
}
