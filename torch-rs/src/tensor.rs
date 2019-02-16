pub mod op;
pub mod type_id;

use std::cell::RefCell;
use std::rc::Rc;
use torch_sys::*;

/// FloatTesnorImpl: TensorImpl
/// DoubleTensorImpl: TensorImpl
/// Tensor(impl TensorImpl)
pub trait TensorImpl {
    fn as_ptr(&self) -> *const at_TensorImpl;
    fn as_mut_ptr(&mut self) -> *mut at_TensorImpl;
}

pub struct Tensor<T> {
    tensor_impl: Rc<RefCell<T>>,
}

impl<T: TensorImpl> Tensor<T> {
    fn as_ptr(&self) -> *const at_TensorImpl {
        self.tensor_impl.borrow().as_ptr()
    }

    fn as_mut_ptr(&mut self) -> *mut at_TensorImpl {
        self.tensor_impl.borrow_mut().as_mut_ptr()
    }
}

pub trait TensorGeneric<T> {
    fn new() -> Tensor<T>;
    // fn storage() ->
    fn is_contiguous(&self) -> bool;

    // Props
    fn dim(&self) -> i32;
    fn size(&self, dim: i32) -> i64;
}

// The reason why defining different types TensorImpl is that
// we can only impl Drop for one type.
//
// Rust has no f16 and some other types,
// so we have to use different struct.
macro_rules! impl_tensor_impl {
    ($name:ident, $prefix:ident) => {
        pub struct $name {
            tensor_impl: *mut at_TensorImpl,
        }

        impl TensorImpl for $name {
            fn as_ptr(&self) -> *const at_TensorImpl {
                self.tensor_impl as *const at_TensorImpl
            }

            fn as_mut_ptr(&mut self) -> *mut at_TensorImpl {
                self.tensor_impl
            }
        }

        impl Drop for $name {
            fn drop(&mut self) {
                unsafe {
                    concat_idents!($prefix, free)(self.as_mut_ptr());
                }
            }
        }

        impl $name {
            pub fn new() -> Self {
                $name {
                    tensor_impl: unsafe { concat_idents!($prefix, new)() },
                }
            }
        }

        // impl std::ops::Deref for $name {
        //     type Target = at_TensorImpl;

        //     fn deref(&self) -> &at_TensorImpl {
        //         unsafe {
        //             &*self.tensor_impl
        //         }
        //     }
        // }

        // impl std::ops::DerefMut for $name {

        //     fn deref_mut(&mut self) -> &mut at_TensorImpl {
        //         unsafe {
        //             &mut *self.tensor_impl
        //         }
        //     }
        // }

        impl TensorGeneric<$name> for Tensor<$name> {
            fn new() -> Tensor<$name> {
                Tensor {
                    tensor_impl: Rc::new(RefCell::new($name::new())),
                }
            }

            fn is_contiguous(&self) -> bool {
                let ret = unsafe { concat_idents!($prefix, isContiguous)(self.as_ptr()) };
                ret == 1
            }

            fn dim(&self) -> i32 {
                unsafe { concat_idents!($prefix, nDimension)(self.as_ptr()) }
            }

            fn size(&self, dim: i32) -> i64 {
                unsafe { concat_idents!($prefix, size)(self.as_ptr(), dim) }
            }
        }
    };
}

impl_tensor_impl!(Half, THHalfTensor_);
impl_tensor_impl!(Float, THFloatTensor_);
impl_tensor_impl!(Double, THDoubleTensor_);
impl_tensor_impl!(Byte, THByteTensor_);
impl_tensor_impl!(Char, THCharTensor_);
impl_tensor_impl!(Int, THIntTensor_);
impl_tensor_impl!(Long, THLongTensor_);
impl_tensor_impl!(Short, THShortTensor_);

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn new_float_tensor() {
        let _t = Tensor::<Float>::new();
        assert_eq!(_t.is_contiguous(), true);
        println!("{}", _t.dim());
        // let _t = Tensor::<f32>::new();
        // let _t = Tensor::<f32>::new();
    }

    #[test]
    fn new_double_tensor() {
        let _t = Tensor::<Double>::new();
        assert_eq!(_t.is_contiguous(), true);
    }
}
