pub mod op;
pub mod type_id;

use std::marker::PhantomData;
use std::rc::Rc;
use std::cell::RefCell;
use torch_sys::*;


/// FloatTesnorImpl: TensorImpl
/// DoubleTensorImpl: TensorImpl
/// Tensor(impl TensorImpl)
pub trait TensorImpl {
    fn as_ptr(&self) -> *const at_TensorImpl;
    fn as_mut_ptr(&mut self) -> *mut at_TensorImpl;
}

pub struct Tensor<T> {
    tensor_impl: Rc<RefCell<TensorImpl>>,
    phantom: PhantomData<T>,
}

impl<T> Tensor<T> {
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
}


// The reason why defining different types TensorImpl is that
// we can only impl Drop for one type.
macro_rules! impl_tensor_impl {
    ($name:ident, $prefix:ident, $type:ident) => {
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
                    concat_idents!($prefix, _free)(self.as_mut_ptr());
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

        impl TensorGeneric<$type> for Tensor<$type> {
            fn new() -> Tensor<$type> {
                Tensor {
                    tensor_impl: Rc::new(RefCell::new($name::new())),
                    phantom: PhantomData,
                }
            }

            fn is_contiguous(&self) -> bool {
                let ret = unsafe { concat_idents!($prefix, _isContiguous)(self.as_ptr()) };
                ret == 1
            }

            fn dim(&self) -> i32 {
                unsafe {
                    concat_idents!($prefix, _nDimension)(self.as_ptr())
                }
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
