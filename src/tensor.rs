pub mod math;
pub mod op;
pub mod type_id;
pub mod vector;

use crate::intrusive_ptr::{IntrusivePtr, WrappedPtr};
use crate::storage::*;
use std::marker::PhantomData;
use std::ptr;
use torch_sys::*;

/// FloatTesnorImpl: TensorImpl
/// DoubleTensorImpl: TensorImpl
/// Tensor(impl TensorImpl)
// pub trait TensorImpl {
//     /// This fn should return *const at_TensorImpl,
//     /// but c functions accept *mut as *const,
//     /// so for convinent, just return *mut at_Tensor, but using &self
//     fn as_ptr(&self) -> *mut at_TensorImpl;
//     fn as_mut_ptr(&mut self) -> *mut at_TensorImpl;
// }

pub struct Tensor<T> {
    // tensor_impl: Rc<RefCell<TensorImpl>>,
    tensor_impl: IntrusivePtr<at_TensorImpl>,
    phantom: PhantomData<T>,
}

impl<T> Tensor<T> {
    fn as_ptr(&self) -> *mut at_TensorImpl {
        // self.tensor_impl.borrow().as_ptr()
        self.tensor_impl.as_ptr()
    }

    fn as_mut_ptr(&mut self) -> *mut at_TensorImpl {
        // self.tensor_impl.borrow_mut().as_mut_ptr()
        self.tensor_impl.as_mut_ptr()
    }
}

pub trait TensorGeneric<T> {
    fn new() -> Self;
    // fn storage() ->
    fn is_contiguous(&self) -> bool;
    fn is_transposed(&self) -> bool;

    // access methods
    fn storage(&self) -> Storage<T>;
    fn storage_offset(&self) -> isize;

    // Props
    fn n_dimension(&self) -> i32;
    fn size(&self, dim: i32) -> i64;
    fn stride(&self, dim: i32) -> i64;
    fn data(&self) -> *mut T;

    /// methods with _ will mutate itself
    /// methods without _ will create new Tensor
    /// follow the same path as Pytorch
    fn transpose_(&mut self, dim1: i32, dim2: i32) -> &mut Self;
    fn transpose(&self, dim1: i32, dim2: i32) -> Self;
}

// The reason why defining different types TensorImpl is that
// we can only impl Drop for one type.
//
// Rust has no f16 and some other types,
// so we have to use different struct.
macro_rules! impl_tensor {
    ($prefix:ident, $impl_name:ident, $tensor_name:ident, $type_name:ident, $type:ident) => {
        pub type $type_name = $type;
        pub type $tensor_name = Tensor<$type_name>;

        pub struct $impl_name {
            tensor_impl: *mut at_TensorImpl,
        }

        // impl TensorImpl for $impl_name {
        //     fn as_ptr(&self) -> *mut at_TensorImpl {
        //         // self.tensor_impl as *const at_TensorImpl
        //         self.tensor_impl
        //     }

        //     fn as_mut_ptr(&mut self) -> *mut at_TensorImpl {
        //         self.tensor_impl
        //     }
        // }
        impl WrappedPtr for $impl_name {
            type Ptr = at_TensorImpl;

            fn as_ptr(&self) -> *mut at_TensorImpl {
                // self.tensor_impl as *const at_TensorImpl
                self.tensor_impl
            }

            fn as_mut_ptr(&mut self) -> *mut at_TensorImpl {
                self.tensor_impl
            }
        }

        impl Drop for $impl_name {
            fn drop(&mut self) {
                unsafe {
                    concat_idents!($prefix, free)(self.as_mut_ptr());
                }
            }
        }

        impl $impl_name {
            pub fn new() -> Self {
                $impl_name {
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

        impl TensorGeneric<$type_name> for Tensor<$type_name> {
            fn new() -> Tensor<$type_name> {
                Tensor {
                    // tensor_impl: Rc::new(RefCell::new($impl_name::new())),
                    tensor_impl: IntrusivePtr::new($impl_name::new()),
                    phantom: PhantomData,
                }
            }

            fn is_contiguous(&self) -> bool {
                let ret = unsafe { concat_idents!($prefix, isContiguous)(self.as_ptr()) };
                ret == 1
            }

            fn is_transposed(&self) -> bool {
                let ret = unsafe { concat_idents!($prefix, isTransposed)(self.as_ptr()) };
                ret == 1
            }

            // access methods
            fn storage(&self) -> Storage<$type_name> {
                let ptr = unsafe { concat_idents!($prefix, storage)(self.as_ptr()) };
                Storage::from(ptr)
            }

            fn storage_offset(&self) -> isize {
                unsafe { concat_idents!($prefix, storageOffset)(self.as_ptr()) }
            }

            // Props
            fn n_dimension(&self) -> i32 {
                unsafe { concat_idents!($prefix, nDimension)(self.as_ptr()) }
            }

            fn size(&self, dim: i32) -> i64 {
                unsafe { concat_idents!($prefix, size)(self.as_ptr(), dim) }
            }

            fn stride(&self, dim: i32) -> i64 {
                unsafe { concat_idents!($prefix, stride)(self.as_ptr(), dim) }
            }

            fn data(&self) -> *mut $type_name {
                unsafe { concat_idents!($prefix, data)(self.as_ptr()) }
            }

            fn transpose_(&mut self, dim1: i32, dim2: i32) -> &mut Self {
                unsafe {
                    concat_idents!($prefix, transpose)(
                        self.as_mut_ptr(),
                        ptr::null_mut(),
                        dim1,
                        dim2,
                    )
                };
                self
            }

            fn transpose(&self, dim1: i32, dim2: i32) -> Self {
                let mut ret = Self::new();
                unsafe {
                    concat_idents!($prefix, transpose)(self.as_ptr(), ret.as_mut_ptr(), dim1, dim2)
                };
                ret
            }
        }
    };
}

/// struct FloatTensorImpl {}
/// type Float = f32;
/// type FloatTensor = Tensor<Float>;
impl_tensor!(THHalfTensor_, HalfTensorImpl, HalfTensor, Half, c10_Half);
impl_tensor!(THFloatTensor_, FloatTensorImpl, FloatTensor, Float, f32);
impl_tensor!(THDoubleTensor_, DoubleTensorImpl, DoubleTensor, Double, f64);
impl_tensor!(THByteTensor_, ByteTensorImpl, ByteTensor, Byte, u8);
impl_tensor!(THCharTensor_, CharTensorImpl, CharTensor, Char, i8);
impl_tensor!(THIntTensor_, IntTensorImpl, IntTensor, Int, i32);
impl_tensor!(THLongTensor_, LongTensorImpl, LongTensor, Long, i64);
impl_tensor!(THShortTensor_, ShortTensorImpl, ShortTensor, Short, i16);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_float_tensor() {
        let _t = Tensor::<Float>::new();
        assert_eq!(_t.is_contiguous(), true);
        println!("{}", _t.n_dimension());
        // let _t = Tensor::<f32>::new();
        // let _t = Tensor::<f32>::new();
    }

    #[test]
    fn new_double_tensor() {
        let _t = Tensor::<Double>::new();
        assert_eq!(_t.is_contiguous(), true);
    }
}
