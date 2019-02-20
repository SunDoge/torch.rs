use crate::intrusive_ptr::{IntrusivePtr, WrappedPtr};
use std::cell::RefCell;
use std::marker::PhantomData;
use std::ptr;
use std::rc::Rc;
use torch_sys::*;

/// FloatTesnorImpl: TensorImpl
/// DoubleTensorImpl: TensorImpl
/// Tensor(impl TensorImpl)
// pub trait StorageImpl {
//     /// This fn should return *const at_TensorImpl,
//     /// but c functions accept *mut as *const,
//     /// so for convinent, just return *mut at_Tensor, but using &self
//     fn as_ptr(&self) -> *mut c10_StorageImpl;
//     fn as_mut_ptr(&mut self) -> *mut c10_StorageImpl;
// }

pub struct Storage<T> {
    storage_impl: IntrusivePtr<c10_StorageImpl>,
    phantom: PhantomData<T>,
}

impl<T> From<IntrusivePtr<c10_StorageImpl>> for Storage<T> {
    fn from(ptr: IntrusivePtr<c10_StorageImpl>) -> Storage<T> {
        Storage {
            storage_impl: ptr,
            phantom: PhantomData
        }
    }
}


impl<T> Storage<T> {
    fn as_ptr(&self) -> *mut c10_StorageImpl {
        // self.storage_impl.borrow().as_ptr()
        self.storage_impl.as_ptr()
    }

    fn as_mut_ptr(&mut self) -> *mut c10_StorageImpl {
        // self.storage_impl.borrow_mut().as_mut_ptr()
        self.storage_impl.as_mut_ptr()
    }
}


pub trait StorageGeneric<T> {
    fn new() -> Storage<T>;
    fn new_with_size(size: usize) -> Storage<T>;
    fn data(&self) -> *mut T;
    fn size(&self) -> usize;
}

// Maybe this should be merge to StorageGeneric
pub trait StorageCopy<T> {
    fn raw_copy(&mut self, src: *mut T);
    fn copy(&mut self, src: *mut c10_StorageImpl);
    fn copy_float(&mut self, src: &mut FloatStorage);
}

macro_rules! impl_storage {
    ($prefix:ident, $impl_name:ident, $storage_name:ident, $type_name:ident, $type:ident) => {
        pub type $type_name = $type;
        pub type $storage_name = Storage<$type_name>;

        pub struct $impl_name {
            storage_impl: *mut c10_StorageImpl,
        }

        // impl StorageImpl for $impl_name {
        //     fn as_ptr(&self) -> *mut c10_StorageImpl {
        //         // self.tensor_impl as *const c10_StorageImpl
        //         self.storage_impl
        //     }

        //     fn as_mut_ptr(&mut self) -> *mut c10_StorageImpl {
        //         self.storage_impl
        //     }
        // }

        impl WrappedPtr for $impl_name {
            type Ptr = c10_StorageImpl;
            fn as_ptr(&self) -> *mut c10_StorageImpl {
                // self.tensor_impl as *const c10_StorageImpl
                self.storage_impl
            }

            fn as_mut_ptr(&mut self) -> *mut c10_StorageImpl {
                self.storage_impl
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
                    storage_impl: unsafe { concat_idents!($prefix, new)() },
                }
            }
        }

        impl From<*mut c10_StorageImpl> for $impl_name {
            fn from(ptr: *mut c10_StorageImpl) -> Self {
                $impl_name { storage_impl: ptr }
            }
        }

        impl From<*mut c10_StorageImpl> for Storage<$type_name> {
            fn from(ptr: *mut c10_StorageImpl) -> Self {
                Self::from(IntrusivePtr::new($impl_name::from(ptr)))
            }
        }

        impl StorageGeneric<$type_name> for Storage<$type_name> {
            fn new() -> Self {
                Storage {
                    // storage_impl: Rc::new(RefCell::new($impl_name::new())),
                    storage_impl: IntrusivePtr::new($impl_name::new()),
                    phantom: PhantomData,
                }
            }

            fn new_with_size(size: usize) -> Self {
                let ptr = unsafe { concat_idents!($prefix, newWithSize)(size as isize) };
                Storage {
                    storage_impl: IntrusivePtr::new($impl_name::from(ptr)),
                    phantom: PhantomData,
                }
            }

            fn data(&self) -> *mut $type_name {
                unsafe { concat_idents!($prefix, data)(self.as_ptr()) }
            }

            fn size(&self) -> usize {
                let size = unsafe { concat_idents!($prefix, size)(self.as_ptr()) };
                size as usize
            }
        }

        impl StorageCopy<$type_name> for Storage<$type_name> {
            fn raw_copy(&mut self, src: *mut $type_name) {
                unsafe {
                    concat_idents!($prefix, rawCopy)(self.as_mut_ptr(), src);
                }
            }

            fn copy(&mut self, src: *mut c10_StorageImpl) {
                unsafe {
                    concat_idents!($prefix, copy)(self.as_mut_ptr(), src);
                }
            }

            fn copy_float(&mut self, src: &mut FloatStorage) {
                unsafe {
                    concat_idents!($prefix, copy)(self.as_mut_ptr(), src.as_mut_ptr());
                }
            }
        }
    };
}

impl_storage!(THFloatStorage_, FloatStorageImpl, FloatStorage, Float, f32);
impl_storage!(
    THDoubleStorage_,
    DoubleStorageImpl,
    DoubleStorage,
    Double,
    f64
);
impl_storage!(THHalfStorage_, HalfStorageImpl, HalfStorage, Half, c10_Half);
impl_storage!(THByteStorage_, ByteStorageImpl, ByteStorage, Byte, u8);
impl_storage!(THCharStorage_, CharStorageImpl, CharStorage, Char, i8);
impl_storage!(THShortStorage_, ShortStorageImpl, ShortStorage, Short, i16);
impl_storage!(THIntStorage_, IntStorageImpl, IntStorage, Int, i32);
impl_storage!(THLongStorage_, LongStorageImpl, LongStorage, Long, i64);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn copy_vec_to_storage() {
        let mut data = vec![1.0, 2.0, 3.0];
        let mut storage = FloatStorage::new_with_size(3);
        println!("size: {}", storage.size());
        storage.raw_copy(data.as_mut_ptr());
        println!("size: {}", storage.size());
        println!("{:?}", unsafe {
            std::slice::from_raw_parts(storage.data(), storage.size())
        });
    }
}
