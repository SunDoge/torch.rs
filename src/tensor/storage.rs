use super::device::CPU;
// use crate::intrusive_ptr::{IntrusivePtr};
use std::cell::RefCell;
use std::marker::PhantomData;
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

/// A torch.Storage is a contiguous, one-dimensional array of a single data type.
/// Every torch.Tensor has a corresponding storage of the same data type.
pub struct StorageBase<T, C> {
    storage_impl: Rc<RefCell<StorageImpl>>,
    _type: PhantomData<T>,
    _context: PhantomData<C>,
}

pub struct StorageImpl {
    ptr: *mut c10_StorageImpl,
}

impl StorageImpl {
    pub fn new(ptr: *mut c10_StorageImpl) -> StorageImpl {
        StorageImpl { ptr }
    }

    pub fn as_ptr(&self) -> *mut c10_StorageImpl {
        self.ptr
    }

    pub fn as_mut_ptr(&mut self) -> *mut c10_StorageImpl {
        self.ptr
    }
}

impl Drop for StorageImpl {
    fn drop(&mut self) {
        unsafe {
            THStorage_free(self.ptr);
        }
    }
}

pub type Storage<T> = StorageBase<T, CPU>;

// impl<T, C> From<IntrusivePtr<c10_StorageImpl>> for StorageBase<T, C> {
//     fn from(ptr: IntrusivePtr<c10_StorageImpl>) -> StorageBase<T, C> {
//         Storage {
//             storage_impl: ptr,
//             storage_type: PhantomData,
//             storage_context: PhantomData,
//         }
//     }
// }

// impl<T, C> From<*mut c10_StorageImpl> for StorageBase<T, C> {
//     fn from(ptr: *mut c10_StorageImpl) -> Self {
//         Storage {
//             storage_impl: Rc::new(RefCell::new(StorageImpl::new(ptr))),
//             _type: PhantomData,
//             _context: PhantomData,
//         }
//     }
// }

impl<T, C> StorageBase<T, C> {
    pub fn as_ptr(&self) -> *mut c10_StorageImpl {
        self.storage_impl.as_ref().borrow().as_ptr()
    }

    pub fn as_mut_ptr(&mut self) -> *mut c10_StorageImpl {
        self.storage_impl.as_ref().borrow_mut().as_mut_ptr()
    }
}

impl<T, C> From<*mut c10_StorageImpl> for StorageBase<T, C> {
    fn from(ptr: *mut c10_StorageImpl) -> Self {
        StorageBase {
            storage_impl: Rc::new(RefCell::new(StorageImpl::new(ptr))),
            _type: PhantomData,
            _context: PhantomData,
        }
    }
}

pub trait StorageGeneric<T, C> {
    fn new() -> StorageBase<T, C>;
    fn new_with_size(size: usize) -> StorageBase<T, C>;
    fn data_ptr(&self) -> *mut T;
    fn size(&self) -> usize;
    fn tolist(&self) -> &[T];
}

// Maybe this should be merge to StorageGeneric
pub trait StorageCopy<T, C> {
    fn raw_copy(&mut self, src: *mut T);
    fn copy(&mut self, src: *mut c10_StorageImpl);
    fn copy_float(&mut self, src: &mut StorageBase<Float, CPU>);
}

macro_rules! impl_storage {
    ($prefix:ident, $storage_name:ident, $type_name:ident, $type:ident) => {
        pub type $type_name = $type;
        pub type $storage_name = Storage<$type_name>;

        impl StorageGeneric<$type_name, CPU> for Storage<$type_name> {
            fn new() -> Self {
                let ptr = unsafe { concat_idents!($prefix, new)() };
                Storage::from(ptr)
            }

            fn new_with_size(size: usize) -> Self {
                let ptr = unsafe { concat_idents!($prefix, newWithSize)(size as isize) };
                Storage::from(ptr)
            }

            fn data_ptr(&self) -> *mut $type_name {
                unsafe { concat_idents!($prefix, data)(self.as_ptr()) }
            }

            fn size(&self) -> usize {
                let size = unsafe { concat_idents!($prefix, size)(self.as_ptr()) };
                size as usize
            }

            fn tolist(&self) -> &[$type_name] {
                unsafe { std::slice::from_raw_parts(self.data_ptr(), self.size()) }
            }
        }

        impl StorageCopy<$type_name, CPU> for Storage<$type_name> {
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
                    concat_idents!($prefix, copyFloat)(self.as_mut_ptr(), src.as_mut_ptr());
                }
            }
        }
    };
}

impl_storage!(THFloatStorage_, FloatStorage, Float, f32);
impl_storage!(THDoubleStorage_, DoubleStorage, Double, f64);
impl_storage!(THHalfStorage_, HalfStorage, Half, c10_Half);
impl_storage!(THByteStorage_, ByteStorage, Byte, u8);
impl_storage!(THCharStorage_, CharStorage, Char, i8);
impl_storage!(THShortStorage_, ShortStorage, Short, i16);
impl_storage!(THIntStorage_, IntStorage, Int, i32);
impl_storage!(THLongStorage_, LongStorage, Long, i64);

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
        println!("{:?}", storage.tolist());
    }
}
