use std::cell::RefCell;
use std::marker::PhantomData;
use std::ptr;
use std::rc::Rc;
use torch_sys::*;

/// FloatTesnorImpl: TensorImpl
/// DoubleTensorImpl: TensorImpl
/// Tensor(impl TensorImpl)
pub trait StorageImpl {
    /// This fn should return *const at_TensorImpl,
    /// but c functions accept *mut as *const,
    /// so for convinent, just return *mut at_Tensor, but using &self
    fn as_ptr(&self) -> *mut c10_StorageImpl;
    fn as_mut_ptr(&mut self) -> *mut c10_StorageImpl;
}

pub struct Storage<T> {
    storage_impl: Rc<RefCell<StorageImpl>>,
    phantom: PhantomData<T>,
}

impl<T> Storage<T> {
    fn as_ptr(&self) -> *mut c10_StorageImpl {
        self.storage_impl.borrow().as_ptr()
    }

    fn as_mut_ptr(&mut self) -> *mut c10_StorageImpl {
        self.storage_impl.borrow_mut().as_mut_ptr()
    }
}

pub trait StorageGeneric<T> {
    fn new() -> Storage<T>;
    fn data(&self) -> *mut T;
}

// Maybe this should be merge to StorageGeneric
pub trait StorageCopy<T> {
    fn raw_copy(&mut self, src: *mut T);
}

macro_rules! impl_storage {
    ($prefix:ident, $impl_name:ident, $storage_name:ident, $type_name:ident, $type:ident) => {
        pub type $type_name = $type;
        pub type $storage_name = Storage<$type_name>;

        pub struct $impl_name {
            storage_impl: *mut c10_StorageImpl,
        }

        impl StorageImpl for $impl_name {
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

        impl StorageGeneric<$type_name> for Storage<$type_name> {
            fn new() -> Self {
                Storage {
                    storage_impl: Rc::new(RefCell::new($impl_name::new())),
                    phantom: PhantomData,
                }
            }

            fn data(&self) -> *mut $type_name {
                unsafe { concat_idents!($prefix, data)(self.as_ptr()) }
            }
        }

        impl StorageCopy<$type_name> for Storage<$type_name> {
            fn raw_copy(&mut self, src: *mut $type_name) {
                unsafe {
                    concat_idents!($prefix, rawCopy)(self.as_mut_ptr(), src);
                }
            }
        }
    };
}

impl_storage!(THFloatStorage_, FloatStorageImpl, FloatStorage, Float, f32);

#[cfg(test)]
mod tests {
    use super::*;
}
