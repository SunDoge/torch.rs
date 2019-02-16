use std::cell::RefCell;
use std::marker::PhantomData;
use std::rc::Rc;
use torch_sys::*;

/// FloatStorageImpl: StorageImpl
/// DoubleStorageImpl: TensorImpl
/// Tensor(impl TensorImpl)
pub trait StorageImpl {
    fn as_ptr(&self) -> *const c10_StorageImpl;
    fn as_mut_ptr(&mut self) -> *mut c10_StorageImpl;
}

pub struct Storage<T> {
    Storage_impl: Rc<RefCell<StorageImpl>>,
    phantom: PhantomData<T>,
}

impl<T> StorageImpl for Storage<T> {
    fn as_ptr(&self) -> *const c10_StorageImpl {
        self.Storage_impl.borrow().as_ptr()
    }

    fn as_mut_ptr(&mut self) -> *mut c10_StorageImpl {
        self.Storage_impl.borrow_mut().as_mut_ptr()
    }
}

pub trait StorageGeneric<T> {
    fn data(&self) -> T;
}

macro_rules! impl_Storage_impl {
    ($name:ident, $prefix:ident, $type:ident) => {
        pub struct $name {
            Storage_impl: *mut c10_StorageImpl,
        }

        impl StorageImpl for $name {
            fn as_ptr(&self) -> *const c10_StorageImpl {
                self.Storage_impl as *const c10_StorageImpl
            }

            fn as_mut_ptr(&mut self) -> *mut c10_StorageImpl {
                self.Storage_impl
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
                    Storage_impl: unsafe { concat_idents!($prefix, _new)() },
                }
            }
        }

        // impl StorageGeneric<$type> for Storage<$type> {

        // }
    };
}

impl_Storage_impl!(FloatStorageImpl, THFloatStorage, f32);
impl_Storage_impl!(DoubleStorageImpl, THDoubleStorage, f64);

#[cfg(test)]
mod tests {
    use super::*;

}
