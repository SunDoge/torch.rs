use super::super::context::STATE;
use super::super::device::CUDA;
use super::super::storage::{StorageBase, StorageCopy, StorageGeneric, StorageImpl};
use std::cell::RefCell;
use std::marker::PhantomData;
use std::rc::Rc;
use torch_sys::*;

pub type Storage<T> = StorageBase<T, CUDA>;

macro_rules! impl_storage {
    ($prefix:ident, $storage_name:ident, $type_name:ident, $type:ident) => {
        pub type $type_name = $type;
        pub type $storage_name = Storage<$type_name>;

        impl StorageGeneric<$type_name, CUDA> for Storage<$type_name> {
            fn new() -> Self {
                let ptr = unsafe { concat_idents!($prefix, new)(STATE.as_ptr()) };
                Storage::from(ptr)
            }

            fn new_with_size(size: usize) -> Self {
                let ptr =
                    unsafe { concat_idents!($prefix, newWithSize)(STATE.as_ptr(), size as isize) };
                Storage::from(ptr)
            }

            fn data_ptr(&self) -> *mut $type_name {
                unsafe { concat_idents!($prefix, data)(STATE.as_ptr(), self.as_ptr()) }
            }

            fn size(&self) -> usize {
                let size = unsafe { concat_idents!($prefix, size)(STATE.as_ptr(), self.as_ptr()) };
                size as usize
            }

            fn tolist(&self) -> &[$type_name] {
                unsafe { std::slice::from_raw_parts(self.data_ptr(), self.size()) }
            }
        }

        impl StorageCopy<$type_name, CUDA> for Storage<$type_name> {
            fn raw_copy(&mut self, src: *mut $type_name) {
                unsafe {
                    concat_idents!($prefix, rawCopy)(STATE.as_ptr(), self.as_mut_ptr(), src);
                }
            }

            fn copy(&mut self, src: *mut c10_StorageImpl) {
                unsafe {
                    concat_idents!($prefix, copy)(STATE.as_ptr(), self.as_mut_ptr(), src);
                }
            }

            fn copy_float(&mut self, src: &mut FloatStorage) {
                unsafe {
                    concat_idents!($prefix, copyFloat)(
                        STATE.as_ptr(),
                        self.as_mut_ptr(),
                        src.as_mut_ptr(),
                    );
                }
            }
        }
    };
}

impl_storage!(THCudaStorage_, FloatStorage, Float, f32);
impl_storage!(THCudaDoubleStorage_, DoubleStorage, Double, f64);
impl_storage!(THCudaHalfStorage_, HalfStorage, Half, c10_Half);
impl_storage!(THCudaByteStorage_, ByteStorage, Byte, u8);
impl_storage!(THCudaCharStorage_, CharStorage, Char, i8);
impl_storage!(THCudaShortStorage_, ShortStorage, Short, i16);
impl_storage!(THCudaIntStorage_, IntStorage, Int, i32);
impl_storage!(THCudaLongStorage_, LongStorage, Long, i64);

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
