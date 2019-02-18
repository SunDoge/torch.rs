use super::*;
use std::ops::{Add, AddAssign};

pub trait TensorMath<T> {
    fn add(&mut self, t: &mut Self, value: T);
}

macro_rules! impl_tensor_math {
    ($prefix:ident, $tensor_name:ident, $type_name:ident) => {
        impl TensorMath<$type_name> for Tensor<$type_name> {
            fn add(&mut self, t: &mut Self, value: $type_name) {
                unsafe {
                    concat_idents!($prefix, add)(self.as_mut_ptr(), t.as_mut_ptr(), value);
                }
            }
        }

        impl std::ops::Add for Tensor<$type_name> {
            type Output = Self;
            // they don't need to be mut
            fn add(mut self, mut other: Self) -> Self {
                TensorMath::add(&mut self, &mut other, 1 as $type_name);
                self
            }
        }
    };
}

impl_tensor_math!(THFloatTensor_, FloatTensor, Float);
impl_tensor_math!(THDoubleTensor_, DoubleTensor, Double);

