#![feature(concat_idents)]

#[macro_use]
extern crate lazy_static;

pub mod intrusive_ptr;
pub mod tensor;

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
