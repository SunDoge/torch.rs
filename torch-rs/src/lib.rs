#![feature(concat_idents)]

pub mod storage;
pub mod tensor;
pub mod intrusive_ptr;

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
