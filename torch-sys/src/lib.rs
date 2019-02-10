#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }

    #[test]
    fn cuda_is_available() {
        assert_eq!(unsafe { torch_cuda_is_available() }, true);
    }

    #[test]
    fn cudnn_is_available() {
        assert_eq!(unsafe { torch_cuda_cudnn_is_available() }, true);
    }
}
