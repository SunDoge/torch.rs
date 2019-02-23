use torch_sys::*;

pub struct Generator {
    ptr: *mut THGenerator,
}

impl Generator {
    pub fn new() -> Generator {
        Generator {
            ptr: unsafe { THGenerator_new() },
        }
    }
}

impl Drop for Generator {
    fn drop(&mut self) {
        unsafe {
            THGenerator_free(self.ptr);
        }
    }
}
