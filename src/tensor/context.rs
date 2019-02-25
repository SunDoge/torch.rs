// use std::sync::Mutex;
use torch_sys::*;

lazy_static! {
    // static ref STATE: Mutex<State> = Mutex::new(State::new().init());
    static ref STATE: State = State::new();
}

pub struct State {
    ptr: *mut THCState,
}

impl State {
    pub fn new() -> State {
        let ptr = unsafe { THCState_alloc() };
        unsafe {
            THCudaInit(ptr);
        }

        State { ptr }
    }

    pub fn as_ptr(&self) -> *mut THCState {
        self.ptr
    }

    pub fn as_mut_ptr(&mut self) -> *mut THCState {
        self.ptr
    }
}

unsafe impl Sync for State {}
unsafe impl Send for State {}

impl Drop for State {
    fn drop(&mut self) {
        unsafe {
            THCState_free(self.ptr);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use torch_sys::*;

    #[test]
    fn multi_state() {
        // let mut _state1 = State::new();
        // let _state2 = State::new();

        unsafe {
            // let ptr = THCState_alloc();
            // THCudaInit(ptr);

            // let arc_ptr = std::sync::Arc::new(ptr);
            // THCudaInit(_state1.ptr);

            let n = 10;
            let mut children = Vec::with_capacity(n);
            for i in 0..n {
                children.push(std::thread::spawn(|| {
                    let prop = THCState_getNumDevices(STATE.as_ptr());
                    //println!("num devices: {}", prop);
                    let t1 = THCudaTensor_newWithSize1d(STATE.as_ptr(), 1);
                    THCudaTensor_free(STATE.as_ptr(), t1);
                }));
            }

            for child in children {
                child.join();
            }

            // let prop = THCState_getNumDevices(ptr);
            // println!("num devices: {}", prop);
            // let t1 = THCudaTensor_newWithSize1d(ptr, 1);
            // THCudaTensor_free(ptr, t1);
            // THCState_free(ptr);
        }
    }
}
