use std::cell::RefCell;
use std::rc::Rc;

pub struct IntrusivePtr<T> {
    ptr: Rc<RefCell<T>>,
}

impl<T> IntrusivePtr<T> {
    pub fn new(value: T) -> Self {
        IntrusivePtr {
            ptr: Rc::new(RefCell::new(value)),
        }
    }
}
