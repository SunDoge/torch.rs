use std::cell::RefCell;
use std::rc::Rc;

pub trait WrappedPtr {
    type Ptr;
    fn as_ptr(&self) -> *mut Self::Ptr;
    fn as_mut_ptr(&mut self) -> *mut Self::Ptr;
}

pub struct IntrusivePtr<T> {
    ptr: Rc<RefCell<WrappedPtr<Ptr=T>>>,
}

impl<T> IntrusivePtr<T> {
    pub fn new(value: impl WrappedPtr<Ptr=T> + 'static) -> Self {
        IntrusivePtr {
            ptr: Rc::new(RefCell::new(value)),
        }
    }

    pub fn as_ptr(&self) -> *mut T {
        self.ptr.borrow().as_ptr()
    }

    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr.borrow_mut().as_mut_ptr()
    }
}

