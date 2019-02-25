#[repr(u16)]
#[derive(PartialEq, Clone, Copy)]
pub enum DeviceType {
    CPU = 0,
    CUDA = 1,   // CUDA.
    MKLDNN = 2, // Reserved for explicit MKLDNN
    OPENGL = 3, // OpenGL
    OPENCL = 4, // OpenCL
    IDEEP = 5,  // IDEEP.
    HIP = 6,    // AMD HIP
    FPGA = 7,   // FPGA
    // NB: If you add more devices:
    //  - Change the implementations of DeviceTypeName and isValidDeviceType
    //    in DeviceType.cpp
    //  - Change the number below
    COMPILE_TIME_MAX_DEVICE_TYPES = 8,
    ONLY_FOR_TEST = 20901, // This device type is only for test.
}

// For generic
pub struct CPU;
pub struct CUDA;

pub type DeviceIndex = i16;

impl Default for DeviceType {
    fn default() -> Self {
        DeviceType::CPU
    }
}

impl DeviceType {
    pub fn is_cpu(&self) -> bool {
        *self == DeviceType::CPU
    }

    pub fn is_cuda(&self) -> bool {
        *self == DeviceType::CUDA
    }
}

#[derive(PartialEq)]
pub struct Device {
    device_type: DeviceType,
    index: DeviceIndex,
}

impl Default for Device {
    fn default() -> Device {
        Device {
            device_type: Default::default(),
            index: -1,
        }
    }
}

impl Device {
    pub fn new(device_type: DeviceType, index: DeviceIndex) -> Device {
        // Have no idea why not index >= -1
        assert!(
            index == -1 || index >= 0,
            "Device index must be -1 or non-negative, got {}",
            index
        );
        assert!(
            !device_type.is_cpu() || index <= 0,
            "CPU device index must be -1 or zero, got {}",
            index
        );
        Device { device_type, index }
    }

    pub fn set_index(&mut self, index: DeviceIndex) {
        self.index = index;
    }

    pub fn index(&self) -> DeviceIndex {
        self.index
    }

    pub fn has_index(&self) -> bool {
        self.index != -1
    }

    pub fn device_type(&self) -> DeviceType {
        self.device_type
    }

    pub fn is_cuda(&self) -> bool {
        self.device_type.is_cuda()
    }

    pub fn is_cpu(&self) -> bool {
        self.device_type.is_cpu()
    }
}
