use wgpu::{
    Adapter, BufferDescriptor, BufferUsages, Device, Instance, InstanceDescriptor, Maintain, Queue,
    RequestDeviceError,
};

use crate::{
    shapes::{Shape, Unit},
    tensor::{
        cache::TensorCache, cpu::Cpu, Cache, Error, NoneTape, RandomU64, Storage, Synchronize,
        Tensor,
    },
};
use super::resources::{unary_op_layout_desc, binary_op_layout_desc};

#[cfg(feature = "no-std")]
use spin::Mutex;

#[cfg(not(feature = "no-std"))]
use std::sync::Mutex;

use std::{marker::PhantomData, sync::Arc, vec::Vec};

use futures_lite::future::block_on;

use super::allocate::round_to_buffer_alignment;

#[derive(Debug)]
pub struct Buffer {
    pub(crate) data: wgpu::Buffer,
    pub(crate) size: usize,
}

impl core::ops::Deref for Buffer {
    type Target = wgpu::Buffer;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl Buffer {
    pub(crate) fn size(&self) -> usize {
        self.size
    }

    #[allow(unused)]
    pub(crate) fn capacity(&self) -> usize {
        self.data.size() as usize
    }

    pub(crate) fn copy_to_device<E: Unit>(&self, dev: &Device, queue: &Queue, slice: &[E]) {
        let slice = unsafe {
            std::slice::from_raw_parts(
                slice.as_ptr() as *const u8,
                slice.len() * std::mem::size_of::<E>(),
            )
        };
        queue.write_buffer(&self.data, 0, slice);
        queue.submit(std::iter::empty());
        dev.poll(Maintain::Wait);
    }

    pub(crate) fn copy_to_host<E: Unit>(&self, dev: &Device, queue: &Queue, buf: &mut [E]) {
        let (sender, receiver) = thingbuf::mpsc::channel(1);
        let buffer = dev.create_buffer(&BufferDescriptor {
            label: None,
            size: self.size() as u64,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        {
            let mut encoder = dev.create_command_encoder(&Default::default());
            encoder.copy_buffer_to_buffer(&self.data, 0, &buffer, 0, self.size() as u64);
            queue.submit(Some(encoder.finish()));
        }
        let slice = buffer.slice(..self.size() as u64);
        slice.map_async(wgpu::MapMode::Read, move |_| {
            futures_lite::future::block_on(sender.send(())).unwrap();
        });
        dev.poll(Maintain::Wait);

        let _ = futures_lite::future::block_on(receiver.recv());
        let data = slice.get_mapped_range();
        // TODO: How are we sure this is safe?
        let slice = unsafe {
            std::slice::from_raw_parts(
                data.as_ptr() as *const E,
                self.size() / std::mem::size_of::<E>(),
            )
        };
        buf.copy_from_slice(slice);
        drop(data);
        buffer.unmap();
    }
}

#[derive(Clone, Debug)]
pub struct Webgpu {
    pub(crate) cpu: Cpu,
    #[allow(unused)]
    pub(crate) instance: Arc<Instance>,
    #[allow(unused)]
    pub(crate) adapter: Arc<Adapter>,
    pub(crate) dev: Arc<Device>,
    pub(crate) queue: Arc<Queue>,

    pub(crate) cache: Arc<TensorCache<Buffer>>,

    // pipeline resources
    /// `[unary, binary]` pipeline layouts
    /// 
    /// storing them for re-use reduces resource allocation pressure on the GPU
    pub(super) layouts: [Arc<wgpu::BindGroupLayout>; 2]
}

impl From<RequestDeviceError> for Error {
    fn from(e: RequestDeviceError) -> Self {
        Error::WebgpuRequestDeviceError(e)
    }
}

impl Default for Webgpu {
    fn default() -> Self {
        Self::seed_from_u64(0)
    }
}

static CONSTRUCTOR_MUTEX: Mutex<()> = Mutex::new(());

impl Webgpu {
    pub fn seed_from_u64(seed: u64) -> Self {
        Self::try_build(seed).unwrap()
    }

    pub fn try_build(seed: u64) -> Result<Self, Error> {
        #[cfg(feature = "no-std")]
        let _lock = { CONSTRUCTOR_MUTEX.lock() };
        #[cfg(not(feature = "no-std"))]
        let _lock = { CONSTRUCTOR_MUTEX.lock().unwrap() };

        #[cfg(not(feature = "f16"))]
        let features: wgpu::Features = Default::default();
        #[cfg(feature = "f16")]
        let features: wgpu::Features = wgpu::Features::default() | wgpu::Features::SHADER_F16;

        let limits: wgpu::Limits = Default::default();
        let device_desc = wgpu::DeviceDescriptor {
            label: Some("dfdx"),
            features,
            limits
        };
        let adapter_desc = wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        };

        // request adapter
        let instance = Arc::new(Instance::new(InstanceDescriptor::default()));
        // note: may also fail b/c adapter doesn't support requested features/limits
        let adapter = block_on(instance.request_adapter(&adapter_desc))
            .ok_or(Error::WebgpuAdapterNotFound)?;
        let adapter = Arc::new(adapter);

        // request device from adapter
        let (dev, queue) =
            block_on(adapter.request_device(&device_desc, None))?;
        let dev = Arc::new(dev);
        let queue = Arc::new(queue);

        let cpu = Cpu::seed_from_u64(seed);

        let layouts = [
            Arc::new(dev.create_bind_group_layout(&unary_op_layout_desc())),
            Arc::new(dev.create_bind_group_layout(&binary_op_layout_desc()))
        ];

        Ok(Self {
            cpu,
            instance,
            adapter,
            dev,
            queue,

            cache: Default::default(),

            layouts
        })
    }

    pub(crate) fn submit_commands<F>(&self, command_builder: F) -> wgpu::SubmissionIndex
    where
        F: FnOnce(&mut wgpu::CommandEncoder),
    {
        let mut encoder = self
            .dev
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("submit_commands"),
            });
        command_builder(&mut encoder);
        let cmd = [encoder.finish()];
        return self.queue.submit(cmd);
    }

    /// Convienence function for submitting single-stage compute operations.
    /// 
    /// see: [`submit_commands`]
    pub(crate) fn submit_basic_op(
        &self,
        pipeline: &wgpu::ComputePipeline,
        params: &wgpu::BindGroup,
        label: Option<&str>,
        work_groups: &(u32, u32, u32),
    ) -> wgpu::SubmissionIndex {
        return self.submit_commands(|encoder| {
            let (x, y, z) = *work_groups;
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label, ..Default::default() });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, params, &[]);
            pass.dispatch_workgroups(x, y, z);
        })
    }
}

impl Webgpu {
    // todo: support configuration of usage flags
    pub(crate) unsafe fn alloc_empty<E>(&self, len: usize) -> Result<Buffer, Error> {
        let data = self.cache.try_pop::<E>(len).map_or_else(
            || Buffer {
                data: self.dev.create_buffer(&BufferDescriptor {
                    label: None,
                    size: round_to_buffer_alignment((len * std::mem::size_of::<E>()) as u64),
                    usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }),
                size: len * std::mem::size_of::<E>(),
            },
            |bfr| bfr,
        );
        Ok(data)
    }

    // #[allow(unused)]
    // pub(crate) unsafe fn get_workspace<E>(&self, len: usize) -> Result<MutexGuard<Buffer>, Error> {
    //     let num_bytes_required = len * std::mem::size_of::<E>();
    //     let mut workspace = self.workspace.as_ref().lock().unwrap();

    //     // re-allocate a larger workspace
    //     if (workspace.size() as usize) < num_bytes_required {
    //         *workspace = self.dev.create_buffer(&BufferDescriptor {
    //             label: None,
    //             size: (num_bytes_required) as u64,
    //             usage: BufferUsages::all(),
    //             mapped_at_creation: true,
    //         });
    //     }

    //     Ok(workspace)
    // }
}

#[derive(Debug)]
pub struct CachableBuffer<E> {
    pub(crate) dev: Arc<Device>,
    pub(crate) queue: Arc<Queue>,
    pub(crate) data: Buffer,
    pub(crate) cache: Arc<TensorCache<Buffer>>,
    pub(crate) _phantom: PhantomData<E>,
}

impl<E> Clone for CachableBuffer<E> {
    fn clone(&self) -> Self {
        let len = self.data.size() as usize / std::mem::size_of::<E>();
        let (encoder, data) = self.cache.try_pop::<E>(len).map_or_else(
            || {
                let mut encoder = self.dev.create_command_encoder(&Default::default());
                let bfr = self.dev.create_buffer(&BufferDescriptor {
                    label: None,
                    size: round_to_buffer_alignment(self.data.size() as u64),
                    usage: BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                encoder.copy_buffer_to_buffer(&self.data, 0, &bfr, 0, self.data.size() as u64);
                (
                    encoder,
                    Buffer {
                        data: bfr,
                        size: self.data.size as usize,
                    },
                )
            },
            |bfr| {
                let mut encoder = self.dev.create_command_encoder(&Default::default());
                encoder.copy_buffer_to_buffer(&self.data, 0, &bfr, 0, self.data.size() as u64);
                (encoder, bfr)
            },
        );
        self.queue.submit(Some(encoder.finish()));
        Self {
            dev: self.dev.clone(),
            queue: self.queue.clone(),
            data,
            cache: self.cache.clone(),
            _phantom: PhantomData,
        }
    }
}

impl<E> std::ops::Deref for CachableBuffer<E> {
    type Target = Buffer;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl<E> std::ops::DerefMut for CachableBuffer<E> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.data
    }
}

impl<E> Drop for CachableBuffer<E> {
    fn drop(&mut self) {
        if self.cache.is_enabled() {
            let data = std::mem::replace(
                &mut self.data,
                Buffer {
                    data: self.dev.create_buffer(&BufferDescriptor {
                        label: None,
                        size: 0,
                        usage: BufferUsages::MAP_READ,
                        mapped_at_creation: false,
                    }),
                    size: 0,
                },
            );
            let len = data.size() as usize / std::mem::size_of::<E>();
            self.cache.insert::<E>(len, data);
        }
    }
}

impl RandomU64 for Webgpu {
    fn random_u64(&self) -> u64 {
        self.cpu.random_u64()
    }
}

impl Cache for Webgpu {
    fn try_enable_cache(&self) -> Result<(), Error> {
        self.cache.enable();
        Ok(())
    }

    fn try_disable_cache(&self) -> Result<(), Error> {
        self.cache.disable();
        self.try_empty_cache()
    }

    fn try_empty_cache(&self) -> Result<(), Error> {
        #[cfg(not(feature = "no-std"))]
        let mut cache = self.cache.allocations.write().unwrap();
        #[cfg(feature = "no-std")]
        let mut cache = self.cache.allocations.write();
        for (&_key, allocations) in cache.iter_mut() {
            for alloc in allocations.drain(..) {
                drop(alloc);
            }
        }
        cache.clear();
        Ok(())
    }
}

impl Synchronize for Webgpu {
    fn try_synchronize(&self) -> Result<(), Error> {
        self.dev.poll(wgpu::MaintainBase::Wait);
        Ok(())
    }
}

impl<E: Unit> Storage<E> for Webgpu {
    type Vec = CachableBuffer<E>;

    fn try_alloc_len(&self, len: usize) -> Result<Self::Vec, Error> {
        let data = unsafe { self.alloc_empty::<E>(len) }?;
        Ok(CachableBuffer {
            dev: self.dev.clone(),
            queue: self.queue.clone(),
            data,
            cache: self.cache.clone(),
            _phantom: PhantomData,
        })
    }

    fn len(&self, v: &Self::Vec) -> usize {
        v.size() as usize / std::mem::size_of::<E>()
    }

    fn tensor_to_vec<S: Shape, T>(&self, tensor: &Tensor<S, E, Self, T>) -> Vec<E> {
        let buf = self
            .cpu
            .try_alloc_elem::<E>(
                tensor.data.data.size() as usize / std::mem::size_of::<E>(),
                Default::default(),
            )
            .unwrap();
        let mut cpu_tensor = Tensor {
            id: tensor.id,
            data: Arc::new(buf),
            shape: tensor.shape,
            strides: tensor.strides,
            device: self.cpu.clone(),
            tape: NoneTape,
        };
        let buf = Arc::get_mut(&mut cpu_tensor.data).unwrap();
        tensor.data.copy_to_host::<E>(&self.dev, &self.queue, buf);
        self.cpu.tensor_to_vec::<S, _>(&cpu_tensor)
    }
}
