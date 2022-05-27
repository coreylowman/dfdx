use super::Cpu;
use crate::arrays::CountElements;
use std::alloc::{alloc_zeroed, Layout};

/// Allocate an Nd array on the heap.
pub trait AllocateZeros {
    /// Allocate T directly on the heap.
    fn zeros<T: CountElements>() -> Box<T>;

    /// Copy `inp` into `out`.
    fn copy<T: CountElements>(inp: &T, out: &mut T);
}

impl AllocateZeros for Cpu {
    /// Allocates using [alloc_zeroed].
    fn zeros<T: CountElements>() -> Box<T> {
        // TODO is this function safe for any T?
        // TODO move to using safe code once we can allocate an array directly on the heap.
        let layout = Layout::new::<T>();
        debug_assert_eq!(layout.size(), T::NUM_BYTES);
        unsafe {
            let ptr = alloc_zeroed(layout) as *mut T;
            Box::from_raw(ptr)
        }
    }

    /// Copies use [std::ptr::copy_nonoverlapping()].
    fn copy<T: CountElements>(inp: &T, out: &mut T) {
        unsafe { std::ptr::copy_nonoverlapping(inp as *const T, out as *mut T, 1) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{prelude::StdRng, Rng, SeedableRng};

    #[test]
    fn test_0d_zeros() {
        let t: Box<f32> = Cpu::zeros();
        assert_eq!(t.as_ref(), &0.0);
    }

    #[test]
    fn test_0d_copy() {
        let a = 3.14;
        let mut b = 0.0;
        Cpu::copy(&a, &mut b);
        assert_eq!(a, b);
    }

    #[test]
    fn test_1d_zeros() {
        let t: Box<[f32; 5]> = Cpu::zeros();
        assert_eq!(t.as_ref(), &[0.0; 5]);
    }

    #[test]
    fn test_1d_copy() {
        let a = [1.0, 2.0, 3.0];
        let mut b = [0.0; 3];
        Cpu::copy(&a, &mut b);
        assert_eq!(a, b);
    }

    #[test]
    fn test_2d_zeros() {
        let t: Box<[[f32; 3]; 5]> = Cpu::zeros();
        assert_eq!(t.as_ref(), &[[0.0; 3]; 5]);
    }

    #[test]
    fn test_2d_copy() {
        let a = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let mut b = [[0.0; 3]; 2];
        Cpu::copy(&a, &mut b);
        assert_eq!(a, b);
    }

    #[test]
    fn test_3d_zeros() {
        let t: Box<[[[f32; 2]; 3]; 5]> = Cpu::zeros();
        assert_eq!(t.as_ref(), &[[[0.0; 2]; 3]; 5]);
    }

    #[test]
    fn test_3d_copy() {
        let a = [
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            [[-1.0, -2.0, -3.0], [-4.0, -5.0, -6.0]],
        ];
        let mut b = [[[0.0; 3]; 2]; 2];
        Cpu::copy(&a, &mut b);
        assert_eq!(a, b);
    }

    #[test]
    fn test_zeros_0d() {
        let mut data: Box<f32> = Cpu::zeros();
        assert_eq!(data.as_ref(), &0.0);
        *data = 1.0;
        assert_eq!(data.as_ref(), &1.0);
        *data = -1.0;
        assert_eq!(data.as_ref(), &-1.0);
    }

    #[test]
    fn test_zeros_1d() {
        let mut data: Box<[f32; 5]> = Cpu::zeros();
        assert_eq!(data.as_ref(), &[0.0; 5]);
        for i in 0..5 {
            data[i] = i as f32;
        }
        assert_eq!(data.as_ref(), &[0.0, 1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_zeros_2d() {
        let mut data: Box<[[f32; 3]; 2]> = Cpu::zeros();
        assert_eq!(data.as_ref(), &[[0.0; 3]; 2]);
        data[0][0] = 0.0;
        data[0][1] = 1.0;
        data[0][2] = 2.0;
        data[1][0] = 3.0;
        data[1][1] = 4.0;
        data[1][2] = 5.0;
        assert_eq!(data.as_ref(), &[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);
    }

    #[test]
    fn test_alloc_large() {
        const N: usize = 1_000_000;
        let mut rng = StdRng::seed_from_u64(0);
        // NOTE: using [0.0f32; N] causes stack overflow
        let mut q: Box<[f32; N]> = Cpu::zeros();
        for i in 0..N {
            q[i] = rng.gen();
        }
        let sum: f32 = q.iter().sum();
        assert!(sum > 0.0);
    }
}
