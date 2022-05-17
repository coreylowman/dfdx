use super::Cpu;
use crate::arrays::HasInner;

pub trait MapInnerElements<T: HasInner> {
    fn map_assign_inner<F: FnMut(&mut T::Inner) + Copy>(out: &mut T, f: F);
}

impl<const M: usize> MapInnerElements<[f32; M]> for Cpu {
    fn map_assign_inner<F: FnMut(&mut <[f32; M] as HasInner>::Inner) + Copy>(
        out: &mut [f32; M],
        mut f: F,
    ) {
        f(out)
    }
}

impl<T: HasInner, const M: usize> MapInnerElements<[T; M]> for Cpu
where
    Cpu: MapInnerElements<T>,
{
    fn map_assign_inner<F: FnMut(&mut <[T; M] as HasInner>::Inner) + Copy>(out: &mut [T; M], f: F) {
        for i in 0..M {
            Self::map_assign_inner(&mut out[i], f);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_1d_map_inner() {
        let mut t = [0.0; 5];
        Cpu::map_assign_inner(&mut t, |f| {
            f[0] = 1.0;
            f[4] = 2.0;
        });
        assert_eq!(t, [1.0, 0.0, 0.0, 0.0, 2.0]);
    }

    #[test]
    fn test_2d_map_inner() {
        let mut t = [[0.0; 3], [1.0; 3], [2.0; 3], [3.0; 3], [4.0; 3]];
        Cpu::map_assign_inner(&mut t, |f| *f = [f.iter().sum(); 3]);
        assert_eq!(t, [[0.0; 3], [3.0; 3], [6.0; 3], [9.0; 3], [12.0; 3]])
    }

    #[test]
    fn test_3d_map_inner() {
        let mut t = [[[1.0; 2]; 3]; 5];
        Cpu::map_assign_inner(&mut t, |f| *f = [2.0; 2]);
        assert_eq!(t, [[[2.0; 2]; 3]; 5]);
    }
}
