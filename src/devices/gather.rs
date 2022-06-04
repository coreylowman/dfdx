use super::{AllocateZeros, Cpu};
use crate::arrays::CountElements;

/// Something that can be indexed by a usize array ([GatherElements::Indices]).
///
/// Output type is [GatherElements::Gathered].
pub trait GatherElements<T: CountElements>: AllocateZeros {
    type Indices: CountElements<Dtype = usize>;
    type Gathered: CountElements<Dtype = T::Dtype>;

    fn gather_into(inp: &T, indices: &Self::Indices, out: &mut Self::Gathered);

    fn gather(inp: &T, indices: &Self::Indices) -> Box<Self::Gathered> {
        let mut out = Self::zeros();
        Self::gather_into(inp, indices, &mut out);
        out
    }
}

impl GatherElements<f32> for Cpu {
    type Indices = usize;
    type Gathered = f32;

    fn gather_into(inp: &f32, _: &Self::Indices, out: &mut Self::Gathered) {
        *out = *inp;
    }
}

impl<const M: usize> GatherElements<[f32; M]> for Cpu {
    type Indices = usize;
    type Gathered = f32;

    fn gather_into(inp: &[f32; M], indices: &Self::Indices, out: &mut Self::Gathered) {
        *out = inp[*indices];
    }
}

impl<const M: usize, const N: usize> GatherElements<[[f32; N]; M]> for Cpu {
    type Indices = [usize; M];
    type Gathered = [f32; M];

    fn gather_into(inp: &[[f32; N]; M], indices: &Self::Indices, out: &mut Self::Gathered) {
        for i in 0..M {
            Self::gather_into(&inp[i], &indices[i], &mut out[i]);
        }
    }
}

impl<const M: usize, const N: usize, const O: usize> GatherElements<[[[f32; O]; N]; M]> for Cpu {
    type Indices = [[usize; N]; M];
    type Gathered = [[f32; N]; M];

    fn gather_into(inp: &[[[f32; O]; N]; M], indices: &Self::Indices, out: &mut Self::Gathered) {
        for i in 0..M {
            Self::gather_into(&inp[i], &indices[i], &mut out[i]);
        }
    }
}

impl<const M: usize, const N: usize, const O: usize, const P: usize>
    GatherElements<[[[[f32; P]; O]; N]; M]> for Cpu
{
    type Indices = [[[usize; O]; N]; M];
    type Gathered = [[[f32; O]; N]; M];

    fn gather_into(
        inp: &[[[[f32; P]; O]; N]; M],
        indices: &Self::Indices,
        out: &mut Self::Gathered,
    ) {
        for i in 0..M {
            Self::gather_into(&inp[i], &indices[i], &mut out[i]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_1d_gather() {
        let t = [0.0, 1.0, 2.0];
        let i = 1;
        let mut out = 0.0;
        Cpu::gather_into(&t, &i, &mut out);
        assert_eq!(out, 1.0);
    }

    #[test]
    fn test_2d_gather() {
        let t = [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]];
        let i = [1, 2, 0];
        let mut out = [0.0, 0.0, 0.0];
        Cpu::gather_into(&t, &i, &mut out);
        assert_eq!(out, [1.0, 5.0, 6.0]);
    }

    #[test]
    fn test_3d_gather() {
        let t = [
            [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]],
            [[-0.0, -1.0, -2.0], [-3.0, -4.0, -5.0], [-6.0, -7.0, -8.0]],
        ];
        let i = [[2, 1, 0], [1, 0, 1]];
        let mut out = [[0.0; 3]; 2];
        Cpu::gather_into(&t, &i, &mut out);
        assert_eq!(out, [[2.0, 4.0, 6.0], [-1.0, -3.0, -7.0]]);
    }
}
