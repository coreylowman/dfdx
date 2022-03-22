use super::*;

impl Default for Tensor0D<NoTape> {
    fn default() -> Self {
        <Self as TensorCreator>::zeros()
    }
}

impl<const N: usize> Default for Tensor1D<N, NoTape> {
    fn default() -> Self {
        <Self as TensorCreator>::zeros()
    }
}

impl<const M: usize, const N: usize> Default for Tensor2D<M, N, NoTape> {
    fn default() -> Self {
        <Self as TensorCreator>::zeros()
    }
}

impl<const M: usize, const N: usize, const O: usize> Default for Tensor3D<M, N, O, NoTape> {
    fn default() -> Self {
        <Self as TensorCreator>::zeros()
    }
}

impl<const M: usize, const N: usize, const O: usize, const P: usize> Default
    for Tensor4D<M, N, O, P, NoTape>
{
    fn default() -> Self {
        <Self as TensorCreator>::zeros()
    }
}
