use super::*;

pub trait Tensor: IsShapedArray + CanUpdateWithTape + HasUniqueId {
    type TapeHolder: TapeHolder;
    type NoTape: Tensor<TapeHolder = NoTape, Dimension = Self::Dimension>;
    type WithTape: Tensor<TapeHolder = WithTape, Dimension = Self::Dimension>;

    fn split_tape_holder(self) -> (Self::NoTape, Self::TapeHolder);
}

impl<H: TapeHolder> Tensor for Tensor0D<H> {
    type TapeHolder = H;
    type NoTape = Tensor0D<NoTape>;
    type WithTape = Tensor0D<WithTape>;

    fn split_tape_holder(self) -> (Self::NoTape, Self::TapeHolder) {
        (
            Self::NoTape {
                id: self.id,
                data: self.data,
                tape: NoTape::default(),
            },
            self.tape,
        )
    }
}

impl<const N: usize, H: TapeHolder> Tensor for Tensor1D<N, H> {
    type TapeHolder = H;
    type NoTape = Tensor1D<N, NoTape>;
    type WithTape = Tensor1D<N, WithTape>;

    fn split_tape_holder(self) -> (Self::NoTape, Self::TapeHolder) {
        (
            Self::NoTape {
                id: self.id,
                data: self.data,
                tape: NoTape::default(),
            },
            self.tape,
        )
    }
}

impl<const M: usize, const N: usize, H: TapeHolder> Tensor for Tensor2D<M, N, H> {
    type TapeHolder = H;
    type NoTape = Tensor2D<M, N, NoTape>;
    type WithTape = Tensor2D<M, N, WithTape>;

    fn split_tape_holder(self) -> (Self::NoTape, Self::TapeHolder) {
        (
            Self::NoTape {
                id: self.id,
                data: self.data,
                tape: NoTape::default(),
            },
            self.tape,
        )
    }
}

impl<const M: usize, const N: usize, const O: usize, H: TapeHolder> Tensor
    for Tensor3D<M, N, O, H>
{
    type TapeHolder = H;
    type NoTape = Tensor3D<M, N, O, NoTape>;
    type WithTape = Tensor3D<M, N, O, WithTape>;

    fn split_tape_holder(self) -> (Self::NoTape, Self::TapeHolder) {
        (
            Self::NoTape {
                id: self.id,
                data: self.data,
                tape: NoTape::default(),
            },
            self.tape,
        )
    }
}
impl<const M: usize, const N: usize, const O: usize, const P: usize, H: TapeHolder> Tensor
    for Tensor4D<M, N, O, P, H>
{
    type TapeHolder = H;
    type NoTape = Tensor4D<M, N, O, P, NoTape>;
    type WithTape = Tensor4D<M, N, O, P, WithTape>;

    fn split_tape_holder(self) -> (Self::NoTape, Self::TapeHolder) {
        (
            Self::NoTape {
                id: self.id,
                data: self.data,
                tape: NoTape::default(),
            },
            self.tape,
        )
    }
}
