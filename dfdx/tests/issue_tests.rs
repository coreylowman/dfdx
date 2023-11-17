use dfdx::prelude::*;
use std::fmt::Debug;

#[test]
fn test_issue_891() {
    #[derive(Default, Debug, Clone, Copy, CustomModule)]
    pub struct Id;

    impl<Input> Module<Input> for Id {
        type Output = Input;
        fn try_forward(&self, x: Input) -> Result<Self::Output, Error> {
            Ok(x)
        }
    }

    #[derive(Default, Debug, Clone, Copy, dfdx_derives::CustomModule)]
    struct ConcatTensorAlong<Ax: Axes<Array = [isize; 1]> + Debug>(pub Ax);

    impl<Input, const AXIS: isize> Module<Input> for ConcatTensorAlong<Axis<AXIS>>
    where
        Input: TryConcatTensorAlong<Axis<AXIS>>,
    {
        type Output = <Input as TryConcatTensorAlong<Axis<AXIS>>>::Output;

        fn try_forward(&self, x: Input) -> Result<Self::Output, Error> {
            x.try_concat_tensor_along(Axis)
        }
    }

    type Arch = (SplitInto<(Id, Id)>, ConcatTensorAlong<Axis<0>>);

    let dev = Cpu::default();
    let x = dev.tensor([1.]);
    let m = dev.build_module::<f32>(Arch::default());
    let _y: Tensor<Rank1<2>, _, _, _> = m.forward(x);
}
