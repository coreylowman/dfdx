use crate::{shapes::*, tensor_ops::*};

use super::module::{BuildModule, BuildOnDevice, DeviceStorage, Module, ModuleMut, OnDevice};

use super::{TensorFunction, TensorVisitor, ToDevice, VisitTensors};

macro_rules! tuple_impls {
    ([$($name:ident),+] [$($idx:tt),+], $last:ident, [$($rev_tail:ident),+]) => {
        impl<
            $($name: VisitTensors<E, D> + std::fmt::Debug,)+
            E: Dtype,
            D: DeviceStorage,
        > VisitTensors<E, D> for ($($name,)+)
        {
            fn visit_groups<const N: usize, const M: usize, Func: TensorFunction<N, M, E, D>>(
                mut visitor: TensorVisitor<N, M, Self, Func>,
            ) -> Result<(), Func::Err> {
                $(visitor.visit_field(|s| &s.$idx, |s| &mut s.$idx, &std::format!("{}.", $idx))?;)+
                Ok(())
            }
        }

        impl<D: Device<E>, E: Dtype, $($name: BuildOnDevice<D, E>),+> BuildOnDevice<D, E> for ($($name,)+) {
            type Built = ($($name::Built, )+);
        }

        impl<D: Device<E>, E: Dtype, $($name: BuildModule<D, E>),+> BuildModule<D, E> for ($($name,)+) {
            #[allow(non_snake_case)]
            fn try_build(device: &D) -> Result<Self, D::Err> {
                $(let $name = BuildModule::try_build(device)?;)*
                Ok(($($name, )*))
            }
        }

        impl<$($name: ToDevice<D>,)+ D> ToDevice<D> for ($($name,)+) {
            type Output = ($(OnDevice<$name, D>,)+);
            fn to_device(&self, device: &D) -> Self::Output {
                ($(self.$idx.to_device(device)),+)
            }
        }

        /*This macro expands like this for a 4-tuple:

        impl<
            Input: Tensor,

            // `$last:`
            D:

            // `$(Module::<$rev_tail ::Output>, $rev_tail: )+`
            Module<C ::Output>, C:
            Module<B ::Output>, B:
            Module<A ::Output>, A:

            Module<Input>
        > Module<Input> for (A, B, C, D) {
            type Output = D::Output;
            fn forward(&self, x: Input) -> Self::Output {
                let x = self.0.forward(x);
                let x = self.1.forward(x);
                let x = self.2.forward(x);
                let x = self.3.forward(x);
                x
            }
        }
        */
        impl<
            Input,
            $last:
            $(Module::<$rev_tail ::Output>, $rev_tail: )+
            Module<Input>
        > Module<Input> for ($($name,)+) {
            type Output = $last ::Output;

            /// Calls forward sequentially on each module in the tuple.
            fn forward(&self, x: Input) -> Self::Output {
                $(let x = self.$idx.forward(x);)+
                x
            }
        }

        impl<
            Input,
            $last:
            $(ModuleMut::<$rev_tail ::Output>, $rev_tail: )+
            ModuleMut<Input>
        > ModuleMut<Input> for ($($name,)+) {
            type Output = $last ::Output;

            /// Calls forward sequentially on each module in the tuple.
            fn forward_mut(&mut self, x: Input) -> Self::Output {
                $(let x = self.$idx.forward_mut(x);)+
                x
            }
        }
    };
}

tuple_impls!([M1, M2] [0, 1], M2, [M1]);
tuple_impls!([M1, M2, M3] [0, 1, 2], M3, [M2, M1]);
tuple_impls!([M1, M2, M3, M4] [0, 1, 2, 3], M4, [M3, M2, M1]);
tuple_impls!([M1, M2, M3, M4, M5] [0, 1, 2, 3, 4], M5, [M4, M3, M2, M1]);
tuple_impls!([M1, M2, M3, M4, M5, M6] [0, 1, 2, 3, 4, 5], M6, [M5, M4, M3, M2, M1]);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::tests::SimpleUpdater;
    use crate::tests::TestDtype;
    use crate::unique_id::HasUniqueId;
    use crate::{
        nn::{builders::*, *},
        optim::*,
        tensor::*,
        tests::TestDevice,
    };

    #[test]
    fn test_2_tuple() {
        let dev: TestDevice = Default::default();

        let model: (ReLU, Tanh) = Default::default();

        let x = dev.tensor([-2.0, -1.0, 0.0, 1.0, 2.0]);
        let y = model.forward(x);
        assert_eq!(y.array(), [0.0, 0.0, 0.0, 1.0f32.tanh(), 2.0f32.tanh()]);
    }

    #[test]
    fn test_2_tuple_update() {
        let dev: TestDevice = Default::default();
        type Model = (Linear<2, 3>, Linear<3, 4>);
        let mut model = Model::build_on_device(&dev);
        assert_ne!(model.0.weight.array(), [[0.0; 2]; 3]);
        assert_ne!(model.0.bias.array(), [0.0; 3]);
        assert_ne!(model.1.weight.array(), [[0.0; 3]; 4]);
        assert_ne!(model.1.bias.array(), [0.0; 4]);

        let m0 = model.clone();

        let x = dev.sample_normal::<Rank1<2>>().traced();
        let loss = model.forward_mut(x).square().mean();
        let g = loss.backward();

        assert_ne!(g.get(&model.0.weight).array(), [[0.0; 2]; 3]);
        assert_ne!(g.get(&model.0.bias).array(), [0.0; 3]);
        assert_ne!(g.get(&model.1.weight).array(), [[0.0; 3]; 4]);
        assert_ne!(g.get(&model.1.bias).array(), [0.0; 4]);

        let mut sgd = Sgd::new(
            &model,
            SgdConfig {
                lr: 1.0,
                momentum: None,
                weight_decay: None,
            },
        );
        sgd.update(&mut model, g).unwrap();

        assert_ne!(model.0.weight.array(), m0.0.weight.array());
        assert_ne!(model.0.bias.array(), m0.0.bias.array());
        assert_ne!(model.1.weight.array(), m0.1.weight.array());
        assert_ne!(model.1.bias.array(), m0.1.bias.array());
    }

    /// A struct to test the forward method of tuples. This sets the `I`th valuein a 1d tensors of size `N` to 1.0.
    #[derive(Debug, Default, Clone)]
    struct SetTo1<const I: usize, const N: usize>;
    impl<const I: usize, const N: usize> ZeroSizedModule for SetTo1<I, N> {}
    impl<const I: usize, const N: usize> Module<Tensor<Rank1<N>, f32, Cpu>> for SetTo1<I, N> {
        type Output = Tensor<Rank1<N>, f32, Cpu>;
        fn forward(&self, mut input: Tensor<Rank1<N>, f32, Cpu>) -> Self::Output {
            std::sync::Arc::make_mut(&mut input.storage.data)[I] = 1.0;
            input
        }
    }

    #[test]
    fn test_set_to_1() {
        let dev: Cpu = Default::default();
        assert_eq!(
            SetTo1::<0, 5>::default().forward(dev.zeros()).array(),
            [1.0, 0.0, 0.0, 0.0, 0.0]
        );

        assert_eq!(
            SetTo1::<1, 5>::default().forward(dev.zeros()).array(),
            [0.0, 1.0, 0.0, 0.0, 0.0]
        );

        assert_eq!(
            SetTo1::<2, 5>::default().forward(dev.zeros()).array(),
            [0.0, 0.0, 1.0, 0.0, 0.0]
        );

        assert_eq!(
            SetTo1::<3, 5>::default().forward(dev.zeros()).array(),
            [0.0, 0.0, 0.0, 1.0, 0.0]
        );

        assert_eq!(
            SetTo1::<4, 5>::default().forward(dev.zeros()).array(),
            [0.0, 0.0, 0.0, 0.0, 1.0]
        );
    }

    #[test]
    fn test_2_tuple_forward() {
        let dev: Cpu = Default::default();
        let model: (SetTo1<0, 2>, SetTo1<1, 2>) = Default::default();
        let y = model.forward(dev.zeros());
        assert_eq!(y.array(), [1.0, 1.0]);
    }

    #[test]
    fn test_3_tuple_forward() {
        let dev: Cpu = Default::default();
        let model: (SetTo1<0, 3>, SetTo1<1, 3>, SetTo1<2, 3>) = Default::default();
        let y = model.forward(dev.zeros());
        assert_eq!(y.array(), [1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_4_tuple_forward() {
        let dev: Cpu = Default::default();
        let model: (SetTo1<0, 4>, SetTo1<1, 4>, SetTo1<2, 4>, SetTo1<3, 4>) = Default::default();
        let y = model.forward(dev.zeros());
        assert_eq!(y.array(), [1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_5_tuple_forward() {
        let dev: Cpu = Default::default();
        let model: (
            SetTo1<0, 5>,
            SetTo1<1, 5>,
            SetTo1<2, 5>,
            SetTo1<3, 5>,
            SetTo1<4, 5>,
        ) = Default::default();
        let y = model.forward(dev.zeros());
        assert_eq!(y.array(), [1.0, 1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_6_tuple_forward() {
        let dev: Cpu = Default::default();
        let model: (
            SetTo1<0, 6>,
            SetTo1<1, 6>,
            SetTo1<2, 6>,
            SetTo1<3, 6>,
            SetTo1<4, 6>,
            SetTo1<5, 6>,
        ) = Default::default();
        let y = model.forward(dev.zeros());
        assert_eq!(y.array(), [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_tuple_missing_gradients() {
        let dev: TestDevice = Default::default();
        type Model = (Linear<5, 3>, Linear<5, 3>, Linear<5, 3>);
        let mut model = dev.build_module::<Model, TestDtype>();
        let mut g: SimpleUpdater = Default::default();

        // no gradients present
        let mut unused: UnusedTensors = Default::default();
        model.update(&mut g, &mut unused).unwrap();
        assert_eq!(
            &unused.ids,
            &[
                *model.0.weight.id(),
                *model.0.bias.id(),
                *model.1.weight.id(),
                *model.1.bias.id(),
                *model.2.weight.id(),
                *model.2.bias.id(),
            ]
        );

        // weight gradient is present
        g.0.try_alloc_for(&model.0.weight).unwrap();
        g.0.try_alloc_for(&model.1.weight).unwrap();
        g.0.try_alloc_for(&model.2.weight).unwrap();

        let mut unused: UnusedTensors = Default::default();
        model.update(&mut g, &mut unused).unwrap();
        assert_eq!(
            &unused.ids,
            &[*model.0.bias.id(), *model.1.bias.id(), *model.2.bias.id(),]
        );

        g.0.try_alloc_for(&model.0.weight).unwrap();
        g.0.try_alloc_for(&model.0.bias).unwrap();
        g.0.try_alloc_for(&model.1.weight).unwrap();
        g.0.try_alloc_for(&model.1.bias).unwrap();
        g.0.try_alloc_for(&model.2.weight).unwrap();
        g.0.try_alloc_for(&model.2.bias).unwrap();

        let mut unused: UnusedTensors = Default::default();
        model.update(&mut g, &mut unused).unwrap();
        assert!(unused.is_empty());
    }
}
