use crate::prelude::*;
use std::marker::PhantomData;

// TODO: try making a Call module, whih allows calling an arbitrary method on the input.

/// Access the input that is stored in a wrapper structure.
#[derive(
    Debug, Default, Clone, ResetParams, ZeroGrads, UpdateParams, LoadSafeTensors, SaveSafeTensors,
)]
#[repr(transparent)]
pub struct On<N, T> {
    #[module]
    #[serialize]
    pub t: T,

    pub _n: PhantomData<N>,
}

impl<E: Dtype, D: Device<E>, N: Clone + std::fmt::Debug, T: BuildOnDevice<E, D>> BuildOnDevice<E, D>
    for On<N, T>
{
    type Built = On<N, T::Built>;
    fn try_build_on_device(&self, device: &D) -> Result<Self::Built, Error> {
        let t = self.t.try_build_on_device(device)?;
        Ok(On { t, _n: PhantomData })
    }
}

// TODO: define On access for standard tuples,
// so that it's possible to access them with something like:
// On<tuple::_0, T>
pub mod tuple {}

// cargo 'test' '--package' 'dfdx' '--lib' '--' 'nn::layers::on::tests' '--nocapture'
// test based on nn/layers/residual_add.rs
#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::*;

    #[input_wrapper]
    pub struct MyWrapper<A, B> {
        pub a: A,
        pub b: B,
    }


    #[input_wrapper]
    pub struct Split1<Forward, Skip> {
        pub forward: Forward,
        pub skip: Skip,
    }

    #[derive(Default, Clone, Sequential)]
    pub struct ResidualAdd1<T: Clone + std::fmt::Debug> {
        // input is Input
        pub split: SplitInto<(Id, Id)>,

        // input is (Input, Input)
        pub input_to_wrapper: split1::FromTuple,

        // input is Split1 { Input, Input }
        pub t: On<split1::forward, T>,

        // input is Split1 { T::Output, Input }
        pub input_to_tuple: split1::IntoTuple,

        // input is (T::Output, Input)
        pub add: ops::Add,
        // input is T::Output = Input
    }

    #[test]
    fn test_residual_add_backward() {
        let dev: TestDevice = Default::default();

        let model = dev.build_module::<f32>(<ResidualAdd1<LinearConstConfig<2, 2>>>::default());

        let x: Tensor<Rank2<4, 2>, f32, _> = dev.sample_normal();
        let x = x.to_dtype::<TestDtype>();
        let y = model.forward(x.leaky_trace());

        #[rustfmt::skip]
        assert_close_to_literal!(y, [[0.25372928, -2.4258814],[1.7892148, -2.6242268],[1.5131638, 0.23407778],[3.4201493, 1.597525]]);

        let g = y.mean().backward();
        assert_close_to_literal!(g.get(&model.t.t.weight), [[0.475242, -0.075136]; 2]);
        assert_close_to_literal!(g.get(&model.t.t.bias), [0.5; 2]);
        assert_close_to_literal!(g.get(&x), [[0.18806472, 0.21419683]; 4]);
    }

    #[input_wrapper]
    pub struct Split2<Forward, Skip>(Forward, Skip);

    #[derive(Default, Clone, Sequential)]
    pub struct ResidualAdd2<T: Clone + std::fmt::Debug> {
        // input is Input
        pub split: SplitInto<(Id, Id)>,

        // input is (Input, Input)
        pub input_to_wrapper: split2::FromTuple,

        // input is Split2 ( Input, Input )
        pub t: On<split2::_0, T>,

        // input is Split2 ( T::Output, Input )
        pub input_to_tuple: split2::IntoTuple,

        // input is (T::Output, Input)
        pub add: ops::Add,
        // input is T::Output = Input
    }

    #[test]
    fn test_residual_add_backward2() {
        let dev: TestDevice = Default::default();

        let model = dev.build_module::<f32>(<ResidualAdd2<LinearConstConfig<2, 2>>>::default());

        let x: Tensor<Rank2<4, 2>, f32, _> = dev.sample_normal();
        let x = x.to_dtype::<TestDtype>();
        let y = model.forward(x.leaky_trace());

        #[rustfmt::skip]
        assert_close_to_literal!(y, [[0.25372928, -2.4258814],[1.7892148, -2.6242268],[1.5131638, 0.23407778],[3.4201493, 1.597525]]);

        let g = y.mean().backward();
        assert_close_to_literal!(g.get(&model.t.t.weight), [[0.475242, -0.075136]; 2]);
        assert_close_to_literal!(g.get(&model.t.t.bias), [0.5; 2]);
        assert_close_to_literal!(g.get(&x), [[0.18806472, 0.21419683]; 4]);
    }
}
