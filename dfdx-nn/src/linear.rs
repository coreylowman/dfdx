use crate::*;

use dfdx::prelude::*;

/// A linear transformation of the form `weight * x + bias`, where `weight` is a matrix, `x` is a vector or matrix,
/// and `bias` is a vector.
///
/// Generics:
/// - `I` The "input" size of vectors & matrices.
/// - `O` The "output" size of vectors & matrices.
///
/// Example:
/// ```rust
/// # use dfdx::prelude::*;
/// # use dfdx_nn::*;
/// # let dev: Cpu = Default::default();
/// type Model = LinearConstConfig<5, 2>;
/// let model = dev.build_module::<f32>(Model::default());
/// // single item forward
/// let _: Tensor<Rank1<2>, f32, _> = model.forward(dev.zeros::<Rank1<5>>());
/// // batched forward
/// let _: Tensor<Rank2<10, 2>, f32, _> = model.forward(dev.zeros::<Rank2<10, 5>>());
/// ```
#[derive(Default, Debug, Clone, Copy, Sequential)]
#[built(Linear)]
pub struct LinearConfig<I: Dim, O: Dim> {
    pub matmul: MatMulConfig<I, O>,
    pub add: Bias1DConfig<O>,
}

/// Compile time sugar alias around [LinearConfig].
pub type LinearConstConfig<const I: usize, const O: usize> = LinearConfig<Const<I>, Const<O>>;

impl<I: Dim, O: Dim> LinearConfig<I, O> {
    pub fn new(inp: I, out: O) -> Self {
        Self {
            matmul: MatMulConfig { inp, out },
            add: Bias1DConfig(out),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::*;

    const W: [[f64; 5]; 2] = [
        [-0.3458893, -0.30371523, -0.3712057, 0.14303583, -0.0268966],
        [0.11733949, 0.14059687, -0.10670426, -0.09373143, 0.18974298],
    ];
    const B: [f64; 2] = [0.3765365, -0.290717];

    #[test]
    fn test_forward_1d() {
        let dev: TestDevice = Default::default();

        let model = Linear {
            matmul: MatMul {
                weight: dev.tensor(W).to_dtype::<TestDtype>(),
            },
            add: Bias1D {
                bias: dev.tensor(B).to_dtype::<TestDtype>(),
            },
        };

        let x = dev
            .tensor([-0.8808001, 2.4185333, 2.2478335, 0.0565211, 2.031299])
            .to_dtype::<TestDtype>();
        let y = model.forward(x.leaky_trace());
        assert_close_to_literal!(y, [-0.93430865, 0.08624211]);

        let g = y.square().mean().backward();
        assert_close_to_literal!(
            g.get(&model.matmul.weight),
            [
                [0.82293916, -2.2596567, -2.1001704, -0.05280815, -1.8978603],
                [-0.07596206, 0.20857942, 0.19385791, 0.004874499, 0.17518352],
            ]
        );
        assert_close_to_literal!(g.get(&model.add.bias), [-0.93430865, 0.08624211]);
    }

    #[test]
    fn test_forward_2d() {
        let dev: TestDevice = Default::default();

        let model = Linear {
            matmul: MatMul {
                weight: dev.tensor(W).to_dtype::<TestDtype>(),
            },
            add: Bias1D {
                bias: dev.tensor(B).to_dtype::<TestDtype>(),
            },
        };

        let x = dev
            .tensor([
                [-1.9468665, 1.4611785, -1.6698982, 1.408863, 1.3425643],
                [-1.3399831, 3.0510678, -0.17936817, -0.04943254, -0.8052705],
                [-0.8291412, 0.07691376, -0.26538327, 0.90017676, -1.8790455],
            ])
            .to_dtype::<TestDtype>();
        let y = model.forward(x.leaky_trace());
        assert_close_to_literal!(
            y,
            [
                [1.3914378, -0.012851536],
                [-0.005462587, -0.14800104],
                [0.9177769, -0.7897872],
            ]
        );

        let g = y.square().mean().backward();
        assert_close_to_literal!(
            g.get(&model.matmul.weight),
            [
                [-1.1541969, 0.6956873, -0.8553807, 0.9289255, 0.04931633],
                [0.29272807, -0.17702839, 0.08586791, -0.24057935, 0.5286576],
            ]
        );
        assert_close_to_literal!(g.get(&model.add.bias), [0.7679174, -0.31687993]);
    }

    #[test]
    fn test_forward_3d() {
        let dev: TestDevice = Default::default();

        let model = Linear {
            matmul: MatMul {
                weight: dev.tensor(W).to_dtype::<TestDtype>(),
            },
            add: Bias1D {
                bias: dev.tensor(B).to_dtype::<TestDtype>(),
            },
        };

        #[rustfmt::skip]
        let x = dev.tensor([
            [[-1.9468665, 1.4611785, -1.6698982, 1.408863, 1.3425643], [-1.3399831, 3.0510678, -0.17936817, -0.04943254, -0.8052705], [-0.8291412, 0.07691376, -0.26538327, 0.90017676, -1.8790455]],
            [[1.2879219, 0.70150787, -1.6746868, 1.7261779, -0.94021803], [-2.6883178, 2.9369607, 2.9256766, 0.27559614, -0.17530347], [0.17499207, -0.11440835, 0.16627812, -0.91773695, 1.1128315]],
        ]).to_dtype::<TestDtype>();
        let y = model.forward(x.leaky_trace());
        assert_close_to_literal!(
            y,
            [
                [
                    [1.3914378, -0.012851536],
                    [-0.005462587, -0.14800104],
                    [0.9177769, -0.7897872],
                ],
                [
                    [0.6118467, -0.2024638],
                    [-0.6274954, -0.56451213],
                    [0.12783213, -0.0068387985],
                ],
            ]
        );

        let g = y.square().mean().backward();
        #[rustfmt::skip]
        assert_close_to_literal!(
            g.get(&model.matmul.weight),
            [[-0.16088384, 0.10978711, -0.9008978, 0.59211355, -0.029177088], [0.35563633, -0.38838047, -0.17600831, -0.2034213, 0.31128058]]
        );
        assert_close_to_literal!(g.get(&model.add.bias), [0.40265593, -0.2874091]);
    }
}
