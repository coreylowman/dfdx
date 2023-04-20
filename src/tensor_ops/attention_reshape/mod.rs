use crate::{shapes::*, tensor::*};

mod cpu_kernel;
#[cfg(feature = "cuda")]
mod cuda_kernel;

pub type Query<const NUM_HEADS: usize, const HEAD_DIM: usize, E, D> =
    Tensor<(Const<NUM_HEADS>, usize, Const<HEAD_DIM>), E, D>;
pub type Key<const NUM_HEADS: usize, const HEAD_DIM: usize, E, D> =
    Tensor<(Const<NUM_HEADS>, Const<HEAD_DIM>, usize), E, D>;
pub type Value<const NUM_HEADS: usize, const HEAD_DIM: usize, E, D> =
    Tensor<(Const<NUM_HEADS>, usize, Const<HEAD_DIM>), E, D>;

type QkvTuple<const NUM_HEADS: usize, const HEAD_DIM: usize, E, D> = (
    Query<NUM_HEADS, HEAD_DIM, E, D>,
    Key<NUM_HEADS, HEAD_DIM, E, D>,
    Value<NUM_HEADS, HEAD_DIM, E, D>,
);

/// AttentionReshape qkv + past_key + past_value into (q, k, v) used
/// in attention layer
pub trait TryAttentionReshape<E: Dtype>: DeviceStorage {
    /// This is an inference only kernel:
    /// Within `transformers` architecture, a core component is the `attention`
    /// layer, which can be written in many forms.
    ///
    /// This particular version expects a `qkv` tensor (gotten from one single
    /// Linear layer, corresponding of stacked `query`, `key`, `value`.
    /// And `past_key` and `past_value` which are the cached values within attention
    /// (This speeds up inference speed).
    /// For the first pass, just send zero-width tensors when the cache isn't present
    /// already.
    ///
    /// Having a single layer instead of many `cat`, `reshape`, `permute` makes this
    /// operation very efficient on GPU.
    fn attention_reshape<
        const THREE_HIDDEN_DIM: usize,
        const NUM_HEADS: usize,
        const HEAD_DIM: usize,
    >(
        &self,
        qkv: &Tensor<(usize, Const<THREE_HIDDEN_DIM>), E, Self>,
        past_key: &Tensor<(Const<NUM_HEADS>, Const<HEAD_DIM>, usize), E, Self>,
        past_value: &Tensor<(Const<NUM_HEADS>, usize, Const<HEAD_DIM>), E, Self>,
    ) -> QkvTuple<NUM_HEADS, HEAD_DIM, E, Self> {
        self.try_attention_reshape(qkv, past_key, past_value)
            .unwrap()
    }

    /// Fallible version of [TryAttentionReshape::attention_reshape]
    fn try_attention_reshape<
        const THREE_HIDDEN_DIM: usize,
        const NUM_HEADS: usize,
        const HEAD_DIM: usize,
    >(
        &self,
        qkv: &Tensor<(usize, Const<THREE_HIDDEN_DIM>), E, Self>,
        past_key: &Tensor<(Const<NUM_HEADS>, Const<HEAD_DIM>, usize), E, Self>,
        past_value: &Tensor<(Const<NUM_HEADS>, usize, Const<HEAD_DIM>), E, Self>,
    ) -> Result<QkvTuple<NUM_HEADS, HEAD_DIM, E, Self>, Self::Err>;
}

pub trait AttentionReshapeKernel<E: Dtype>: DeviceStorage {
    fn forward<const THREE_HIDDEN_DIM: usize, const NUM_HEADS: usize, const HEAD_DIM: usize>(
        &self,
        qkv: &Tensor<(usize, Const<THREE_HIDDEN_DIM>), E, Self>,
        past_key: &Tensor<(Const<NUM_HEADS>, Const<HEAD_DIM>, usize), E, Self>,
        past_value: &Tensor<(Const<NUM_HEADS>, usize, Const<HEAD_DIM>), E, Self>,
    ) -> Result<QkvTuple<NUM_HEADS, HEAD_DIM, E, Self>, Self::Err>;
}

impl<E: Dtype, D: AttentionReshapeKernel<E>> TryAttentionReshape<E> for D {
    /// Fallible version of [TryAttentionReshape::attention_reshape]
    fn try_attention_reshape<
        const THREE_HIDDEN_DIM: usize,
        const NUM_HEADS: usize,
        const HEAD_DIM: usize,
    >(
        &self,
        qkv: &Tensor<(usize, Const<THREE_HIDDEN_DIM>), E, Self>,
        past_key: &Tensor<(Const<NUM_HEADS>, Const<HEAD_DIM>, usize), E, Self>,
        past_value: &Tensor<(Const<NUM_HEADS>, usize, Const<HEAD_DIM>), E, Self>,
    ) -> Result<QkvTuple<NUM_HEADS, HEAD_DIM, E, Self>, Self::Err> {
        let device = qkv.device.clone();
        device.forward(qkv, past_key, past_value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{tensor_ops::*, tests::*};

    #[test]
    fn test_attention_reshape() {
        let dev: TestDevice = Default::default();

        const NUM_HEADS: usize = 2;
        const HEAD_DIM: usize = 3;
        let sequence_length = 1;
        let past_length = 3;

        let qkv: Tensor<(usize, Const<{ NUM_HEADS * HEAD_DIM * 3 }>), TestDtype, _> =
            dev.zeros_like(&(sequence_length, Const)) + 1.0;
        let past_key: Tensor<(Const<NUM_HEADS>, Const<HEAD_DIM>, usize), TestDtype, _> =
            dev.zeros_like(&(Const, Const, past_length)) + 2.0;
        let past_value: Tensor<(Const<NUM_HEADS>, usize, Const<HEAD_DIM>), TestDtype, _> =
            dev.zeros_like(&(Const, past_length, Const)) + 3.0;

        let (q, k, v) = dev.attention_reshape(&qkv, &past_key, &past_value);

        let q = q
            .realize::<(Const<NUM_HEADS>, Const<1>, Const<HEAD_DIM>)>()
            .unwrap();
        let k = k
            .realize::<(Const<NUM_HEADS>, Const<HEAD_DIM>, Const<4>)>()
            .unwrap();
        let v = v
            .realize::<(Const<NUM_HEADS>, Const<4>, Const<HEAD_DIM>)>()
            .unwrap();

        assert_close_to_literal!(q, [[[1.0; HEAD_DIM]; 1]; NUM_HEADS]);
        assert_close_to_literal!(
            k,
            [
                [
                    [2.0, 2.0, 2.0, 1.0],
                    [2.0, 2.0, 2.0, 1.0],
                    [2.0, 2.0, 2.0, 1.0]
                ],
                [
                    [2.0, 2.0, 2.0, 1.0],
                    [2.0, 2.0, 2.0, 1.0],
                    [2.0, 2.0, 2.0, 1.0]
                ]
            ]
        );
        assert_close_to_literal!(
            v,
            [
                [
                    [3.0, 3.0, 3.0],
                    [3.0, 3.0, 3.0],
                    [3.0, 3.0, 3.0],
                    [1.0, 1.0, 1.0]
                ],
                [
                    [3.0, 3.0, 3.0],
                    [3.0, 3.0, 3.0],
                    [3.0, 3.0, 3.0],
                    [1.0, 1.0, 1.0]
                ]
            ]
        );
    }
}
