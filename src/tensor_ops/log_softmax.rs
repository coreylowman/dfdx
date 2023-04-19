use super::*;
use crate::{shapes::*, tensor::*};

/// `log(softmax(t))` in numerically stable way across `Ax`. Does `t - logsumexp(t)` under the hood.
///
/// **Pytorch equivalent**: `t.log_softmax(Ax)`
///
/// Example:
/// ```rust
/// # use dfdx::prelude::*;
/// # let dev: Cpu = Default::default();
/// let t: Tensor<Rank3<2, 3, 5>, f32, _> = dev.zeros();
/// let _ = t.log_softmax::<Axis<2>>();
/// ```
///
/// Using multi axis log_softmax:
/// ```rust
/// # use dfdx::prelude::*;
/// # let dev: Cpu = Default::default();
/// # let t: Tensor<Rank3<2, 3, 5>, f32, _> = dev.zeros();
/// let _ = t.log_softmax::<Axes2<0, 2>>();
/// ```
pub fn log_softmax<Ax: Axes, S: Shape, E: Dtype, D: Device<E>, T: Tape<E, D>>(
    t: Tensor<S, E, D, T>,
) -> Tensor<S, E, D, T>
where
    S: ReduceShape<Ax>,
{
    t.log_softmax::<Ax>()
}

impl<S: Shape, E: Dtype, D: Device<E>, T: Tape<E, D>> Tensor<S, E, D, T> {
    /// See [log_softmax]
    pub fn log_softmax<Ax: Axes>(self) -> Self
    where
        S: ReduceShape<Ax>,
    {
        self.try_log_softmax::<Ax>().unwrap()
    }
    /// See [log_softmax]
    pub fn try_log_softmax<Ax: Axes>(self) -> Result<Self, D::Err>
    where
        S: ReduceShape<Ax>,
    {
        /*
        # Notes on this reduction

        log_softmax is equivalent to:
            `t - t.logsumexp()`

        logsumexp can be inlined to:
            `t - ((t - t.max()).exp().sum().ln() + t.max())`

        we can apply the subtraction in the following way:
            `t - (t - t.max()).exp().sum().ln() - t.max()`
            `t - t.max() - (t - t.max()).exp().sum().ln()`

        Notice there is a repeated expression here of `t - t.max()`.
        So we can re-use this calculation.
            `tm - tm.exp().sum().ln()`
        */
        let shape = *self.shape();
        let (t, tape) = self.split_tape();
        let max = t.clone().try_max::<_, Ax>()?;
        let tm = {
            // Do this calculation off of the tape
            let keep_id = t.id;
            let mut t = t.try_sub(max.try_broadcast_like::<_, Ax>(&shape)?)?;
            t.id = keep_id;
            t.put_tape(tape)
        };
        let logsumexp = tm.retaped::<T>().try_exp()?.try_sum::<_, Ax>()?.try_ln()?;
        tm.try_sub(logsumexp.try_broadcast_like(&shape)?)
    }
}

#[cfg(test)]
mod tests {
    use crate::{shapes::*, tensor::*, tensor_ops::*, tests::*};

    #[test]
    fn test_log_softmax_equivalence() {
        let dev: TestDevice = Default::default();
        let t: Tensor<Rank4<8, 16, 32, 64>, TestDtype, _> = dev.sample_normal();
        let p = t.leaky_trace().log_softmax::<Axis<3>>();
        let p_truth = t.leaky_trace() - t.leaky_trace().logsumexp::<_, Axis<3>>().broadcast();
        // we can't create an array as it will overflow the stack
        for (p_i, pt_i) in p.as_vec().iter().zip(p_truth.as_vec().iter()) {
            assert!((p_i - pt_i).abs() <= TestDtype::DEFAULT_TOLERANCE);
        }
        let g = p.square().mean().backward();
        let g_truth = p_truth.square().mean().backward();
        for (g_i, gt_i) in g
            .get(&t)
            .as_vec()
            .iter()
            .zip(g_truth.get(&t).as_vec().iter())
        {
            assert!((g_i - gt_i).abs() <= TestDtype::DEFAULT_TOLERANCE);
        }
    }

    #[test]
    fn test_log_softmax_1d() {
        let dev: TestDevice = Default::default();
        let a: Tensor<_, TestDtype, _> = dev.tensor([-2.0, -1.0, 0.0, 1.0, 2.0]);
        let r = a.leaky_trace().log_softmax();
        assert_close_to_literal!(
            r,
            [-4.4519143, -3.4519143, -2.4519143, -1.4519143, -0.4519143]
        );
        let g = r.mean().backward();
        assert_close_to_literal!(
            g.get(&a),
            [
                0.18834378,
                0.16831508,
                0.11387146,
                -0.034121647,
                -0.43640864,
            ]
        );
    }

    #[test]
    fn test_log_softmax_2d() {
        let dev: TestDevice = Default::default();
        let a: Tensor<_, TestDtype, _> = dev.tensor([[-2.0, -1.0, 0.0], [1.0, 4.0, 7.0]]);
        let r = a.leaky_trace().log_softmax::<Axis<1>>();
        assert_close_to_literal!(
            r,
            [
                [-2.407606, -1.4076059, -0.40760595],
                [-6.0509458, -3.0509458, -0.05094576],
            ]
        );
        let g = r.mean().backward();
        assert_close_to_literal!(
            g.get(&a),
            [
                [0.12165138, 0.044302434, -0.1659538],
                [0.16548885, 0.14300959, -0.30849844],
            ]
        );
    }
}
