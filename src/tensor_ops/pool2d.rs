use super::utils::move_tape_and_add_backward_op;
use crate::arrays::HasArrayData;
use crate::devices::{Cpu, DevicePool2D, PoolAvg, PoolMax, PoolMin};
use crate::gradients::Tape;
use crate::tensor::*;

impl<const C: usize, const H: usize, const W: usize, T: Tape> Tensor3D<C, H, W, T> {
    pub fn avg2d<const K: usize, const S: usize, const P: usize>(
        self,
    ) -> Tensor3D<C, { (H + 2 * P - K) / S + 1 }, { (W + 2 * P - K) / S + 1 }, T> {
        self.pool2d::<PoolAvg, K, S, P>()
    }

    pub fn max2d<const K: usize, const S: usize, const P: usize>(
        self,
    ) -> Tensor3D<C, { (H + 2 * P - K) / S + 1 }, { (W + 2 * P - K) / S + 1 }, T> {
        self.pool2d::<PoolMax, K, S, P>()
    }

    pub fn min2d<const K: usize, const S: usize, const P: usize>(
        self,
    ) -> Tensor3D<C, { (H + 2 * P - K) / S + 1 }, { (W + 2 * P - K) / S + 1 }, T> {
        self.pool2d::<PoolMin, K, S, P>()
    }

    fn pool2d<Pool, const K: usize, const S: usize, const P: usize>(
        self,
    ) -> Tensor3D<C, { (H + 2 * P - K) / S + 1 }, { (W + 2 * P - K) / S + 1 }, T>
    where
        Cpu: DevicePool2D<K, S, P, Pool>,
    {
        let mut result = Tensor3D::zeros();
        Cpu::pool_forward(self.data(), result.mut_data());
        move_tape_and_add_backward_op(self, result, move |x, r, grads| {
            let (xg, rg) = grads.mut_and_ref(&x, &r);
            Cpu::pool_backward(x.data(), rg, xg);
        })
    }
}

impl<const B: usize, const C: usize, const H: usize, const W: usize, T: Tape>
    Tensor4D<B, C, H, W, T>
{
    pub fn avg2d<const K: usize, const S: usize, const P: usize>(
        self,
    ) -> Tensor4D<B, C, { (H + 2 * P - K) / S + 1 }, { (W + 2 * P - K) / S + 1 }, T> {
        self.pool2d::<PoolAvg, K, S, P>()
    }
    pub fn max2d<const K: usize, const S: usize, const P: usize>(
        self,
    ) -> Tensor4D<B, C, { (H + 2 * P - K) / S + 1 }, { (W + 2 * P - K) / S + 1 }, T> {
        self.pool2d::<PoolMax, K, S, P>()
    }
    pub fn min2d<const K: usize, const S: usize, const P: usize>(
        self,
    ) -> Tensor4D<B, C, { (H + 2 * P - K) / S + 1 }, { (W + 2 * P - K) / S + 1 }, T> {
        self.pool2d::<PoolMin, K, S, P>()
    }

    fn pool2d<Pool: 'static, const K: usize, const S: usize, const P: usize>(
        self,
    ) -> Tensor4D<B, C, { (H + 2 * P - K) / S + 1 }, { (W + 2 * P - K) / S + 1 }, T>
    where
        Cpu: DevicePool2D<K, S, P, Pool>,
    {
        let mut result = Tensor4D::zeros();
        for (x_i, r_i) in self.data().iter().zip(result.mut_data().iter_mut()) {
            Cpu::pool_forward(x_i, r_i);
        }
        let (x, mut tape) = self.split_tape();
        let r = result.phantom();
        tape.add_backward_op(move |grads| {
            let (xg, rg) = grads.mut_and_ref(&x, &r);
            for ((x_i, rg_i), xg_i) in x.data().iter().zip(rg.iter()).zip(xg.iter_mut()) {
                Cpu::pool_backward(x_i, rg_i, xg_i);
            }
        });
        result.put_tape(tape)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{tensor_ops::backward, tests::assert_close};
    use rand::{rngs::StdRng, SeedableRng};

    #[test]
    fn test_3d_max2d() {
        let mut rng = StdRng::seed_from_u64(234);
        let x: Tensor3D<2, 3, 4> = TensorCreator::randn(&mut rng);
        let r = x.trace().max2d::<2, 2, 0>();
        assert_close(
            r.data(),
            &[[[1.79155397, 1.10126066]], [[1.14464748, 2.26301837]]],
        );
        let g = backward(r.exp().mean());
        #[rustfmt::skip]
        assert_close(
            g.ref_gradient(&x),
            &[
                [[1.49969184, 0., 0., 0.75198889],[0., 0., 0., 0.],[0., 0., 0., 0.]],
                [[0., 0., 2.40301466, 0.],[0.78533345, 0., 0., 0.],[0., 0., 0., 0.]]
            ]
        );
    }

    #[test]
    fn test_3d_min2d() {
        let mut rng = StdRng::seed_from_u64(234);
        let x: Tensor3D<2, 3, 4> = TensorCreator::randn(&mut rng);
        let r = x.trace().min2d::<2, 2, 0>();
        assert_close(
            r.data(),
            &[[[-1.09635627, -1.07717276]], [[-0.01996479, -1.82562149]]],
        );
        let g = backward(r.exp().mean());
        #[rustfmt::skip]
        assert_close(
            g.ref_gradient(&x),
            &[
                [[0., 0., 0., 0.],[0.083521545, 0., 0., 0.08513925],[0., 0., 0., 0.]],
                [[0., 0.2450583, 0., 0.04027937],[0., 0., 0., 0.],[0., 0., 0., 0.]],
            ],
        );
    }

    #[test]
    fn test_3d_avg2d() {
        let mut rng = StdRng::seed_from_u64(234);
        let x: Tensor3D<2, 3, 4> = TensorCreator::randn(&mut rng);
        let r = x.trace().avg2d::<2, 2, 0>();
        assert_close(
            r.data(),
            &[[[0.03031558, -0.25052455]], [[0.39499030, 0.04878314]]],
        );
        let g = backward(r.exp().mean());
        #[rustfmt::skip]
        assert_eq!(
            g.ref_gradient(&x),
            &[
                [[0.06442373, 0.06442373, 0.048649523, 0.048649523],[0.06442373, 0.06442373, 0.048649523, 0.048649523],[0.0, 0.0, 0.0, 0.0]],
                [[0.09277311, 0.09277311, 0.06562454, 0.06562454],[0.09277311, 0.09277311, 0.06562454, 0.06562454],[0.0, 0.0, 0.0, 0.0]]
            ]
        );
    }
}
