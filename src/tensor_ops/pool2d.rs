use super::utils::move_tape_and_add_backward_op;
use crate::arrays::HasArrayData;
use crate::devices::{Cpu, DevicePool2D, PoolAvg, PoolMax, PoolMin};
use crate::gradients::Tape;
use crate::tensor::*;

/// **Requires nightly** Performs a 2d max pool
pub fn max2d<
    T: Tape,
    const C: usize,
    const K: usize,
    const S: usize,
    const P: usize,
    const H: usize,
    const W: usize,
>(
    x: Tensor3D<C, H, W, T>,
) -> Tensor3D<C, { (H + 2 * P - K) / S + 1 }, { (W + 2 * P - K) / S + 1 }, T> {
    pool2d::<PoolMax, T, C, K, S, P, H, W>(x)
}

/// **Requires nightly** Performs a batched 2d max pool
pub fn max2d_batched<
    T: Tape,
    const B: usize,
    const C: usize,
    const K: usize,
    const S: usize,
    const P: usize,
    const H: usize,
    const W: usize,
>(
    x: Tensor4D<B, C, H, W, T>,
) -> Tensor4D<B, C, { (H + 2 * P - K) / S + 1 }, { (W + 2 * P - K) / S + 1 }, T> {
    pool2d_batched::<PoolMax, T, B, C, K, S, P, H, W>(x)
}

/// **Requires nightly** Performs a 2d min pool
pub fn min2d<
    T: Tape,
    const C: usize,
    const K: usize,
    const S: usize,
    const P: usize,
    const H: usize,
    const W: usize,
>(
    x: Tensor3D<C, H, W, T>,
) -> Tensor3D<C, { (H + 2 * P - K) / S + 1 }, { (W + 2 * P - K) / S + 1 }, T> {
    pool2d::<PoolMin, T, C, K, S, P, H, W>(x)
}

/// **Requires nightly** Performs a batched 2d min pool
pub fn min2d_batched<
    T: Tape,
    const B: usize,
    const C: usize,
    const K: usize,
    const S: usize,
    const P: usize,
    const H: usize,
    const W: usize,
>(
    x: Tensor4D<B, C, H, W, T>,
) -> Tensor4D<B, C, { (H + 2 * P - K) / S + 1 }, { (W + 2 * P - K) / S + 1 }, T> {
    pool2d_batched::<PoolMin, T, B, C, K, S, P, H, W>(x)
}

/// **Requires nightly** Performs a 2d avg pool
pub fn avg2d<
    T: Tape,
    const C: usize,
    const K: usize,
    const S: usize,
    const P: usize,
    const H: usize,
    const W: usize,
>(
    x: Tensor3D<C, H, W, T>,
) -> Tensor3D<C, { (H + 2 * P - K) / S + 1 }, { (W + 2 * P - K) / S + 1 }, T> {
    pool2d::<PoolAvg, T, C, K, S, P, H, W>(x)
}

/// **Requires nightly** Performs a batched 2d avg pool
pub fn avg2d_batched<
    T: Tape,
    const B: usize,
    const C: usize,
    const K: usize,
    const S: usize,
    const P: usize,
    const H: usize,
    const W: usize,
>(
    x: Tensor4D<B, C, H, W, T>,
) -> Tensor4D<B, C, { (H + 2 * P - K) / S + 1 }, { (W + 2 * P - K) / S + 1 }, T> {
    pool2d_batched::<PoolAvg, T, B, C, K, S, P, H, W>(x)
}

fn pool2d<
    Pool,
    T: Tape,
    const C: usize,
    const K: usize,
    const S: usize,
    const P: usize,
    const H: usize,
    const W: usize,
>(
    x: Tensor3D<C, H, W, T>,
) -> Tensor3D<C, { (H + 2 * P - K) / S + 1 }, { (W + 2 * P - K) / S + 1 }, T>
where
    Cpu: DevicePool2D<K, S, P, Pool>,
{
    let mut result = Tensor3D::zeros();
    Cpu::pool_forward(x.data(), result.mut_data());
    move_tape_and_add_backward_op(x, result, move |x, r, grads| {
        let (xg, rg) = grads.mut_and_ref(&x, &r);
        Cpu::pool_backward(x.data(), rg, xg);
    })
}

fn pool2d_batched<
    Pool: 'static,
    T: Tape,
    const B: usize,
    const C: usize,
    const K: usize,
    const S: usize,
    const P: usize,
    const H: usize,
    const W: usize,
>(
    x: Tensor4D<B, C, H, W, T>,
) -> Tensor4D<B, C, { (H + 2 * P - K) / S + 1 }, { (W + 2 * P - K) / S + 1 }, T>
where
    Cpu: DevicePool2D<K, S, P, Pool>,
{
    let mut result = Tensor4D::zeros();
    for (x_i, r_i) in x.data().iter().zip(result.mut_data().iter_mut()) {
        Cpu::pool_forward(x_i, r_i);
    }
    let (x, mut tape) = x.split_tape();
    let r = result.phantom();
    tape.add_backward_op(move |grads| {
        let (xg, rg) = grads.mut_and_ref(&x, &r);
        for ((x_i, rg_i), xg_i) in x.data().iter().zip(rg.iter()).zip(xg.iter_mut()) {
            Cpu::pool_backward(x_i, rg_i, xg_i);
        }
    });
    result.put_tape(tape)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_max_pool() {
        todo!();
    }

    #[test]
    fn test_min_pool() {
        todo!();
    }

    #[test]
    fn test_avg_pool() {
        todo!();
    }
}
